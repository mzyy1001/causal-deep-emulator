"""Train widened baseline (~531K params) for param-matched comparison."""
import os, time, torch, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from random import shuffle
import data_loader


class WideGraphMLP(nn.Module):
    """Widened Deep Emulator baseline to match our model's param count."""
    def __init__(self):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(37, 192), nn.Tanh(),
            nn.Linear(192, 192), nn.Tanh(),
            nn.Linear(192, 192), nn.Tanh(),
            nn.Linear(192, 192), nn.Tanh(),
            nn.Linear(192, 192))
        self.point_mlp = nn.Sequential(
            nn.Linear(18, 96), nn.Tanh(),
            nn.Linear(96, 96), nn.Tanh(),
            nn.Linear(96, 96), nn.Tanh(),
            nn.Linear(96, 96), nn.Tanh(),
            nn.Linear(96, 96))
        self.instance_mlp = nn.Sequential(
            nn.Linear(288, 288), nn.Tanh(),
            nn.Linear(288, 288), nn.Tanh(),
            nn.Linear(288, 288), nn.Tanh(),
            nn.Linear(288, 288), nn.Tanh(),
            nn.Linear(288, 3))

    def forward(self, constraint, dynamic_f, reference_f, adj_matrix, stiffness, mass):
        dev = dynamic_f.device
        adj_matrix = adj_matrix.to(dev)
        constraint = constraint.to(dev)
        mask = torch.ones(adj_matrix.shape[0], adj_matrix.shape[1], device=dev)
        mask[adj_matrix == 0] = 0.0
        mask = mask.unsqueeze(0).unsqueeze(3).expand(dynamic_f.shape[0], -1, -1, -1)

        pfu = dynamic_f[:, :, 0:3] - reference_f[:, :, 3:6]
        pfp = dynamic_f[:, :, 3:6] - reference_f[:, :, 3:6]
        pfpp = dynamic_f[:, :, 6:9] - reference_f[:, :, 3:6]
        pfxn = reference_f[:, :, 0:3] - reference_f[:, :, 3:6]
        pfxp = reference_f[:, :, 6:9] - reference_f[:, :, 3:6]
        pkm = stiffness / mass
        pf = torch.cat((pfu, pfp, pfpp, pfxn, pfxp, stiffness, mass, pkm), 2)

        efc = constraint[adj_matrix].unsqueeze(0).unsqueeze(3).expand(dynamic_f.shape[0], -1, -1, -1)
        rc = reference_f[:, :, 3:6].unsqueeze(2).expand(-1, -1, adj_matrix.shape[1], -1)
        nfu = dynamic_f[:, adj_matrix, 0:3] - rc
        nfp = dynamic_f[:, adj_matrix, 3:6] - rc
        nfpp = dynamic_f[:, adj_matrix, 6:9] - rc
        nfxn = reference_f[:, adj_matrix, 0:3] - rc
        nfxc = reference_f[:, adj_matrix, 3:6] - rc
        nfxp = reference_f[:, adj_matrix, 6:9] - rc
        nf = torch.cat((nfu, nfp, nfpp, nfxn, nfxc, nfxp), 3)
        ef = torch.cat((efc.float(), nf, pf.unsqueeze(2).expand(-1, -1, adj_matrix.shape[1], -1)), 3)

        pf = pf[:, constraint == 0, :]
        ef = ef[:, constraint == 0, :, :]
        mask = mask[:, constraint == 0, :, :]

        oe = self.edge_mlp(ef) * mask
        oe = torch.sum(oe, 2)
        op = self.point_mlp(pf)
        ii = torch.cat((op, oe), 2)
        return self.instance_mlp(ii)


if __name__ == '__main__':
    EPOCHS, LR, BATCH, DECAY = 60, 1e-4, 64, 0.96
    data_root = '../data/sphere_all/'
    weight_dir = './weight_wide_baseline/'
    os.makedirs(weight_dir, exist_ok=True)

    net = WideGraphMLP().cuda()
    print('Wide baseline params:', sum(p.numel() for p in net.parameters()))
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    train_files = sorted([os.path.join(data_root, 'train', f)
                          for f in os.listdir(data_root + '/train') if f.startswith('motion_')])
    test_files = sorted([os.path.join(data_root, 'test', f)
                         for f in os.listdir(data_root + '/test') if f.startswith('motion_')])
    print('Train: %d, Test: %d' % (len(train_files), len(test_files)))

    best_test = float('inf')
    for epoch in range(EPOCHS):
        lr = LR * (DECAY ** epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        net.train()
        tl, tc = 0.0, 0
        t0 = time.time()
        shuffle(train_files)
        for mp in train_files:
            fn = len([f for f in os.listdir(mp + '/1/') if f.startswith('x_')]) - 1
            ds = data_loader.MeshDataset(mp, 7, fn)
            dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2)
            for c, dyn, ref, adj, k, m, gt in dl:
                dp = torch.cat([dyn[:, 0], dyn[:, 1], dyn[:, 2]], dim=-1).cuda()
                rp = torch.cat([ref[:, 0], ref[:, 1], ref[:, 2]], dim=-1).cuda()
                cb = c[0].cuda()
                pred = net(cb, dp, rp, adj[0], k.cuda(), m.cuda())
                tgt = gt.cuda()[:, cb == 0, :]
                loss = ((pred - tgt) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tl += loss.item()
                tc += 1
        et = time.time() - t0
        net.eval()
        vl, vc = 0.0, 0
        with torch.no_grad():
            for mp in test_files:
                fn = len([f for f in os.listdir(mp + '/1/') if f.startswith('x_')]) - 1
                ds = data_loader.MeshDataset(mp, 7, fn)
                dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2)
                for c, dyn, ref, adj, k, m, gt in dl:
                    dp = torch.cat([dyn[:, 0], dyn[:, 1], dyn[:, 2]], dim=-1).cuda()
                    rp = torch.cat([ref[:, 0], ref[:, 1], ref[:, 2]], dim=-1).cuda()
                    cb = c[0].cuda()
                    pred = net(cb, dp, rp, adj[0], k.cuda(), m.cuda())
                    tgt = gt.cuda()[:, cb == 0, :]
                    loss = ((pred - tgt) ** 2).mean()
                    vl += loss.item()
                    vc += 1
        av = vl / max(vc, 1)
        mk = ' *best*' if av < best_test else ''
        if av < best_test:
            best_test = av
            torch.save(net.state_dict(), weight_dir + 'best.weight')
        print('[%d] lr=%.6f train=%.6f test=%.6f time=%.1fs%s' % (
            epoch, lr, tl / max(tc, 1), av, et, mk))
    print('Best test: %.6f' % best_test)
    print('Wide baseline complete.')
