import os, time, torch, numpy as np
from model_baseline import Graph_MLP
import data_loader
from torch.utils.data import DataLoader
from random import shuffle

EPOCHS, LR, BATCH, DECAY = 60, 1e-4, 64, 0.96
data_root = "../data/sphere_5stiff/"
weight_dir = "./weight_baseline_retrained/"
os.makedirs(weight_dir, exist_ok=True)

net = Graph_MLP().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

train_files = sorted([os.path.join(data_root, "train", f) for f in os.listdir(data_root + "/train") if f.startswith("motion_")])
test_files = sorted([os.path.join(data_root, "test", f) for f in os.listdir(data_root + "/test") if f.startswith("motion_")])
print("Train: %d, Test: %d" % (len(train_files), len(test_files)))

best_test = float("inf")
for epoch in range(EPOCHS):
    lr = LR * (DECAY ** epoch)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    net.train()
    tloss, tc = 0.0, 0
    t0 = time.time()
    shuffle(train_files)
    for mp in train_files:
        fn = len([f for f in os.listdir(mp + "/1/") if f.startswith("x_")]) - 1
        ds = data_loader.MeshDataset(mp, 5, fn)
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
            tloss += loss.item()
            tc += 1
    et = time.time() - t0
    at = tloss / max(tc, 1)

    net.eval()
    vloss, vc = 0.0, 0
    with torch.no_grad():
        for mp in test_files:
            fn = len([f for f in os.listdir(mp + "/1/") if f.startswith("x_")]) - 1
            ds = data_loader.MeshDataset(mp, 2, fn)
            dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2)
            for c, dyn, ref, adj, k, m, gt in dl:
                dp = torch.cat([dyn[:, 0], dyn[:, 1], dyn[:, 2]], dim=-1).cuda()
                rp = torch.cat([ref[:, 0], ref[:, 1], ref[:, 2]], dim=-1).cuda()
                cb = c[0].cuda()
                pred = net(cb, dp, rp, adj[0], k.cuda(), m.cuda())
                tgt = gt.cuda()[:, cb == 0, :]
                loss = ((pred - tgt) ** 2).mean()
                vloss += loss.item()
                vc += 1
    av = vloss / max(vc, 1)
    mk = " *best*" if av < best_test else ""
    if av < best_test:
        best_test = av
        torch.save(net.state_dict(), weight_dir + "best.weight")
    print("[%d] lr=%.6f train=%.6f test=%.6f time=%.1fs%s" % (epoch, lr, at, av, et, mk))
    torch.save(net.state_dict(), weight_dir + "epoch_%04d.weight" % epoch)

print("Best test: %.6f" % best_test)
print("Baseline training complete.")
