"""Stages 1 and 2 of docs/briefs/colab-parity.md.

    python colab/test_parity.py <bundle> [<cache_out>]

Stage 1 checks the port against fixtures run_export wrote from the MATLAB code.
Stage 2 runs every fold's MATLAB weights through the PyTorch network, compares P
with the cached MATLAB P, and writes it to <cache_out> for run_cache to score.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rnvtorch as rt


def stage1(root, cfg, images, folds):
    fx = sio.loadmat(Path(root) / "fixtures.mat", squeeze_me=False)
    p, sigma = int(cfg["patch"]), float(cfg["sigmaSeed"])

    chans = fx["patchChans"]
    chans = chans[:, :, None] if chans.ndim == 2 else chans
    for s in range(fx["patchSeed"].shape[0]):
        X, rows, cols = rt.seed_patch(chans, fx["patchSeed"][s], fx["patchCenter"][s], p, sigma)
        assert np.array_equal(rows, fx["patchRows"][s]) and np.array_equal(cols, fx["patchCols"][s]), f"crop geometry, case {s}"
        assert np.array_equal(X[:, :, :-1], fx["patchX"][:, :, :-1, s]), f"crop values, case {s}"
        assert np.abs(X[:, :, -1] - fx["patchX"][:, :, -1, s]).max() <= 1e-6, f"heat-map, case {s}"
    print(f"  seedPatch      ok ({s + 1} cases, edges and jitter)")

    A = fx["dihA"]
    assert np.array_equal(np.fliplr(A), fx["dihFlipLR"]) and np.array_equal(np.flipud(A), fx["dihFlipUD"])
    for r in (1, 2, 3):
        assert np.array_equal(np.rot90(A, r), fx[f"dihRot{r}"]), f"rot90 {r}"
    print("  dihedral       ok")

    Y = torch.from_numpy(fx["gdY"]).permute(3, 2, 0, 1).double()
    T = torch.from_numpy(fx["gdT"]).permute(3, 2, 0, 1).double()
    gd = rt.generalized_dice(Y, T).numpy()
    want = fx["gd"].ravel()
    assert np.allclose(gd, want, rtol=1e-6, atol=1e-9), f"generalizedDice {gd} vs {want}"
    loss = rt.loss_fn(Y, T).item()
    assert np.isclose(loss, fx["gdLoss"].item(), atol=1e-6, equal_nan=True), f"loss {loss} vs {fx['gdLoss'].item()}"
    print(f"  loss           ok (per observation {np.round(gd, 6)})")

    for f, fold in enumerate(folds, 1):
        train, val = rt.training_rows(cfg, images, fold)
        n = len(rt.epoch_rows(cfg, train, np.random.default_rng(0)))
        assert (len(train), len(val), n) == (fold["trainSeeds"], fold["valSeeds"], fold["patchesPerEpoch"]), \
            f"fold {f}: seeds/val/patches {len(train)}/{len(val)}/{n} vs MATLAB {fold['trainSeeds']}/{fold['valSeeds']}/{fold['patchesPerEpoch']}"
    print(f"  patch counts   ok ({len(folds)} folds match MATLAB's info)")
    return fx


def stage2(cfg, images, folds, fx, cache_out, device):
    rt.exact_fp32()
    p, sigma, batch = int(cfg["patch"]), float(cfg["sigmaSeed"]), int(cfg["inferBatch"])
    worst = 0.0
    for f, fold in enumerate(folds, 1):
        model = rt.UNet().to(device).load_matlab(fold).eval()
        if f == 1:
            X = torch.from_numpy(fx["fwdX"][None].transpose(0, 3, 1, 2).copy()).to(device)
            with torch.no_grad():
                Y = model(X)[0].permute(1, 2, 0).cpu().numpy()
            d = np.abs(Y - fx["fwdY"]).max()
            assert d <= 1e-4, f"forward pass differs from MATLAB by {d:.3g}"
            print(f"  forward pass   ok (max |dY| {d:.2g})")
        for name in fold["test"]:
            im = images[name]
            P = rt.infer_prob_map(im["chans"], im["seeds"], model, p, sigma, batch, device)
            if im["refP"] is not None:
                assert im["refFold"] == f, f"{name}: cached MATLAB P is from fold {im['refFold']}"
                worst = max(worst, np.abs(P - im["refP"]).max())
            if cache_out:
                rt.write_prob_cache(cache_out, name, P, f, arm=str(cfg["arm"]),
                                    netFile=f"MATLAB fold {f} weights in PyTorch")
        print(f"  fold {f}         {len(fold['test'])} images, running max |dP| {worst:.2g}")
    assert worst <= 1e-3, f"P differs from the MATLAB cache by {worst:.3g}"
    print(f"  inference      ok (max |dP| {worst:.2g} over every image)")


if __name__ == "__main__":
    root = sys.argv[1]
    cache_out = sys.argv[2] if len(sys.argv) > 2 else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, images, folds = rt.load_bundle(root)
    print(f"test_parity | {len(images)} images, {len(folds)} folds | {device}")
    print("stage 1: fixtures")
    fx = stage1(root, cfg, images, folds)
    print("stage 2: MATLAB weights")
    stage2(cfg, images, folds, fx, cache_out, device)
    print("All checks passed.")
