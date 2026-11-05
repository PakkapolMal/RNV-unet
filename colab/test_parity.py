"""Stages 1 and 2 of docs/briefs/colab-parity.md.

    python colab/test_parity.py <bundle> [<cache_out>]

Stage 1 checks the port against fixtures run_export wrote from the MATLAB code.
Stage 2 runs every fold's MATLAB weights through the PyTorch network, checks one
patch and one whole-image P against MATLAB's CPU, reports the distance to the
cached (GPU) MATLAB P, and writes P to <cache_out> for run_cache to score.
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
            assert d <= 1e-5, f"forward pass differs from MATLAB by {d:.3g}"
            print(f"  forward pass   ok (max |dY| {d:.2g})")
            im = images[str(fx["fwdImage"][0])]
            P = rt.infer_prob_map(im["chans"], im["seeds"], model, p, sigma, batch, device)
            d = np.abs(P - fx["fwdP"]).max()
            assert d <= 1e-5, f"P differs from MATLAB's CPU P by {d:.3g}"
            print(f"  whole-image P  ok (max |dP| {d:.2g} vs MATLAB CPU, {fx['fwdImage'][0]})")
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
    # Not asserted: the cache came from MATLAB's GPU, a few 1e-4 from its own CPU
    # (docs/results/2026-11-02-colab-parity-inference.md). run_cache judges it.
    print(f"  vs GPU cache   max |dP| {worst:.2g}; score <cache_out> with run_cache")


def stage3(root, run_root, workers, budget=False):
    """Fold 1: train 1 epoch, stop, resume to epoch 2, write P.

    budget runs the options of docs/briefs/train-budget.md (cosine, last epoch,
    inner images training) and checks the schedule survives the resume.
    """
    import csv
    import math
    cfg, images, folds = rt.load_bundle(root)
    fold = folds[0]
    opts = dict(lr_schedule="cosine", output="last", inner_val=False) if budget else {}
    assert rt.train_fold(root, run_root, 1, 0, epochs=2, stop_after=1, workers=workers, **opts) is None
    assert rt.train_fold(root, run_root, 1, 0, epochs=2, workers=workers, **opts) is not None
    log = list(csv.DictReader(open(Path(run_root) / "fold1" / "log.csv")))
    assert [int(r["epoch"]) for r in log] == [1, 2], f"epochs logged {[r['epoch'] for r in log]}"
    st = torch.load(Path(run_root) / "fold1" / "net.pt", weights_only=False)
    ck = torch.load(Path(run_root) / "fold1" / "ckpt.pt", weights_only=False)
    if budget:
        assert st["epoch"] == 2, f"net.pt holds epoch {st['epoch']}, not the last"
        assert all(r["val_loss"] == "" for r in log), "a validation loss was logged without inner images"
        want = [1e-5 + (1e-3 - 1e-5) * (1 + math.cos(math.pi * e / 2)) / 2 for e in (0, 1)]
        got = [float(r["lr"]) for r in log]
        assert np.allclose(got, want, rtol=1e-9), f"learning rates {got}, cosine wants {want}"
        # Patch count, from the bundle directly: every non-test image with a
        # usable seed trains, and the budget is shared equally across them.
        held = set(fold["test"])
        n_img = [n for n, im in images.items() if n not in held and (im["comp"] > 0).any()]
        n_seed = sum(int((images[n]["comp"] > 0).sum()) for n in n_img)
        assert n_seed == fold["trainSeeds"] + fold["valSeeds"], "inner seeds did not join training"
        expect = max(1, int(rt.mround(float(cfg["perSeed"]) * n_seed / len(n_img)))) * len(n_img)
        assert len(ck["rows"]) == expect, f"{len(ck['rows'])} patches/epoch, expected {expect}"
    else:
        want = min(log, key=lambda r: float(r["val_loss"]))
        assert st["epoch"] == int(want["epoch"]), "net.pt is not the best-validation epoch"
        assert len(ck["rows"]) == fold["patchesPerEpoch"], "patches/epoch differ from MATLAB's"
    mean = st["model"]["mean"]
    d = np.abs(mean.numpy() - fold["mean"]).max()
    assert d <= 0.005, f"zero-centre mean {mean.numpy()} vs MATLAB {fold['mean']}"
    assert float(log[1]["train_loss"]) < float(log[0]["train_loss"]), "training loss did not fall"
    n = len(list((Path(run_root) / "probcache").glob("*.mat")))
    assert n == len(fold["test"]), f"{n} P files for {len(fold['test'])} held-out images"
    secs = np.mean([float(r["seconds"]) for r in log])
    epochs = 40 if budget else 20
    print(f"  smoke          ok (resume, output epoch {st['epoch']}, {len(ck['rows'])} patches/epoch, "
          f"mean within {d:.2g}, loss {float(log[0]['train_loss']):.4f} -> {float(log[1]['train_loss']):.4f}"
          + (f", lr {log[0]['lr']} -> {log[1]['lr']})" if budget else ")"))
    print(f"  {secs:.0f} s/epoch on {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}: "
          f"one 5-fold CV of {epochs} epochs ~ {secs * epochs * 5 / 3600:.1f} h")


if __name__ == "__main__":
    if sys.argv[1] in ("stage3", "budget"):
        # python colab/test_parity.py stage3|budget <bundle> <scratch_run_root> [workers]
        print(f"{sys.argv[1]}: one-fold smoke")
        stage3(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0,
               budget=sys.argv[1] == "budget")
        print("All checks passed.")
        sys.exit()
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
