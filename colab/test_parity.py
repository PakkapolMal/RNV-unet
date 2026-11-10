"""Checks of the PyTorch port (docs/briefs/colab-parity.md, stages 1-3).

    python colab/test_parity.py <bundle> [<cache_out>]
    python colab/test_parity.py stage3 <bundle> <scratch_run_root> [workers] [paper]

Stage 1 checks the port against fixtures run_export wrote from the MATLAB code.
Stage 2 runs every fold's MATLAB weights through the PyTorch network, checks one
patch and one whole-image P against MATLAB's CPU, and writes P to <cache_out>
for run_cache to score. Stage 3 is a one-fold smoke run of the trainer; with
"paper" it smokes --recipe paper, after checking its warp moves the heat-map
with the image.
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

    # MATLAB's fold networks held the inner images out; the adopted recipe
    # trains on them, so its seeds are MATLAB's training and validation seeds.
    for f, fold in enumerate(folds, 1):
        n = len(rt.training_rows(cfg, images, fold))
        assert n == fold["trainSeeds"] + fold["valSeeds"], \
            f"fold {f}: {n} seeds vs MATLAB {fold['trainSeeds']} + {fold['valSeeds']}"
    print(f"  seed counts    ok ({len(folds)} folds match MATLAB's info)")
    return fx


def stage2(cfg, images, folds, fx, cache_out, device):
    rt.exact_fp32()
    p, sigma, batch = int(cfg["patch"]), float(cfg["sigmaSeed"]), int(cfg["inferBatch"])
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
            if cache_out:
                P = rt.infer_prob_map(im["chans"], im["seeds"], model, p, sigma, batch, device)
                rt.write_prob_cache(cache_out, name, P, f, arm=str(cfg["arm"]),
                                    netFile=f"MATLAB fold {f} weights in PyTorch")
        if cache_out:
            print(f"  fold {f}         P written for {len(fold['test'])} images; score <cache_out> with run_cache")


def check_affine(cfg):
    """The paper recipe's warp moves the heat-map peak to where the affine
    maps the seed, and the label with it."""
    p, sigma = int(cfg["patch"]), float(cfg["sigmaSeed"])
    seed = np.array([p / 2 + 20, p / 2 - 30])          # (col, row), 1-based like seed_patch
    X, _, _ = rt.seed_patch(np.zeros((p, p, 1), np.float32), seed, np.array([p / 2, p / 2]), p, sigma)
    fg = np.zeros((p, p), bool)
    fg[200:240, 300:340] = True
    draws = iter([0.0, 1.0, 0.0, 1.0, 0.9])            # affine on, rot 15, scale 0.9, shear 5, CLAHE off
    Xw, fgw = rt.paper_augment(X, fg, lambda: next(draws))
    M = rt.affine_matrix(p, 15, 0.9, 5)
    want = M[:, :2] @ seed + M[:, 2]        # the patch starts at image row/col 0: index = coordinate
    got = np.unravel_index(np.argmax(Xw[:, :, -1]), (p, p))[::-1]
    assert np.abs(got - want).max() <= 1, f"heat-map peak at {got}, the affine maps the seed to {want}"
    assert fgw.dtype == bool and 0.5 < fgw.sum() / fg.sum() / 0.81 < 1.5, "label did not follow the warp"
    print(f"  affine         ok (heat-map peak within {np.abs(got - want).max():.2g} px of the mapped seed)")


def stage3(root, run_root, workers, recipe="reference"):
    """Fold 1: train 1 epoch, stop, resume to epoch 2, write P. Checks the
    cosine schedule survives the resume, net.pt holds the last epoch, and the
    inner images train."""
    import csv
    import math
    cfg, images, folds = rt.load_bundle(root)
    fold = folds[0]
    paper = recipe == "paper"
    if paper:
        check_affine(cfg)
    assert rt.train_fold(root, run_root, 1, 0, epochs=2, stop_after=1, workers=workers, recipe=recipe) is None
    assert rt.train_fold(root, run_root, 1, 0, epochs=2, workers=workers, recipe=recipe) is not None
    log = list(csv.DictReader(open(Path(run_root) / "fold1" / "log.csv")))
    assert [int(r["epoch"]) for r in log] == [1, 2], f"epochs logged {[r['epoch'] for r in log]}"
    st = torch.load(Path(run_root) / "fold1" / "net.pt", weights_only=False)
    ck = torch.load(Path(run_root) / "fold1" / "ckpt.pt", weights_only=False)
    assert st["epoch"] == 2, f"net.pt holds epoch {st['epoch']}, not the last"
    lr0, lr_min = (rt.PAPER["lr"], rt.PAPER["lr_min"]) if paper else (float(cfg["learnRate"]), float(cfg["lrMin"]))
    want = [lr_min + (lr0 - lr_min) * (1 + math.cos(math.pi * e / 2)) / 2 for e in (0, 1)]
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
    mean = st["model"]["mean"]
    d = np.abs(mean.numpy() - fold["mean"]).max()
    # CLAHE shifts the intensity mean, so only the reference must match MATLAB's.
    assert paper or d <= 0.005, f"zero-centre mean {mean.numpy()} vs MATLAB {fold['mean']}"
    assert float(log[1]["train_loss"]) < float(log[0]["train_loss"]), "training loss did not fall"
    cache = list((Path(run_root) / "probcache").glob("*.mat"))
    assert len(cache) == len(fold["test"]), f"{len(cache)} P files for {len(fold['test'])} held-out images"
    P = sio.loadmat(cache[0])["P"]
    assert np.isfinite(P).all() and P.min() >= 0 and P.max() <= 1, f"P outside [0, 1] in {cache[0].name}"
    if paper:
        try:    # a checkpoint from the other recipe must refuse to resume
            rt.train_fold(root, run_root, 1, 0, epochs=2, workers=workers)
            raise RuntimeError("a paper-recipe checkpoint resumed under the reference recipe")
        except AssertionError:
            pass
    secs = np.mean([float(r["seconds"]) for r in log])
    epochs = int(cfg["epochsCv"])
    print(f"  smoke          ok (resume, output epoch {st['epoch']}, {len(ck['rows'])} patches/epoch, "
          f"mean within {d:.2g}, loss {float(log[0]['train_loss']):.4f} -> {float(log[1]['train_loss']):.4f}, "
          f"lr {log[0]['lr']} -> {log[1]['lr']})")
    print(f"  {secs:.0f} s/epoch on {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}: "
          f"one 5-fold CV of {epochs} epochs ~ {secs * epochs * 5 / 3600:.1f} h")


if __name__ == "__main__":
    if sys.argv[1] == "stage3":
        print("stage3: one-fold smoke")
        stage3(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0,
               sys.argv[5] if len(sys.argv) > 5 else "reference")
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
