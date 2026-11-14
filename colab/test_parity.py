"""Checks of the PyTorch port (docs/briefs/colab-parity.md, stages 1-3).

    python colab/test_parity.py <bundle> [<cache_out>]
    python colab/test_parity.py stage3 <bundle> <scratch_run_root> [workers]
    python colab/test_parity.py stage3-seedless <bundle> <scratch_run_root> [workers]
    python colab/test_parity.py selftest

Stage 1 checks the port against fixtures run_export wrote from the MATLAB code.
Stage 2 runs every fold's MATLAB weights through the PyTorch network, checks one
patch and one whole-image P against MATLAB's CPU, and writes P to <cache_out>
for run_cache to score. Stage 3 checks the augmentation's warp moves the
heat-map with the image, then smoke-runs the trainer on one fold.
stage3-seedless does the same for the seedless network
(docs/results/2026-11-12-seedless-fallback.md). selftest needs no bundle: the
negative seeds and the decoder's seed input
(docs/briefs/2026-11-14-hard-neg-seeds.md), which stage 3 also runs.
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
    """affine_clahe's warp moves the heat-map peak to where the affine
    maps the seed, and the label with it."""
    p, sigma = int(cfg["patch"]), float(cfg["sigmaSeed"])
    seed = np.array([p / 2 + 20, p / 2 - 30])          # (col, row), 1-based like seed_patch
    X, _, _ = rt.seed_patch(np.zeros((p, p, 1), np.float32), seed, np.array([p / 2, p / 2]), p, sigma)
    fg = np.zeros((p, p), bool)
    fg[200:240, 300:340] = True
    draws = iter([0.0, 1.0, 0.0, 1.0, 0.9])            # affine on, rot 15, scale 0.9, shear 5, CLAHE off
    Xw, fgw = rt.affine_clahe(X, fg, lambda: next(draws))
    M = rt.affine_matrix(p, 15, 0.9, 5)
    want = M[:, :2] @ seed + M[:, 2]        # the patch starts at image row/col 0: index = coordinate
    got = np.unravel_index(np.argmax(Xw[:, :, -1]), (p, p))[::-1]
    assert np.abs(got - want).max() <= 1, f"heat-map peak at {got}, the affine maps the seed to {want}"
    assert fgw.dtype == bool and 0.5 < fgw.sum() / fg.sum() / 0.81 < 1.5, "label did not follow the warp"
    print(f"  affine         ok (heat-map peak within {np.abs(got - want).max():.2g} px of the mapped seed)")


def check_negatives():
    """negative_rows spends negSeedFrac of the epoch equally across images, only
    on their candidates; a negative row's target is all background and its
    heat-map peaks on its (cx, cy), with and without augmentation; an
    all-negative minibatch leaves the loss and its gradient finite."""
    rng = np.random.default_rng(0)
    label = np.zeros((600, 600), np.uint16)
    label[200:300, 200:300] = 1
    negB = np.zeros((600, 600), bool)
    negB[500:, :] = True
    a = dict(chans=rng.random((600, 600, 1)).astype(np.float32), label=label,
             seeds=np.array([[250.0, 250.0], [550.0, 100.0]]), comp=np.array([1, 0]),
             negA=np.array([False, True]), negB=negB)
    images = {"a": a, "b": dict(a, negA=np.array([False, False]))}
    cfg = dict(patch=512, sigmaSeed=32.0, jitter=32, negSeedFrac=0.2)

    neg = rt.negative_rows(cfg, images, ["a", "b"], 400, rng)
    assert len(neg) == 100, f"{len(neg)} negatives for 400 positives at 0.20, want 100"
    assert [sum(r[0] == n for r in neg) for n in images] == [50, 50], "negatives not shared equally"
    fromA = [r for r in neg if r[1] == (550.0, 100.0)]
    assert all(r[0] == "a" for r in fromA) and 10 < len(fromA) < 40, \
        f"{len(fromA)} of image a's 50 negatives are its rejected detection, want about half"
    assert all(negB[int(cy) - 1, int(cx) - 1] for _, (cx, cy) in neg if (cx, cy) != (550.0, 100.0)), \
        "a background negative is off its candidates"
    assert rt.negative_rows(dict(cfg, negSeedFrac=0.0), images, ["a", "b"], 400, rng) == [], "0 drew negatives"

    X, T = rt.PatchSet(cfg, images, [("a", (550.0, 100.0))], False)[0]
    assert T[1].sum() == 0 and T[0].all(), "a negative seed's target is not all background"
    peak = np.unravel_index(int(X[-1].argmax()), X[-1].shape)
    assert peak == (256, 256), f"heat-map peak at {peak}, not on the negative seed"
    assert rt.PatchSet(cfg, images, [("a", 0)], False)[0][1][1].sum() > 0, "the positive control lost its lesion"
    torch.manual_seed(0)
    aug = rt.PatchSet(cfg, images, [("a", (550.0, 100.0))], True)
    for _ in range(20):
        X, T = aug[0]
        peak = np.array(np.unravel_index(int(X[-1].argmax()), X[-1].shape))
        assert T[1].sum() == 0, "an augmented negative target has foreground"
        assert X[-1].max() > 0.9 and np.abs(peak - 255.5).max() <= 64, f"augmented heat-map peak at {peak}"

    logits = torch.zeros(8, 2, 64, 64)
    logits[:, 1] = -30.0
    logits.requires_grad_()
    T = torch.stack([torch.ones(8, 64, 64), torch.zeros(8, 64, 64)], 1)
    loss = rt.ce_dice_loss(torch.softmax(logits, 1), T)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all(), "all-negative minibatch: loss not finite"
    print(f"  negative seeds ok (share, sources, empty target, heat-map; all-negative loss {loss.item():.4f})")


def check_multiscale(encoder="efficientnet-b3"):
    """multiScaleSeed gives each decoder block's first conv one more input and
    leaves the encoder as it was; the forward pass, which fails if the heat-map
    does not arrive, runs in fp32 and under bf16 autocast."""
    try:
        plain = rt.SmpNet(encoder, 2, False, weights=None)
    except ImportError:
        print("  multi-scale    SKIPPED: segmentation_models_pytorch is not installed (stage 3 runs it)")
        return
    multi = rt.SmpNet(encoder, 2, True, weights=None).eval()
    for p, m in zip(plain.net.decoder.blocks, multi.net.decoder.blocks):
        assert m.conv1[0].in_channels == p.conv1[0].in_channels + 1, "a decoder block lacks the seed channel"
    shapes = lambda net: {k: v.shape for k, v in net.net.encoder.state_dict().items()}
    assert shapes(plain) == shapes(multi), "the encoder changed"
    x = torch.rand(2, 2, 256, 256)
    with torch.no_grad():
        Y = multi(x)
        with torch.autocast("cpu", torch.bfloat16):
            Yb = multi(x)
    assert Y.shape == (2, 2, 256, 256) and torch.allclose(Y.sum(1), torch.ones(2, 256, 256)), "bad output"
    assert torch.isfinite(Yb).all(), "bf16 forward not finite"
    print(f"  multi-scale    ok ({len(multi.net.decoder.blocks)} decoder blocks take the heat-map; encoder unchanged)")


def stage3(root, run_root, workers):
    """Fold 1: train 1 epoch, stop, resume to epoch 2, write P. Checks the
    cosine schedule survives the resume, net.pt holds the last epoch, and the
    inner images train."""
    import csv
    import math
    cfg, images, folds = rt.load_bundle(root)
    fold = folds[0]
    check_affine(cfg)
    check_negatives()
    check_multiscale(str(cfg["encoder"]))
    assert rt.train_fold(root, run_root, 1, 0, epochs=2, stop_after=1, workers=workers) is None
    assert rt.train_fold(root, run_root, 1, 0, epochs=2, workers=workers) is not None
    log = list(csv.DictReader(open(Path(run_root) / "fold1" / "log.csv")))
    assert [int(r["epoch"]) for r in log] == [1, 2], f"epochs logged {[r['epoch'] for r in log]}"
    st = torch.load(Path(run_root) / "fold1" / "net.pt", weights_only=False)
    ck = torch.load(Path(run_root) / "fold1" / "ckpt.pt", weights_only=False)
    assert st["epoch"] == 2, f"net.pt holds epoch {st['epoch']}, not the last"
    lr0, lr_min = float(cfg["learnRate"]), float(cfg["lrMin"])
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
    frac = float(cfg.get("negSeedFrac", 0.0))
    expect += int(rt.mround(frac / (1 - frac) * expect))
    assert len(ck["rows"]) == expect, f"{len(ck['rows'])} patches/epoch, expected {expect}"
    # The negatives MATLAB exported lie where cnngvf.negativeSeeds says, on training images.
    from scipy.ndimage import distance_transform_edt
    far = {}
    for name, (cx, cy) in (r for r in ck["rows"] if isinstance(r[1], tuple)):
        assert name not in held, f"a negative seed on test image {name}"
        if name not in far:
            far[name] = distance_transform_edt(images[name]["label"] == 0)
        assert far[name][int(rt.mround(cy)) - 1, int(rt.mround(cx)) - 1] > 50, f"a negative within 50 px in {name}"
    assert bool(ck["opts"].get("multiScaleSeed")) == bool(cfg.get("multiScaleSeed", False)), "multiScaleSeed not applied"
    assert float(log[1]["train_loss"]) < float(log[0]["train_loss"]), "training loss did not fall"
    cache = list((Path(run_root) / "probcache").glob("*.mat"))
    assert len(cache) == len(fold["test"]), f"{len(cache)} P files for {len(fold['test'])} held-out images"
    P = sio.loadmat(cache[0])["P"]
    assert np.isfinite(P).all() and P.min() >= 0 and P.max() <= 1, f"P outside [0, 1] in {cache[0].name}"
    secs = np.mean([float(r["seconds"]) for r in log])
    epochs = int(cfg["epochsCv"])
    print(f"  smoke          ok (resume, output epoch {st['epoch']}, {len(ck['rows'])} patches/epoch, "
          f"loss {float(log[0]['train_loss']):.4f} -> {float(log[1]['train_loss']):.4f}, "
          f"lr {log[0]['lr']} -> {log[1]['lr']})")
    print(f"  {secs:.0f} s/epoch on {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}: "
          f"one 5-fold CV of {epochs} epochs ~ {secs * epochs * 5 / 3600:.1f} h")


def check_frame_shift():
    """The seedless frame warp translates by at most 5 % and the label with it;
    a single-channel frame keeps its channel axis."""
    X = np.random.default_rng(0).random((1024, 1024, 1)).astype(np.float32)
    fg = np.zeros((1024, 1024), bool)
    fg[400:500, 400:500] = True
    for end, want in ((1.0, 451), (0.0, 349)):     # the two extremes of the shift draw
        draws = iter([0.0, 0.5, 0.5, 0.5, end, end, 0.0])   # affine on, rot 0, scale 1, shear 0, CLAHE on
        Xw, fgw = rt.affine_clahe(X, fg, lambda: next(draws), tiles=8, shift=0.05)
        r, c = np.nonzero(fgw)
        assert Xw.shape == X.shape, f"frame shape {Xw.shape}"
        assert abs(r.min() - want) <= 1 and abs(c.min() - want) <= 1, f"label at {r.min()}, {c.min()}, want {want}"
    print("  frame shift    ok (+-5 % = +-51 px, label follows, channel axis kept)")


def stage3_seedless(root, run_root, workers):
    """Fold 1 of the seedless network: 1 epoch, stop, resume to epoch 2, write P."""
    import csv
    cfg, images, folds = rt.load_bundle(root)
    fold = folds[0]
    check_frame_shift()
    assert rt.train_fold(root, run_root, 1, 0, epochs=2, stop_after=1, workers=workers, seedless=True) is None
    assert rt.train_fold(root, run_root, 1, 0, epochs=2, workers=workers, seedless=True) is not None
    log = list(csv.DictReader(open(Path(run_root) / "fold1" / "log.csv")))
    assert [int(r["epoch"]) for r in log] == [1, 2], f"epochs logged {[r['epoch'] for r in log]}"
    ck = torch.load(Path(run_root) / "fold1" / "ckpt.pt", weights_only=False)
    n_train = len(images) - len(fold["test"])
    assert len(ck["rows"]) == n_train * int(cfg["seedlessCopies"]), f"{len(ck['rows'])} frames/epoch"
    assert ck["opts"].get("seedless"), "checkpoint is not the seedless network's"
    assert float(log[1]["train_loss"]) < float(log[0]["train_loss"]), "training loss did not fall"
    cache = list((Path(run_root) / "probcache").glob("*.mat"))
    assert len(cache) == len(fold["test"]), f"{len(cache)} P files for {len(fold['test'])} held-out images"
    for c in cache:
        P = sio.loadmat(c)["P"]
        assert P.shape == (1024, 1024) and np.isfinite(P).all() and 0 <= P.min() and P.max() <= 1, c.name
    secs = np.mean([float(r["seconds"]) for r in log])
    epochs = int(cfg["seedlessEpochs"])
    print(f"  smoke          ok (resume, {len(ck['rows'])} frames/epoch, loss {float(log[0]['train_loss']):.4f} -> "
          f"{float(log[1]['train_loss']):.4f}, P for {len(cache)} images)")
    print(f"  {secs:.0f} s/epoch on {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}: "
          f"one 5-fold CV of {epochs} epochs ~ {secs * epochs * 5 / 3600:.1f} h")


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        print("selftest: synthetic, no bundle")
        check_negatives()
        check_multiscale()
        print("All checks passed.")
        sys.exit()
    if sys.argv[1] == "stage3-seedless":
        print("stage3-seedless: one-fold smoke")
        stage3_seedless(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
        print("All checks passed.")
        sys.exit()
    if sys.argv[1] == "stage3":
        print("stage3: one-fold smoke")
        stage3(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
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
