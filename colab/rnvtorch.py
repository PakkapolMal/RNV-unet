"""PyTorch port of the seed-conditioned U-Net: patches, network, loss, training, inference.

    python rnvtorch.py train <bundle> <run_root> --fold F --seed S [--epochs N]
                             [--stop-after N] [--workers N]

trains one fold, checkpointing every epoch to <run_root>/fold<F>/ and resuming
from there when rerun, then writes the fold's held-out P to <run_root>/probcache/.

Reads the bundle cnn/run_export.m writes and writes P in the format
cnngvf.probCache reads, so loading, labelling, folds, post-processing and
scoring all stay in MATLAB. docs/briefs/colab-parity.md is the plan; each
function names the MATLAB function it ports. Coordinates follow MATLAB: seeds
are 1-based [cx cy], arrays are indexed [row, col].
"""
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn


def mround(x):
    """MATLAB round: half away from zero (Python's round is half to even)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# --- bundle -----------------------------------------------------------------

def load_bundle(root):
    """Return (cfg, images, folds) from run_export's output directory."""
    root = Path(root)
    cfg = sio.loadmat(root / "bundle.mat", squeeze_me=True)
    images = {}
    for name in cfg["images"]:
        m = sio.loadmat(root / "images" / (Path(name).stem + ".mat"), squeeze_me=True)
        chans = m["chans"]
        images[name] = dict(
            chans=chans[:, :, None] if chans.ndim == 2 else chans,
            label=m["label"],
            seeds=np.asarray(m["seeds"], float).reshape(-1, 2),
            comp=np.atleast_1d(m["comp"]).astype(int),
            dataset=str(m["dataset"]),
            refP=m["refP"] if np.size(m["refP"]) else None,
            refFold=float(m["refFold"]),
        )
    folds = []
    for f in range(1, int(cfg["k"]) + 1):
        m = sio.loadmat(root / "folds" / f"fold{f}.mat", squeeze_me=False)
        folds.append(dict(
            test=[str(s[0]) for s in m["test"].ravel()],
            inner=[str(s[0]) for s in m["inner"].ravel()],
            patchesPerEpoch=int(m["patchesPerEpoch"].item()),
            trainSeeds=int(m["trainSeeds"].item()),
            valSeeds=int(m["valSeeds"].item()),
            layer=[str(s[0]) for s in m["layer"].ravel()],
            param=[str(s[0]) for s in m["param"].ravel()],
            values=[np.asarray(v) for v in m["values"].ravel()],
            mean=np.asarray(m["mean"], np.float32).ravel(),
        ))
    return cfg, images, folds


# --- patches (cnngvf.seedPatch, cnngvf.prepareData) -------------------------

def seed_patch(chans, seed, center, p, sigma):
    """Crop p x p around center, zero-padded, plus the Gaussian heat-map at seed.

    Returns X (p, p, C+1) float32 and the 1-based image rows/cols it covers.
    """
    h, w, C = chans.shape
    half = p // 2
    r0 = int(mround(center[1])) - half
    c0 = int(mround(center[0])) - half
    rows = np.arange(r0, r0 + p)
    cols = np.arange(c0, c0 + p)
    X = np.zeros((p, p, C + 1), np.float32)
    inR = (rows >= 1) & (rows <= h)
    inC = (cols >= 1) & (cols <= w)
    X[np.ix_(inR, inC)] = chans[np.ix_(rows[inR] - 1, cols[inC] - 1)]
    dy = (rows - seed[1])[:, None]
    dx = (cols - seed[0])[None, :]
    X[:, :, C] = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    return X, rows, cols


def training_rows(cfg, images, fold):
    """Seed rows (image, seed index) that train and validate one fold.

    Training images are those with points, minus the fold's test and inner
    images; seeds with comp 0 are dropped (cnngvf.seedLabels).
    """
    held = set(fold["test"])
    train, val = [], []
    for name, im in images.items():
        if name in held:
            continue
        rows = [(name, s) for s in range(len(im["comp"])) if im["comp"][s] > 0]
        (val if name in fold["inner"] else train).extend(rows)
    return train, val


def epoch_rows(cfg, train, rng):
    """One epoch's patch list: perSeed x seeds, shared equally across images."""
    names = list(dict.fromkeys(n for n, _ in train))
    if not cfg["balanceByImage"]:
        return train * int(cfg["perSeed"])
    budget = max(1, int(mround(cfg["perSeed"] * len(train) / len(names))))
    out = []
    for n in names:
        mine = [r for r in train if r[0] == n]
        out += [mine[i] for i in rng.integers(len(mine), size=budget)]
    return out


# --- network (MATLAB unet, EncoderDepth 4) ----------------------------------

def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """MATLAB unet([512 512 C], 2, EncoderDepth=4): no BatchNorm, dropout 0.5
    after encoder stage 4 and the bridge, zero-centred input, softmax output.
    Layer names follow MATLAB's so weights map by name.
    """

    def __init__(self, cin=2, depth=4, first=64):
        super().__init__()
        self.register_buffer("mean", torch.zeros(cin))
        f = [first * 2 ** i for i in range(depth)]
        self.enc = nn.ModuleList(_block(a, b) for a, b in zip([cin] + f[:-1], f))
        self.bridge = _block(f[-1], 2 * f[-1])
        self.up = nn.ModuleList(nn.ConvTranspose2d(2 * c, c, 2, stride=2) for c in reversed(f))
        self.dec = nn.ModuleList(_block(2 * c, c) for c in reversed(f))
        self.head = nn.Conv2d(f[0], 2, 1)
        self.drop = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = x - self.mean.view(1, -1, 1, 1)
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i == len(self.enc) - 1:
                x = self.drop(x)
            skips.append(x)
            x = self.pool(x)
        x = self.drop(self.bridge(x))
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([skip, torch.relu(up(x))], 1))
        return torch.softmax(self.head(x), 1)

    def matlab_layers(self):
        """MATLAB layer name -> PyTorch module."""
        d = len(self.enc)
        m = {}
        for i, blk in enumerate(self.enc):
            m[f"Encoder-Stage-{i + 1}-Conv-1"] = blk[0]
            m[f"Encoder-Stage-{i + 1}-Conv-2"] = blk[2]
        m["LatentNetwork-Bridge-Conv-1"] = self.bridge[0]
        m["LatentNetwork-Bridge-Conv-2"] = self.bridge[2]
        for i in range(d):
            m[f"Decoder-Stage-{i + 1}-UpConv"] = self.up[i]
            m[f"Decoder-Stage-{i + 1}-Conv-1"] = self.dec[i][0]
            m[f"Decoder-Stage-{i + 1}-Conv-2"] = self.dec[i][2]
        m["encoderDecoderFinalConvLayer"] = self.head
        return m

    def load_matlab(self, fold):
        """Copy a fold's MATLAB weights in. Both conv and transposed-conv weights
        are [h w a b] in MATLAB and [b a h w] in PyTorch."""
        mods = self.matlab_layers()
        assert set(fold["layer"]) == set(mods), "layer names differ from MATLAB's"
        with torch.no_grad():
            for layer, param, v in zip(fold["layer"], fold["param"], fold["values"]):
                mod = mods[layer]
                if param == "Weights":
                    v = v.reshape(v.shape + (1,) * (4 - v.ndim)).transpose(3, 2, 0, 1)
                    mod.weight.copy_(torch.from_numpy(np.ascontiguousarray(v)))
                else:
                    mod.bias.copy_(torch.from_numpy(v.ravel()))
            self.mean.copy_(torch.from_numpy(fold["mean"]))
        return self


# --- loss (generalizedDice) -------------------------------------------------

def generalized_dice(Y, T):
    """Per-observation generalized Dice; Y, T are (B, 2, H, W).

    As MATLAB's generalizedDice, epsilon included: fitted to run_export's
    fixture, where a patch with no lesion scores ~0 rather than NaN.
    """
    e = float(np.finfo(np.float32).eps)
    w = 1.0 / (T.sum((2, 3)).square() + e)
    num = 2 * (w * (Y * T).sum((2, 3))).sum(1)
    den = (w * (Y.square() + T.square()).sum((2, 3))).sum(1)
    return (num + e) / (den + e)


def loss_fn(Y, T):
    return 1 - generalized_dice(Y, T).mean()


# --- inference (cnngvf.inferProbMap) ----------------------------------------

@torch.no_grad()
def infer_prob_map(chans, seeds, model, p, sigma, batch, device):
    """Full-image P from one patch per seed, raised-cosine blended."""
    h, w, _ = chans.shape
    P = np.zeros((h, w))
    if len(seeds) == 0:
        return P
    patches, geo = [], []
    for s in seeds:
        X, rows, cols = seed_patch(chans, s, s, p, sigma)
        patches.append(X)
        geo.append((rows, cols))
    Xall = np.stack(patches).transpose(0, 3, 1, 2)
    model.eval()
    Pn = np.concatenate([
        model(torch.from_numpy(Xall[a:a + batch]).to(device))[:, 1].float().cpu().numpy()
        for a in range(0, len(Xall), batch)])
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(p) / (p - 1))
    win = np.maximum(np.outer(hann, hann), 1e-3)
    W = np.zeros((h, w))
    for (rows, cols), pn in zip(geo, Pn):
        inR = (rows >= 1) & (rows <= h)
        inC = (cols >= 1) & (cols <= w)
        ix = np.ix_(rows[inR] - 1, cols[inC] - 1)
        sub = np.ix_(inR, inC)
        P[ix] += pn[sub].astype(np.float64) * win[sub]
        W[ix] += win[sub]
    return P / np.maximum(W, np.finfo(float).eps)


def write_prob_cache(cache_dir, name, P, fold, **meta):
    """One image's P in the format cnngvf.probCache('read') accepts."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(fold=float(fold), image=name, source="pytorch", **meta)
    sio.savemat(cache_dir / (Path(name).stem + ".mat"),
                dict(P=P.astype(np.float32), meta=meta), do_compression=True)


def exact_fp32():
    """No TF32: parity is judged to 1e-3 on P."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# --- training (cnngvf.prepareData's datastore, cnngvf.trainNet) -------------

class PatchSet(torch.utils.data.Dataset):
    """One patch per row; training draws a fresh jitter and dihedral transform
    each read, validation is deterministic. Randomness comes from torch, which
    seeds every DataLoader worker differently."""

    def __init__(self, cfg, images, rows, augment):
        self.images, self.rows, self.augment = images, rows, augment
        self.p, self.sigma, self.jitter = int(cfg["patch"]), float(cfg["sigmaSeed"]), int(cfg["jitter"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name, s = self.rows[i]
        im = self.images[name]
        seed = im["seeds"][s]
        center = seed
        if self.augment:
            center = seed + torch.randint(-self.jitter, self.jitter + 1, (2,)).numpy()
        X, rows, cols = seed_patch(im["chans"], seed, center, self.p, self.sigma)
        h, w = im["label"].shape
        inR = (rows >= 1) & (rows <= h)
        inC = (cols >= 1) & (cols <= w)
        fg = np.zeros((self.p, self.p), bool)
        fg[np.ix_(inR, inC)] = im["label"][np.ix_(rows[inR] - 1, cols[inC] - 1)] == im["comp"][s]
        if self.augment:
            if torch.rand(1) < 0.5:
                X, fg = np.fliplr(X), np.fliplr(fg)
            if torch.rand(1) < 0.5:
                X, fg = np.flipud(X), np.flipud(fg)
            r = int(torch.randint(4, (1,)))
            X, fg = np.rot90(X, r), np.rot90(fg, r)
        X = torch.from_numpy(np.ascontiguousarray(X.transpose(2, 0, 1)))
        T = torch.from_numpy(np.stack([~fg, fg]).astype(np.float32))
        return X, T


def glorot_(model):
    """MATLAB's default initialisation: Glorot-uniform weights, zero biases."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


def channel_mean(loader):
    """trainnet's zerocenter statistic: per-channel mean over one pass of the
    training data, augmentation included."""
    total, n = 0.0, 0
    for X, _ in loader:
        total = total + X.double().sum((0, 2, 3))
        n += X.shape[0] * X.shape[2] * X.shape[3]
    return (total / n).float()


def train_fold(root, run_root, fold_no, seed, epochs=None, stop_after=None, workers=0, device=None):
    """Train one fold as run_cv does, resuming from <run_root>/fold<k>/ckpt.pt.

    Adam at cfg.learnRate with L2 1e-4 on weights only (MATLAB's default, added
    to the gradient: torch Adam's weight_decay, not AdamW), minibatch
    cfg.miniBatch, shuffle every epoch, validation once per epoch,
    best-validation output. The epoch's patch list is drawn once, as
    prepareData does. Returns the trained model, or None when stop_after
    stopped it early.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    exact_fp32()
    torch.backends.cudnn.benchmark = True
    cfg, images, folds = load_bundle(root)
    fold = folds[fold_no - 1]
    epochs = int(epochs or cfg["epochsCv"])
    mb = int(cfg["miniBatch"])
    out = Path(run_root) / f"fold{fold_no}"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "ckpt.pt"

    train, val = training_rows(cfg, images, fold)
    model = UNet().to(device)
    decay = [p for n, p in model.named_parameters() if n.endswith("weight")]
    no_decay = [p for n, p in model.named_parameters() if n.endswith("bias")]
    opt = torch.optim.Adam([dict(params=decay, weight_decay=1e-4), dict(params=no_decay, weight_decay=0.0)],
                           lr=float(cfg["learnRate"]), betas=(0.9, 0.999), eps=1e-8)

    if ckpt.exists():
        # CPU: the RNG states must stay CPU tensors; load_state_dict moves the rest.
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["rng"])
        if device == "cuda" and st["cuda_rng"] is not None:
            torch.cuda.set_rng_state(st["cuda_rng"])
        rows, done, best = st["rows"], st["epoch"], st["best"]
        print(f"fold {fold_no}: resuming after epoch {done} (best epoch {best['epoch']})")
    else:
        torch.manual_seed(seed)
        rows = epoch_rows(cfg, train, np.random.default_rng(seed))
        glorot_(model)
        once = torch.utils.data.DataLoader(PatchSet(cfg, images, rows, True), batch_size=16, num_workers=workers)
        with torch.no_grad():
            model.mean.copy_(channel_mean(once).to(device))
        done, best = 0, dict(loss=float("inf"), epoch=0, state=None)
    print(f"fold {fold_no}: {len(train)} seeds, {len(rows)} patches/epoch, {len(val)} validation seeds, "
          f"mean {np.round(model.mean.cpu().numpy(), 4)} (MATLAB {np.round(fold['mean'], 4)}), {device}")

    train_loader = torch.utils.data.DataLoader(PatchSet(cfg, images, rows, True), batch_size=mb, shuffle=True,
                                               drop_last=True, num_workers=workers, persistent_workers=workers > 0)
    val_loader = torch.utils.data.DataLoader(PatchSet(cfg, images, val, False), batch_size=mb, num_workers=workers)
    log = out / "log.csv"
    for epoch in range(done + 1, epochs + 1):
        t = time.time()
        model.train()
        tl, nb = 0.0, 0
        for X, T in train_loader:
            loss = loss_fn(model(X.to(device)), T.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl, nb = tl + loss.item(), nb + 1
        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for X, T in val_loader:
                vl += (1 - generalized_dice(model(X.to(device)), T.to(device))).sum().item()
                nv += X.shape[0]
        vl /= nv
        if vl < best["loss"]:
            best = dict(loss=vl, epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        secs = time.time() - t
        new = not log.exists()
        with open(log, "a", newline="") as fh:
            wr = csv.writer(fh)
            if new:
                wr.writerow(["epoch", "train_loss", "val_loss", "seconds", "best_epoch"])
            wr.writerow([epoch, tl / nb, vl, round(secs, 1), best["epoch"]])
        print(f"fold {fold_no} epoch {epoch}/{epochs}: train {tl / nb:.4f} val {vl:.4f} "
              f"(best {best['epoch']}) {secs:.0f} s", flush=True)
        st = dict(model=model.state_dict(), opt=opt.state_dict(), rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state() if device == "cuda" else None,
                  rows=rows, epoch=epoch, best=best, seed=seed)
        torch.save(st, out / "ckpt.tmp")
        os.replace(out / "ckpt.tmp", ckpt)    # a disconnect mid-write keeps the last good one
        if stop_after and epoch - done >= stop_after and epoch < epochs:
            print(f"fold {fold_no}: stopped after epoch {epoch}; rerun to resume")
            return None

    model.load_state_dict(best["state"])
    torch.save(dict(model=best["state"], epoch=best["epoch"], val_loss=best["loss"], seed=seed), out / "net.pt")
    p, sigma, batch = int(cfg["patch"]), float(cfg["sigmaSeed"]), int(cfg["inferBatch"])
    for name in fold["test"]:
        im = images[name]
        P = infer_prob_map(im["chans"], im["seeds"], model, p, sigma, batch, device)
        write_prob_cache(Path(run_root) / "probcache", name, P, fold_no, arm=str(cfg["arm"]),
                         netFile=f"PyTorch fold {fold_no} seed {seed} epoch {best['epoch']}")
    print(f"fold {fold_no}: done, best epoch {best['epoch']}, P written for {len(fold['test'])} held-out images")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("bundle")
    ap.add_argument("run_root")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--stop-after", type=int)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    train_fold(a.bundle, a.run_root, a.fold, a.seed, a.epochs, a.stop_after, a.workers)
