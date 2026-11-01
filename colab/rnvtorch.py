"""PyTorch port of the seed-conditioned U-Net: patches, network, loss, inference.

Reads the bundle cnn/run_export.m writes and writes P in the format
cnngvf.probCache reads, so loading, labelling, folds, post-processing and
scoring all stay in MATLAB. docs/briefs/colab-parity.md is the plan; each
function names the MATLAB function it ports. Coordinates follow MATLAB: seeds
are 1-based [cx cy], arrays are indexed [row, col].
"""
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
