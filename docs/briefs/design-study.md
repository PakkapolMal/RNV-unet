---
title: RNV Segmentation via GVF — Original Design Specification
description: The original two-approach design study, written before implementation. Superseded as a specification; retained for its literature review, parameter tables and rationale.
kind: design-study
status: historical
dataset: n/a — written before any dataset was measured
updated: 2026-08-11
---

# RNV Segmentation via Gradient Vector Flow (GVF) on OCTA En-Face Images — MATLAB Implementation Specification

> **Historical.** This is the design study the implementation started from, not a
> description of what was built. Several of its recommendations were measured and
> reversed — `f = P` as a force source, a balloon term of ±1.5, a 256 px patch — and
> the module layout it proposes is not the one that exists. The working
> specification is [SPEC.md](../SPEC.md); the vocabulary is
> [CONTEXT.md](../CONTEXT.md). Kept for the literature review, the parameter ranges and
> the reasoning behind the overall architecture, all of which still hold.

## TL;DR
- Two GVF pipelines are specified. **Approach 1 (CNN probability map → GVF refinement) is the recommended optimal path**; a patch-based U-Net trained around localized (x,y) seeds produces a smooth probability map, and a GVF/GGVF snake or GVF-driven DRLSE level set refines each lesion's boundary. Approach 2 (GVF as model-agnostic post-processing) is the recommended fallback/baseline when labeled data is scarce.
- Prefer a **level-set GVF (DRLSE with GVF external force) over a parametric snake** for the final evolver because the ground truth contains ~10+ disconnected, frond-like lesions per image; level sets handle topological splits/merges natively while parametric snakes require per-component contour bookkeeping.
- Use **GGVF or VFC external forces instead of plain GVF** for frond-like RNV with deep narrow concavities, and drive the small-foreground class imbalance with generalized Dice / focal-Tversky loss. Evaluate with Dice, IoU, Hausdorff, bfscore, and clDice (topology).

## Problem Statement
Segment retinal neovascularization (RNV) lesions in 1024×1024 grayscale OCTA en-face images. Lesions are pre-localized: the pipeline receives (x,y) coordinates (detection points / bounding-box centers / seeds) from a prior localization stage (the IEEE-Access RNV localization work). The task is to convert seeds + image into crisp binary masks matching provided ground-truth masks (white RNV on black), where RNV occupies only a small foreground fraction and appears as multiple disconnected, irregular, blob/frond-like regions.

### Input/Output Contract
- **Image**: `1024×1024`, grayscale (`uint8` or `uint16`), bright vessels/tufts on dark background; contains speckle noise and projection artifacts.
- **Seeds**: `S = [x1 y1; x2 y2; ...]`.
- **Output mask**: `1024×1024` logical, multiple connected components allowed.
- **Ground truth**: `1024×1024` logical, same convention (training/eval only).

## Recommended Overall Architecture

### Approach 1 — CNN probability map → GVF boundary refinement (RECOMMENDED)
1. Preprocess image (despeckle + CLAHE + optional vesselness).
2. Extract patches centered on each seed (e.g. 256×256), append a Gaussian seed heat-map channel.
3. A U-Net / Attention U-Net predicts a soft probability map `p ∈ [0,1]` per patch; stitch back to full 1024×1024 `P`.
4. Build a GVF edge/potential map from `P` (not from raw gradient): `f = P` (signed potential) or `f = |∇P|`, optionally blended with raw-image edges.
5. Compute GVF/GGVF field from `f`; initialize contour/level-set from `P` thresholded near each seed; evolve to crisp boundary.
6. Merge per-lesion results into the full mask.

**Why optimal**: The CNN supplies a smooth, noise-suppressed, semantically-correct force field that fixes GVF's two chronic weaknesses on OCTA (speckle sensitivity and leakage), while GVF/level-set restores the crisp, topologically-clean boundary that CNNs lose to downsampling. This mirrors the published "deep active contour" / "automatic contour refinement" pattern (DACN, DALS, ACR). Concretely, in the ACR study (Ding et al., *Advances in Radiation Oncology* 7(5):100968, 2022; PMC9280040), a level-set ACM applied to CNN probability maps for combined bowels raised mean DSC **from 0.34 to 0.59** in the major-error group and **from 0.72 to 0.79** in the minor-error group — demonstrating that probability-map ACM refinement yields large boundary gains where the CNN is roughest and modest gains where it is already good.

### Approach 2 — GVF as model-agnostic post-processing (FALLBACK/BASELINE)
Any upstream segmenter (Otsu, Frangi vesselness + threshold, region growing from seeds, k-means, SVM/random-forest pixel classifier, or a CNN) produces a binary/soft map. Convert it to a GVF initialization (contour via `bwboundaries` or a binary-mask level set), compute GVF from an edge map, and evolve to refine boundaries. Fully classical; no training data required; tuned only via GVF/snake parameters.

### Combined (best of both)
Use Approach 1 when labeled patches exist; fall back to Approach 2's classical edge map blended into the CNN force field (`f = w·P + (1−w)·|∇(vesselness)|`) so the system degrades gracefully where the CNN is uncertain. A single `runPipeline.m` switch selects `mode = "cnn+gvf" | "classical+gvf"`.

## Proposed MATLAB Module Structure
```
+rnv/
  runPipeline.m        % orchestration; mode switch; loops over seeds/lesions
  preprocess.m         % despeckle, CLAHE, vesselness, normalization
  extractPatches.m     % seed-centered patch crop + Gaussian heatmap channel
  trainRNVNet.m        % build+train U-Net/AttU-Net (Deep Learning + CV Toolbox)
  inferProbMap.m       % patch inference + stitching -> full P
  buildEdgeMap.m       % construct f from P and/or raw image (Approach 1 & 2)
  gvfField.m           % GVF / GGVF / VFC external force field
  initFromSeeds.m      % contour or level-set init from seeds + P/binary
  snakeDeform.m        % parametric GVF snake evolution (Xu-Prince style)
  gvfLevelSet.m        % DRLSE / Chan-Vese level set with GVF force
  multiLesion.m        % per-component handling, merge, topology
  postprocess.m        % morphological cleanup, small-object removal
  evaluate.m           % Dice/IoU/Hausdorff/bfscore/clDice
  tuneParams.m         % bayesopt / grid search vs ground truth
  config.m             % all default params in one struct
```

## PSEUDOCODE — Major Stages

### Preprocessing (`preprocess.m`)
```
function I2 = preprocess(I, cfg)
  I = im2double(mat2gray(I))                 % normalize to [0,1]
  % 1. Despeckle (choose one; OCTA speckle is multiplicative)
  if cfg.denoise == "nlm":   I = imnlmfilt(I, 'DegreeOfSmoothing', cfg.nlm)
  elif cfg.denoise=="wiener":I = wiener2(I,[5 5])
  elif cfg.denoise=="median":I = medfilt2(I,[3 3])
  elif cfg.denoise=="pmad":  I = imdiffusefilt(I,'NumberOfIterations',cfg.pmIter) % Perona-Malik
  % 2. Contrast enhancement
  I = adapthisteq(I,'NumTiles',[16 16],'ClipLimit',cfg.clip)  % CLAHE
  % 3. Optional vesselness channel (multi-scale Hessian / Frangi)
  V = fibermetric(I,cfg.scales,'ObjectPolarity','bright','StructureSensitivity',cfg.ss)
  I2 = cat(3, I, V)                          % feed both to CNN; use I for classical
```
Note: OCTAVA (Untracht et al., *PLOS One* 2021) established median-filter + multi-scale Frangi as a standard en-face OCTA preprocessing workflow; log-transform first if treating speckle as multiplicative (Rayleigh-distributed) so it becomes additive.

### CNN training (`trainRNVNet.m`, Approach 1)
```
function net = trainRNVNet(imds, pxds, cfg)
  % Patch-based to combat class imbalance + memory of 1024^2
  dsTrain = randomPatchExtractionDatastore(imds,pxds,cfg.patch, ...
              'PatchesPerImage',cfg.ppi)
  dsTrain = transform(dsTrain,@augmentFn)    % flips, rot90, small scale/intensity jitter
  lgraph  = unet(cfg.patch, 2, 'EncoderDepth', 4)   % or attention U-Net / ResU-Net
  % Class imbalance: RNV is tiny foreground fraction -> generalized Dice / focal-Tversky
  net = trainnet(dsTrain, lgraph, @(Y,T) generalizedDiceLoss(Y,T), opts)
  % opts = trainingOptions("adam", InitialLearnRate=1e-3, ...
  %          MiniBatchSize=16, MaxEpochs=cfg.epochs, ValidationData=dsVal)
```

### CNN inference + stitch (`inferProbMap.m`)
```
function P = inferProbMap(I2, seeds, net, cfg)
  P = zeros(1024,1024)
  W = zeros(1024,1024)                        % overlap weights for blending
  for each seed s:
     patch = cropCentered(I2, s, cfg.patch)   % pad if near border
     hm    = gaussianHeatmap(cfg.patch, center, cfg.sigmaSeed)
     x     = cat(3, patch, hm)                % coordinate-conditioned channel
     pr    = predict(net, x)                  % softmax foreground channel
     accumulate pr into P with cosine/Gaussian window into W
  P = P ./ max(W, eps)                        % normalized soft probability map
```

### Edge/potential map construction (`buildEdgeMap.m`)
```
function f = buildEdgeMap(P, I, cfg)
  switch cfg.forceSource
    case "prob":      f = P                          % probability as signed potential
    case "probGrad":  f = gradientMag(imgaussfilt(P,cfg.sig))   % |∇p|
    case "distance":  f = normalize(bwdist(~(P>0.5)) - bwdist(P>0.5)) % signed DT potential
    case "blend":     f = cfg.w*P + (1-cfg.w)*normalize(gradientMag(imgaussfilt(I,cfg.sig)))
  f = mat2gray(f)                                     % pre-normalize to [0,1] (Xu-Prince convention)
```
Notes: Xu & Prince pre-normalize all edge maps to [0,1]. For a probability-driven force, `f = P` gives a smooth attractor whose gradient points toward lesion cores (good capture range); `f = |∇P|` reproduces classic edge-map behavior on the CNN boundary. A theoretically clean variant treats the softmax output directly as a level-set/characteristic function (cf. the Mumford-Shah loss of Kim & Ye, arXiv:1904.02872, and the Level Set Loss of Kim et al., arXiv:1910.00950, which shift probability maps into [-0.5, 0.5] to act as φ).

### GVF / GGVF / VFC field (`gvfField.m`)
```
function [u,v] = gvfField(f, cfg)
  switch cfg.forceType
    case "GVF":   [u,v] = GVF(f, cfg.mu, cfg.gvfIter)          % Xu-Prince
    case "GGVF":  [u,v] = GGVF(f, cfg.K, cfg.gvfIter)          % better on thin concavities
    case "VFC":   K = AM_VFK(2, cfg.R, cfg.vfcType, cfg.vfcParam); [u,v] = AM_VFC(f, K)
  [u,v] = normalizeField(u,v)
```
- Plain **GVF** diffuses gradient vectors (PDE, `mu` smoothing, `ITER` iterations); large capture range, but struggles to enter deep even-width concavities.
- **GGVF** replaces constant smoothing with edge-dependent weighting — better for long thin indentations (frond RNV). Note CN-GGVF (Qin et al., IEEE TCSVT 2013) shows both GVF and GGVF fail on even-width indentations; component-normalized GGVF fixes this.
- **VFC** (Li & Acton, IEEE TIP 16(8):2096–2106, 2007) convolves the edge map with a vector-field kernel. The paper explicitly targets the snake's "limited capture range, noise sensitivity, and poor convergence to concavities" and offers "superior noise robustness, reduced computational cost," requiring only *n* convolutions in *n*-dimensional images — attractive for noisy OCTA. Mixed-VFC reduces leakage from inappropriate parameters.

### Initialization from seeds (`initFromSeeds.m`)
```
function [contours, phi0] = initFromSeeds(P, seeds, cfg)
  BW = P > cfg.tau                             % coarse foreground
  BW = imopen(BW, strel('disk',1))
  CC = bwconncomp(BW)
  keep components that contain or are nearest to a seed (assign each seed to a component)
  % Parametric init:
  for each kept component: contours{i} = resample(bwboundaries -> [x y], dmax,dmin)
  % Level-set init (preferred for many lesions):
  phi0 = cfg.c0 * (1 - 2*BW)                    % signed: negative inside, positive outside
```

### Parametric GVF snake evolution (`snakeDeform.m`, Xu-Prince)
```
% Core loop mirrors the original JHU code:
[u,v] = GVF(f, mu, gvfIter)
[x,y] = snakeinit(delta)                        % or from initFromSeeds
[x,y] = snakeinterp(x,y,dmax,dmin)              % adaptive reparameterization
repeat cfg.outer times:
   [x,y] = snakedeform(x,y,alpha,beta,gamma,kappa,u,v,cfg.inner)
   [x,y] = snakeinterp(x,y,dmax,dmin)           % re-mesh each outer step
convergence: stop when mean vertex displacement < cfg.tolPx
```
**Verbatim signatures from the original Xu & Prince (JHU IACL) distribution:**
- `[u,v] = GVF(f, mu, ITER)` — compute the gradient vector flow field. `f` = edge map normalized to [0,1]; `mu` = regularization (increase with noise); `ITER` = diffusion iterations.
- `[x,y] = snakedeform(x,y,alpha,beta,gamma,kappa,fx,fy,ITER)` — `alpha` = elasticity, `beta` = rigidity, `gamma` = viscosity/time-step, `kappa` = external-force weight. Internally builds a pentadiagonal circulant matrix `A`, forms `invAI = inv(A + gamma*I)`, and iterates `x = invAI*(gamma*x + kappa*interp2(fx,x,y))` (and similarly for y).
- `[x,y] = snakedeform2(x,y,alpha,beta,gamma,kappa,kappap,fx,fy,ITER)` — adds a pressure/balloon force weight `kappap`.
- `[xi,yi] = snakeinterp(x,y,dmax,dmin)` — adaptive resample: insert a point where adjacent spacing > `dmax`, remove where < `dmin`.
- `[xi,yi] = snakeinterp1(x,y,dmax,dmin)` — fixed-resolution variant (better implemented).
- `[x,y] = snakeinit(delta)` — interactive init, resampled at spacing `delta`.
- C-mex `gvfc` replaces `GVF.m` and is significantly faster; same `[u,v]=gvfc(f,mu,ITER)` calling convention.
- Helpers in the package: `BoundMirrorExpand/Ensure/Shrink` (boundary conditions), `dt` (distance transform), `gaussianBlur`, `snakedisp`, `snakeindex`, `xconv2`. Example scripts: `gvf_ex.m`, `balloon_ex.m`, `distance_ex.m`, `tradition_ex.m`. **The functions have NO built-in defaults** — all numeric args are required positional parameters.

### GVF level-set evolution (`gvfLevelSet.m`, RECOMMENDED evolver)
```
function BW = gvfLevelSet(f, phi0, u, v, cfg)
  phi = phi0
  for k = 1:cfg.lsIter
     % DRLSE edge term g from f; replace grad(g) with GVF field (u,v) as external force
     dirac  = diracDelta(phi, cfg.eps)
     curv   = divergence(normalizedGradient(phi))
     dist   = distReg(phi)                       % distance regularization (no reinit)
     phi = phi + cfg.dt*( cfg.muDR*dist ...
             + cfg.lambda*dirac.*(u.*phi_x + v.*phi_y + g.*curv) ...
             + cfg.alfa*g.*dirac )               % alfa = balloon/area term
  BW = phi < 0
```
Rationale: DRLSE (Li, Xu, Gui & Fox, IEEE TIP 19(12):3243–3254, 2010) — "The distance regularization effect eliminates the need for reinitialization and thereby avoids its induced numerical errors," and "relatively large time steps can be used in the finite difference scheme to reduce the number of iterations." It splits/merges natively — ideal for the ~10+ lesions. Substituting the GVF field for the raw edge gradient extends capture range so seeds need not be exactly on boundaries.

### Multi-lesion handling (`multiLesion.m`)
```
function M = multiLesion(results, cfg)
  M = false(1024,1024)
  for each per-seed/per-component result: M = M | result
  % dedupe overlaps; if two seeds land in one CC, evolve once
  % topology: level set handles splits; for parametric, evolve each contour independently
```

### Post-processing (`postprocess.m`)
```
M = bwareaopen(M, cfg.minArea)         % drop tiny false blobs
M = imclose(M, strel('disk',1))
M = imfill(M,'holes')                  % optional; fronds may legitimately have holes -> guard
```

### Evaluation (`evaluate.m`)
```
function R = evaluate(pred, gt)
  R.dice   = 2*nnz(pred&gt)/(nnz(pred)+nnz(gt))       % or generalizedDice / dice()
  R.iou    = nnz(pred&gt)/nnz(pred|gt)                % jaccard()
  R.sens   = nnz(pred&gt)/nnz(gt); R.prec = nnz(pred&gt)/nnz(pred)
  R.hd     = imhausdorff(pred,gt)                     % Hausdorff (Image Processing Toolbox)
  R.bf     = bfscore(pred,gt)                         % boundary F-score
  R.clDice = clDice(skeleton(pred),gt, skeleton(gt),pred) % topology (custom via bwskel)
```

## Concrete MATLAB Functions & Toolbox Dependencies

| Step | MATLAB function | Toolbox | Fallback if missing |
|---|---|---|---|
| Normalize | `mat2gray`,`im2double` | base/IPT | — |
| NLM despeckle | `imnlmfilt` | IPT | `wiener2`/`medfilt2` |
| Anisotropic diffusion | `imdiffusefilt` | IPT | custom Perona-Malik loop |
| CLAHE | `adapthisteq` | IPT | `histeq` |
| Vesselness | `fibermetric` | IPT | Frangi File Exchange code |
| Patch datastore | `randomPatchExtractionDatastore` | IPT | manual crop loop |
| Image/label store | `imageDatastore`,`pixelLabelDatastore` | base/CV | — |
| U-Net | `unet`/`unetLayers` | CV Toolbox | manual `dlnetwork` |
| Train | `trainnet`/`trainNetwork`,`trainingOptions` | DL Toolbox | — |
| Loss | `generalizedDice`, custom Dice/focal/Tversky | CV/DL | weighted `crossentropy` |
| Inference | `predict`,`semanticseg` | DL/CV | — |
| Classical AC | `activecontour` (Chan-Vese/edge) | IPT | DRLSE FEX code |
| GVF field | `GVF.m` (Xu-Prince) | File Exchange | `gvfc` C-mex |
| VFC field | `AM_VFC`,`AM_VFK` | FEX #22871 | GVF |
| Snake evolve | `snakedeform`,`snakeinterp` | Xu-Prince FEX | — |
| Level set | DRLSE `drlse_edge` | FEX #12711 | — |
| Boundaries | `bwboundaries`,`bwconncomp` | IPT | — |
| Skeleton (clDice) | `bwskel`,`bwmorph` | IPT | — |
| Hausdorff | `imhausdorff` | IPT | custom |
| Boundary F | `bfscore` | CV Toolbox | custom |
| Eval seg | `evaluateSemanticSegmentation`,`dice`,`jaccard` | CV/IPT | — |
| Tuning | `bayesopt`,`optimizableVariable` | Stats & ML | grid search loop |

## Parameter Table

| Param | Meaning | Default | Search range |
|---|---|---|---|
| `mu` (GVF) | GVF smoothing/regularization | 0.2 | 0.01–0.3 (↑ with noise) |
| `gvfIter` | GVF diffusion iterations | 80 | 40–200 |
| `K` (GGVF) | GGVF edge-weight scale | 0.05 | 0.02–0.2 |
| `R` (VFC) | VFC kernel radius | 32 | 16–64 |
| `vfcParam` γ/ζ | VFC magnitude decay | γ=2 | 1.5–3 (γ), ζ per SNR* |
| `alpha` | snake elasticity (tension) | 0.1 | 0.01–0.6 |
| `beta` | snake rigidity (bending) | 0.0 | 0–1 |
| `gamma` | snake time step / viscosity | 1.0 | 0.5–2 |
| `kappa` | external force weight | 0.6 | 0.3–2.0 |
| `dmax/dmin` | contour resample thresholds | 2 / 0.5 | fixed |
| `inner/outer` | snake iterations | 5 / 25 | tune to convergence |
| `dt` (DRLSE) | level-set time step | 1–5 | CFL-bounded |
| `muDR` | distance-reg weight | 0.2/dt | fixed |
| `lambda` | edge/length weight | 5 | 1–10 |
| `alfa` | balloon/area weight | ±1.5 | −3…3 (sign = shrink/expand) |
| `eps` | Dirac/Heaviside width | 1.5 | 1–2 |
| `tau` | prob threshold for init | 0.5 | 0.3–0.7 |
| `patch` | CNN patch size | 256 | 128–512 |
| `sigmaSeed` | Gaussian seed heatmap σ | 32 | 16–64 |
| `clip` | CLAHE clip limit | 0.01 | 0.005–0.02 |
| `minArea` | small-object removal | 20 px | tune to smallest GT lesion |

*Per Li & Acton (2007), the VFC magnitude functions m1(r)=(r+ε)^(−γ) and m2(r)=exp(−r²/ζ²) use positive parameters where, "as the SNR is decreased, γ (or ζ) should be decreased (or increased)" — i.e. tune γ/ζ to the OCTA speckle level.

Reference default values are drawn from the shipped Xu-Prince demos and papers: `[x,y]=snakeinterp(x,y,2,0.5)`; IDL port of the same code uses `alpha=0.05, beta=0, gamma=1, kappa=0.6`; the GVF paper's U-shape example uses `alpha=0.6, beta=0.0, mu=0.2`; one published demo uses `[u,v]=GVF(f,0.1,80)`. GGVF example values: `GVF(k=0.1)`/`GGVF(K=0.15–0.2)`.

### Parameter tuning (`tuneParams.m`)
```
vars = [optimizableVariable('mu',[0.01 0.3]), ...
        optimizableVariable('kappa',[0.3 2]), ...
        optimizableVariable('alfa',[-3 3]), ...];
obj  = @(t) 1 - meanDiceOverVal(t);          % objective = 1 - Dice on validation
res  = bayesopt(obj, vars, 'MaxObjectiveEvaluations',60, ...
                'AcquisitionFunctionName','expected-improvement-plus');
```
Use grid search for ≤2 params, `bayesopt` (Statistics & ML Toolbox) for ≥3, cross-validated over the train/val split. Optionally per-image adaptive `mu` scaled by estimated speckle level. `bayesopt` maintains a Gaussian-process model + acquisition function and needs far fewer evaluations than grid/random search.

## Data & Directory Layout
```
data/
  images/   *.png  (1024x1024 grayscale)
  masks/    *.png  (1024x1024 logical)
  seeds/    *.mat  (S = Nx2, optional B, c)
splits/     train.txt val.txt test.txt   (patient-level split)
models/     rnvNet.mat
results/    masks/, metrics.csv
```
- **Split**: patient-level (no leakage), ~70/15/15.
- **Augmentation**: horizontal/vertical flips, 90° rotations, small (±10%) scale, mild intensity jitter and speckle simulation; keep augmentations mask-consistent. Avoid heavy elastic warps that distort thin fronds.
- **External datasets for pretraining/validation of the vessel/OCTA machinery**: **ROSE** (Ma et al., IEEE TMI 40(3):928–939, 2021) — "consists of 229 OCTA images with vessel annotations at either centerline-level or pixel level"; ROSE-1 = 117 images from 39 subjects (26 Alzheimer's, 13 controls), ROSE-2 = 112 images (3×3 mm, 304×304). **OCTA-500** (Li et al., *Medical Image Analysis* 2024, arXiv:2012.07261) — "the largest and most comprehensive OCTA dataset... which contains OCTA imaging under two fields of view (FOVs, 6 mm/3 mm)... from 500 subjects," split into OCTA-3M (200 volumes, 304×304) and OCTA-6M (300 volumes, 400×400). These provide vessel-level supervision for pretraining the encoder before fine-tuning on the RNV masks. Note: neither provides RNV/NVE lesion masks, so final training/evaluation must use the project's own ground-truth RNV masks (the DRAC 2022 ultra-wide OCTA challenge, arXiv:2210.00515, is the closest public source of DR lesion segmentation masks).

## Evaluation Protocol
Report per-image and mean ± std over the test set: **Dice/F1, IoU/Jaccard, sensitivity, precision, specificity, 95th-percentile Hausdorff, bfscore, clDice**. clDice (Shit et al., CVPR 2021) is critical because RNV is tubular/frond-like and overlap metrics under-penalize broken topology. Compare three configurations: (a) CNN alone, (b) CNN+GVF (Approach 1), (c) classical+GVF (Approach 2), to quantify GVF's boundary gain. Expect the largest GVF gains on the roughest CNN outputs (paralleling the ACR result of DSC 0.34→0.59 for major-error cases) and smaller gains where the base map is already accurate.

## Risks, Failure Modes & Mitigations
- **Speckle noise → spurious edges**: mitigate with despeckle preprocessing and by driving GVF from the CNN probability map (Approach 1) rather than raw gradient; increase `mu`.
- **Leakage into background / adjacent vessels**: use Mixed-VFC or GGVF, add balloon-term sign control, constrain evolution to a per-seed ROI window.
- **Deep narrow frond concavities (GVF's classic weakness)**: use GGVF/CN-GGVF or VFC; increase capture range; the CNN force map largely removes this problem since the potential already fills concavities.
- **Many disconnected lesions**: prefer level-set (DRLSE) evolver; parametric snakes need one contour per component and cannot split.
- **Computational cost of GVF diffusion on 1024×1024**: run GVF per-ROI (around seeds) not full-frame; use the C-mex `gvfc`; VFC is cheaper (FFT convolution).
- **Initialization sensitivity**: seeds + probability threshold give robust init; GVF's large capture range tolerates imperfect seeds.
- **Extreme class imbalance in CNN training**: generalized Dice / focal-Tversky loss, patch sampling biased toward seed regions.
- **Over-smoothing thin structures**: cap `beta` low (0 often best for fronds), keep CLAHE clip modest, use component-normalized GGVF to preserve weak edges.

## References (papers & code)
**GVF theory**
- Kass, Witkin, Terzopoulos, "Snakes: Active Contour Models," IJCV 1(4):321–331, 1987/1988.
- Xu & Prince, "Gradient Vector Flow: A New External Force for Snakes," CVPR 1997, pp. 66–71. (iacl.ece.jhu.edu/pubs/p087c.pdf)
- Xu & Prince, "Snakes, Shapes, and Gradient Vector Flow," IEEE TIP 7(3):359–369, 1998. doi:10.1109/83.661186 (iacl.ece.jhu.edu/pubs/p084j.pdf)
- Xu & Prince, "Generalized Gradient Vector Flow External Forces for Active Contours," Signal Processing 71(2):131–139, 1998. (iacl.ece.jhu.edu/pubs/p105j.pdf)
- Qin, Zhu, Zhao et al., "Generalized GVF for Snakes: New Observations, Analysis, and Improvement (CN-GGVF)," IEEE TCSVT 23(5):883–897, 2013.
- Ning et al., "NGVF: an improved external force field," Pattern Recognition Letters 28:58–63, 2007.
- Wang, Liu, Zhang et al., "NBGVF / normally biased GVF," IEEE Signal Processing Letters 17(10):875–878, 2010.
- Li & Acton, "Active Contour External Force Using Vector Field Convolution (VFC)," IEEE TIP 16(8):2096–2106, 2007.
- Li, Xu, Gui, Fox, "Distance Regularized Level Set Evolution (DRLSE)," IEEE TIP 19(12):3243–3254, 2010.

**Deep learning + active contours**
- DACN: "Deep Active Contour Network for Medical Image Segmentation," MICCAI 2020.
- DALS: "Deep Active Lesion Segmentation," biorxiv/MICCAI-MLMI.
- Kim et al., "CNN-based Semantic Segmentation using Level Set Loss," arXiv:1910.00950.
- Kim & Ye, "Mumford-Shah Loss Functional for Image Segmentation with Deep Learning," arXiv:1904.02872.
- Kervadec et al., "Boundary loss for highly unbalanced segmentation," MIDL 2019.
- Ding et al., "Automatic Contour Refinement for DL Auto-segmentation" (ACR), *Advances in Radiation Oncology* 7(5):100968, 2022 (PMC9280040).

**OCTA / RNV**
- Ma et al., "ROSE: A Retinal OCT-Angiography Vessel Segmentation Dataset and New Model," IEEE TMI 40(3):928–939, 2021.
- Li et al., "OCTA-500: A retinal dataset for optical coherence tomography angiography study," *Medical Image Analysis* 2024 (arXiv:2012.07261).
- "Deep-OCTA: Ensemble DL for DR Analysis on OCTA (DRAC 2022)," arXiv:2210.00515.
- Untracht et al., "OCTAVA: An open-source toolbox for quantitative analysis of OCTA images," *PLOS One* 2021 (median + Frangi preprocessing).

**Metrics / class imbalance**
- Shit et al., "clDice — Topology-Preserving Loss for Tubular Structure Segmentation," CVPR 2021.
- Sudre et al., "Generalised Dice overlap as a DL loss for unbalanced segmentation," MICCAI-DLMIA 2017.
- Shi et al., "Centerline Boundary Dice Loss for Vascular Segmentation," MICCAI 2024.

**Code URLs**
- Xu-Prince GVF MATLAB: iacl.ece.jhu.edu/static/gvf/ ; mirrors github.com/fibrin/QNI (gvfv5), bitbucket.org/tetonedge/gvf, user.engineering.uiowa.edu/~image/Prince-snakes/. Functions: `[u,v]=GVF(f,mu,ITER)`, `[x,y]=snakedeform(x,y,alpha,beta,gamma,kappa,fx,fy,ITER)`, `[xi,yi]=snakeinterp(x,y,dmax,dmin)`, `snakeinit(delta)`, C-mex `gvfc`. (Research/educational; unsupported.)
- VFC / Parametric Active Model Toolbox (Bing Li), MATLAB File Exchange #22871: `AM_VFK`, `AM_VFC`, `AC_deform`, `AC_initial`, `AC_remesh`, `AM_GVF`, `AM_PIG`; examples `example_vfc.m`, `example_pig.m`. Requires Image Processing Toolbox; sign-in required to download.
- DRLSE MATLAB, File Exchange #12711 (Chunming Li); mirror github.com/balcilar/DRLSE-Image-Segmentation.
- cbDice/clDice reference: github.com/PengchengShi1220/cbDice.

---
*Note on sourcing: The exact JHU function signatures (`snakedeform`, `snakeinterp`, `GVF`) were confirmed verbatim from mirrored source; `snakedeform2`/`snakeinit`/`gvfc` and the VFC `AM_VFC`/`AM_VFK`/`AC_deform` headers are reconstructed from README/file-listing evidence and the source papers — verify against the downloaded `.m` files before finalizing code. The original Xu-Prince functions define no in-code defaults; the "default" values above come from the shipped example scripts and papers.*