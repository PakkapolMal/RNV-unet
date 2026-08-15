# CLAUDE.md — Theory Reference (RNV Segmentation)

Theory behind this repo, compiled from `/wiki`. Source notes: `wiki/index.md`,
`wiki/Mathematical_Formulas.md`, `wiki/Snake_Active_Contour.md`,
`wiki/RNV_Localization_OCTA.md`, `wiki/Enhancing_Fine_Vessel_Segmentation.md`,
`wiki/Unsupervised_Vessel_Segmentation_Curvelet_FCM.md`.

## Task

**RNV = retinal neovascularization**: abnormal, fragile, thin new vessels in
proliferative diabetic retinopathy. Imaged en face OCTA (superficial layer),
1024×1024, 12×12 mm. Types: *dense* (knots, beehives) and *loose* (medusa,
filamentous, fronds, meshes).

Two distinct problems — do not conflate them:

| | Localization | Segmentation |
|---|---|---|
| Output | a point per lesion | a boundary/mask per lesion |
| Method | Tun & Aimmanee 2024 (SVM_FM) | this repo: GVF snake seeded at those points |

The localization points are the *input* here: `RNV-DRAC22NV/DRAC22_RNV_Localization_Points_Sorted.csv`
gives `(img_name, cx, cy)`. This repo turns each seed point into a contour.

Hard parts: lesion size/pattern variability, RNV overlapping normal background
vasculature, IRMA/telangiectasia look-alikes, low contrast, noise near the optic
disk.

## 1. Active contours (the core model of this repo)

A snake is a parametric curve $v(s)=(x(s),y(s))$, $s\in[0,1]$, minimizing

$$E_{snake} = \int_0^1 \big[E_{int}(v(s)) + E_{ext}(v(s))\big]\,ds$$

$$E_{int}(s) = \tfrac{1}{2}\big(\alpha(s)|v'(s)|^2 + \beta(s)|v''(s)|^2\big)$$

- $\alpha$ — **membrane / elasticity** (1st order): resists stretching, pulls
  nodes together. `Options.Alpha`.
- $\beta$ — **thin-plate / rigidity** (2nd order): resists bending. `Options.Beta`.
  **Near-zero $\beta$ is what lets a contour enter deep concavities** — essential
  for loose/filamentous RNV, whose boundary is fern-like, not convex.

### External (image) energy

Kass's $E_{ext} = -|\nabla(G_\sigma * I)|^2$ is generalized in `ExternalForceImage2D.m` to
a weighted sum of three terms, all computed from Gaussian derivatives at scale `Sigma1`:

$$E_{ext} = W_{line}E_{line} - W_{edge}E_{edge} - W_{term}E_{term}$$

- $E_{line} = G_\sigma * I$ — attraction to bright lines ($W_{line}<0$ → dark lines).
- $E_{edge} = \sqrt{I_x^2+I_y^2}$ — gradient magnitude.
- $E_{term} = \dfrac{I_{yy}I_x^2 - 2I_{xy}I_xI_y + I_{xx}I_y^2}{(1+I_x^2+I_y^2)^{3/2}}$ — line
  terminations and corners (curvature of level lines).

Using a *smoothed* line/edge image rather than a raw gradient is deliberate: it
widens the capture range and stabilizes the evolution.

### Gradient Vector Flow (Xu & Prince)

The plain edge force only exists in a thin band around edges. GVF diffuses it
over the whole image. The field $\mathbf{g}=(u,v)$ minimizes

$$E_{GVF} = \iint \mu(u_x^2+u_y^2+v_x^2+v_y^2) + |\nabla f|^2|\mathbf{g}-\nabla f|^2\,dx\,dy$$

with $f$ the edge map. Euler–Lagrange gives the iteration in
`GVFOptimizeImageForces2D.m`:

$$u \leftarrow u + \mu\nabla^2 u - |\nabla f|^2(u - f_x)$$

Two regimes: where $|\nabla f|$ is large the field is pinned to $\nabla f$; where
it is small the Laplacian term smoothly extrapolates. That extrapolation is what
gives long capture range **and concavity convergence**.

`Mu` is the noise/edge trade-off. **`Mu > 0.5` makes the explicit scheme unstable**
(this is a CFL condition on the diffusion step, not a tuning preference).
`GIterations` controls how far the field diffuses — it must be large enough that
the field reaches the initial contour, or the snake never moves.

### Discrete evolution

Semi-implicit Euler with the internal-force (pentadiagonal, circulant) matrix $A$
built from $\alpha,\beta$:

$$v^{t} = (A + \gamma I)^{-1}\big(\gamma v^{t-1} + F_{ext}(v^{t-1})\big)$$

`SnakeInternalForceMatrix2D.m` precomputes $B=(A+\gamma I)^{-1}$ once;
`SnakeMoveIteration2D.m` applies it per step. Implicit in the internal force,
explicit in the external one — internal stiffness never limits $\gamma$.

The per-step external contribution is
$$F = \kappa\,\mathbf{g}(v) + \delta\,\mathbf{n}(v)$$

- $\kappa$ (`Kappa`) — GVF weight. Higher = harder edge snapping.
- $\delta$ (`Delta`) — **balloon force** along the unit normal $\mathbf{n}$
  (`GetContourNormals2D.m`). *Not* in Kass et al. Sign convention depends on
  contour orientation, hence `MakeContourClockwise2D.m`. Negative `Delta` =
  inward pressure, used here to stop the contour drifting outward into
  background vasculature when the image force is weak.

Balloon force is the standard cure for "initialized far from the boundary in a
flat region", but it fights the image force everywhere — it must be small
relative to $\kappa$ or it will collapse/inflate through real edges.

3D analogues (`Snake3D.m`, `SnakeInternalForceMatrix3D.m`) use the inverse
internal-force matrix for stability plus a mesh self-intersection guard
(`removeContourSelfIntersections.m`).

## 2. Initialization: ray casting

A snake only finds the boundary it starts near. Given a localization seed
$(c_x,c_y)$, cast `n_points` rays outward, and per ray find the radius where the
image profile drops past a peak (`peak_drop_tol`, default 15%), clamped to
`[r_min, r_max]` and ignoring peaks inside `min_peak_radius`; smooth the
resulting radius profile (`smoothing_window`) and use it as the initial contour.

This replaces a single global radius (`adaptive_gvf.m`, flagged redundant) with a
per-angle boundary estimate — important because RNV lesions are markedly
non-circular. Smoothing the radius profile matters: an unsmoothed profile injects
high-curvature noise that a low-$\beta$ snake will happily preserve.

## 3. Vesselness front-ends

`rnv_gvf_ray_filter.m` prepends a **Frangi** multiscale vesselness filter; the
edge map is then computed on the vesselness response rather than raw intensity.
Rationale: RNV is a vessel structure, and a vesselness response suppresses the
background texture the raw gradient would latch onto.

The wiki documents three related enhancement families, all serving the same goal:

- **Frangi multiscale vesselness** — used by Tun & Aimmanee across three scale
  channels (large/medium/small); Frangi sensitivity 25/15/5, σ ranges 38–42 /
  11–15 / 1–11.
- **Oriented 2nd-order LoG** (Abdushkour et al.) — elongated anisotropic Gaussian
  with independent length $\sigma_u$ and width $\sigma_v$; take the max response
  over (length, width, orientation):
  $$G_{uu}(u,v) = \frac{1}{2\pi\sigma_u\sigma_v}\Big(\frac{u^2}{\sigma_u^4}-\frac{1}{\sigma_u^2}\Big)e^{-\frac{u^2}{2\sigma_u^2}-\frac{v^2}{2\sigma_v^2}}$$
  Scale-normalization exponents: Lindeberg's ideal ridge is $\alpha=1.5,\beta=0.5$;
  **for thin, low-contrast vessels use $\alpha=1,\beta=0.5$** — trades accuracy for
  sensitivity. That trade is the right one for RNV.
- **Curvelet** (Chalakkal & Abdulla) — zero the coarse approximation
  coefficients, amplify the remaining subbands, invert.

⚠️ Naming collision: LoG scale-normalization $\alpha,\beta$ are *unrelated* to
snake $\alpha,\beta$ (`Options.Alpha`/`Beta`).

## 4. The three RNV biomarkers (Tun & Aimmanee 2024)

The localization method that produces this repo's seed points. Over a 21×21
moving window of area $A$, unit step:

$$D_v = \frac{N_v}{A}\quad\text{(vessel density)}\qquad D_b = \frac{N_b}{A}\quad\text{(bifurcation density)}$$

$$D_t(x,y) = \frac{1}{A}\sum_{i=-10}^{10}\sum_{j=-10}^{10} T(x+i,y+j)\quad\text{(avg thinness)}$$

where thinness at a skeleton point $c$ is $\dfrac{1}{2\cdot\text{shortestDist}(c,E)}$,
$E$ = vessel edge set via Euclidean distance transform; skeleton by
medial-surface-axis thinning (Lee et al.). Bifurcations = skeleton points with 3
connected non-end neighbors, all pairwise angles > 45°.

**Thinness is the novel feature** — new RNV vessels are thinner than normal
vasculature — and it is what lifted the method past the authors' earlier WCFM
(which could only find one lesion per image).

Then: min–max normalize each map (rank-preserving), threshold at the 95th
percentile, argmax per connected cluster → candidate points; per candidate, a
20×20 window on each map gives {min, max, mean, std} → 12 features; **Gaussian
SVM** classifies; **DBSCAN** groups survivors spatially with representative

$$loc(g) = \arg\max_{u\in g}\big[D_v(u)+D_b(u)+D_t(u)\big]$$

Result: precision 90.33%, F1 ≈ 85%, miss rate 18.92% over 69 images — beating
WCFM by +44% F1 and VNet by +57% F1.

**Deep learning lost here.** VNet over-segments because RNV data is scarce.
Classical features + SVM win on small datasets — the standing justification for
this repo being classical.

Known failure modes, inherited by anything seeded from these points:
under-localization when two lesions are close (DBSCAN merges them), and
over-localization from filament noise near the optic disk mis-segmented as vessel.

## 5. Preprocessing theory (reference)

Used by the localization literature; relevant if the snake front-end is revised.

**Selective linear mapping** (OCTA contrast, optimal $t_1=0.2$, $t_2=0.8$):

$$I_{new} = \begin{cases} 0 & 0\le I_o < t_1\\ \frac{I_o-t_1}{t_2-t_1}I_{max} & t_1\le I_o<t_2\\ I_{max} & t_2\le I_o<I_{max}\end{cases}$$

**Niblack** local threshold $T(x,y)=\mu(x,y)+k\beta(x,y)$; per Frangi channel
(large/medium/small): window 96/32/8 px, $k$ = 1.2/0.7/0.7.

**Homomorphic filtering** — $I = L\cdot R$, so $\ln I = \ln L + \ln R$; Gaussian
high-pass $H(u,v) = (\gamma_H-\gamma_L)[1-e^{-cD^2/D_0^2}]+\gamma_L$ kills the
low-frequency illumination, keeps reflectance; inverse FT then exp.

**Adaptive Wiener** — $\hat I = \mu + \frac{\sigma^2-\nu^2}{\sigma^2}(I-\mu)$;
smooths flat regions, spares edges.

**Top/bottom-hat** — $T_{hat}=I-(I\circ b)$, $B_{hat}=(I\bullet b)-I$; top-hat
minus bottom-hat suppresses the vessel center light reflex.

**Double thresholds** — marker $T_U = \mu - 0.7\sigma$ of the coherence
histogram, mask $T_L$ = mean of the edge histogram; combine by morphological
reconstruction, drop components < 70 px.

**FCM** — $Q = \sum_i\sum_j u_{ij}^m\|v_i-\mu_j\|^2$ with $C=3$, $m=2$
(vessel / OD+exudate / background), $\sum_j u_{ij}=1$.

Recurring theme across all four papers: **optic disk / light-reflex suppression**
is the dominant error source, and **thin-vessel sensitivity** is the standing open
problem — which is exactly the regime RNV lives in.

## 6. Parameter cheat-sheet

`Snake2D.m` options, with theory role and the tuned RNV values from
`rnv_gvf_ray.m`:

| Option | Role | Kroon default | RNV value |
|---|---|---|---|
| `Alpha` | membrane / elasticity | 0.2 | 0.07 |
| `Beta` | thin-plate / rigidity | 0.2 | 0.01 — allows deep concavities |
| `Delta` | balloon force | 0.1 | −0.55 — inward, resists outward drift |
| `Kappa` | external field weight | 2 | 8 — strong GVF snapping |
| `Mu` | GVF noise/edge trade-off | 0.2 | 0.15 (**keep < 0.5**, stability) |
| `GIterations` | GVF diffusion steps | 0 | 800 |
| `Iterations` | contour evolution steps | 100 | 600 |
| `Gamma` | time step $\gamma$ | 1 | default |
| `Sigma1` | scale for image derivatives | 10 | 3 |
| `Sigma2` | scale for edge-energy gradient | 20 | 3 |
| `Sigma3` | scale for GVF Laplacian | 1.0 | default |
| `Wline` | line attraction (sign = polarity) | 0.04 | 0.4 |
| `Wedge` | edge attraction | 2.0 | 4 |
| `Wterm` | termination/corner attraction | 0.01 | 0 |

Small `Sigma1`/`Sigma2` (3 vs. 10/20) reflect the fine scale of RNV vessels;
large sigmas would blur the structure away entirely.

Empirical tuning sweeps for `Alpha`/`Beta`/`Delta`/`Kappa` live in
`RNV-Cursor/tuning_results_ray/` and `tuning_results_adaptive/`
(`best_params*.txt`, `tuning_log*.csv`, per-parameter sweep plots).

## Conventions

- `/raw` = immutable source PDFs; `/wiki` = knowledge base (update it when a new
  paper is ingested, and log in `wiki/log.md`); `Fast-Gradient-Vector-Flow-main/`
  = MATLAB implementation.
- **Vectorize.** Matrix operations over per-pixel loops — retinal images are
  1024×1024 and the GVF runs 800 iterations.
- Consult the relevant wiki note before writing new MATLAB.
- `adaptive_gvf.m` and `rnv_gvf_ray.m` carry `[REDUNDANT]` headers from the
  2026-06-14 review; `rnv_gvf_ray_filter.m` is the current production runner.
