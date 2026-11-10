# CLAUDE.md

RNV lesion segmentation from localization points, in MATLAB; the U-Net trains in PyTorch on
Colab (`colab/`). Read [README.md](README.md)
for what the repo is and the current number, [docs/SPEC.md](docs/SPEC.md) for how the
pipeline works, [docs/CONTEXT.md](docs/CONTEXT.md) for the vocabulary. This file holds
rules and pointers only — never a number or a parameter value; those live in one place
each, and a copy here would go stale.

## Two kinds of document

- **Living docs** describe the current state and must change whenever it does:
  `README.md`, `docs/SPEC.md`, `docs/CONTEXT.md`, `cnn/README.md`, this file.
- **Records** are frozen: `docs/results/*` (including `reference/`), `docs/briefs/*` and
  `docs/research_analysis/*` (literature-backed proposals for what to try next; not measurements).
  Never edit one, except to fix a link or add a banner saying what superseded it.

## Datasets

`datasets/` is the standardized data shared with the peer comparisons: images, masks and
unclustered localization points.

- **Nobody modifies it, the user or Claude.** Do not add, delete, rename, re-encode or edit
  any file in it, and that includes "fixing" a point or a mask. Handle its quirks in code
  (for example, the TUH CSV names `.jpg` and the files are `.png`).
- **Every image is evaluated.** Scoring covers every image in the domain, including an image
  with no points or with wrong points. Do not drop or filter an image from evaluation.
- Training may ignore incorrect images or points. That is the seed tolerance, for example.

## Changing code means changing docs — in the same commit

| when you change | update |
|---|---|
| pipeline behaviour, a default, or add/remove a `cfg` field | `docs/SPEC.md` |
| add, remove or rename a file or runner, or how to run something | `cnn/README.md` (layout and usage) |
| a domain term is introduced or shifts meaning | `docs/CONTEXT.md` (a glossary: no implementation detail) |
| you measure something | a **new** `docs/results/YYYY-MM-DD-<topic>.md` |
| the adopted configuration changes | a new `docs/results/reference/*.csv`, `cfg.CV.reference`, and README's "Current result" table and results index |
| an experiment is rejected | tag its code `archive/<name>`, delete its flag and code, record the result |
| a rule in this file goes stale | this file |

State the domain (17 / 41 / 68 images) and protocol with every number you write anywhere.

## Research rules — do not

- Quote a number that is not cross-validated. `run_batch` scores images that trained its model.
- Present `Split.Val` scores as held-out: `OutputNetwork="best-validation"` selects on them.
- Tune a threshold on the images being scored. Pre-register it; if it must be tuned, cross-fit it.
- Filter seeds at inference using ground truth. That is leakage.
- Compare numbers across domains or protocols.
- Change channel settings (`Denoise`, `Vessel`) without retraining: a network is tied to its channel count.
- Leave an experiment flag in the code after the decision. Adopted → it becomes the default
  and the switch goes. Rejected → tag and delete.

Why each exists: SPEC.md, "Reporting rules" and "Invariants that fail silently".

## Running an experiment

1. Write the prediction down **before** running. If it trains, costs more than ~15 min, or
   introduces a threshold, that means a brief in `docs/briefs/` with the expected result
   and a gate; otherwise the prediction goes at the top of the results record.
2. Work on a branch `exp/<name>`. Refactors, fixes and doc-only changes go on `master`.
3. Post-P changes: score them off the cache with `run_cache` (seconds, out-of-sample), not
   by retraining.
4. Changes that need training: train in `colab/`, once per arm, and judge the result against
   the noise floor and both baseline runs (SPEC.md, Reporting rule 7). The user runs Colab;
   prepare the code and the notebook cell, then score the downloaded `P` with `run_cache`.
   Read every fold's `log.csv` first: a stalled fold is a training failure, not a result.
5. Record the result, then adopt (merge; update SPEC, README, reference) or reject
   (tag `archive/<name>`, delete the branch after asking).

Commit subjects lead with the finding ("The seed fallback is worth +0.041 IoU"); the body
gives the domain and protocol behind every number.

## Done means verified

- `test_cnngvf` passes.
- Any change after `P`: `run_cache` must reproduce the reference (`max |dIoU|` ~1e-15). A
  real difference is a result — record it, and do not call the change a refactor.
- A runner itself changed: also smoke-run it on a couple of images (`cfg.Images`) with
  `cfg.Paths.Output` pointed at a scratch directory, since `run_batch` clears PNGs there.
- Docs changed: run the link check below.

## What you may run without asking

- Freely: `test_cnngvf`, `run_cache`, `run_eval`, `run_steps`, `run_batch` on a subset,
  `run_cv` with `cfg.CV.skipTrain = true` on a frozen arm, read-only git. The current
  configuration trains in `colab/` and `run_cv` refuses it; score it with `run_cache`.
- Ask first: `run_train`, `run_cv` that trains, anything expected to take over ~15 min.
- Commit freely, one logical change per commit. **Never push, force-push, or delete a
  branch or tag without asking.**

## Running MATLAB here

From `cnn/`, R2026a (R2025b lacks the toolboxes and fails immediately):

```bash
"/c/Program Files/MATLAB/R2026a/bin/matlab" -batch "test_cnngvf; run_cache"
```

Windows strips double quotes from `-batch` arguments. Anything with string literals goes in
a `.m` file in a scratch directory: `-batch "addpath('<dir>'); myscript"`. Do not use
`run('<file>')` — it changes directory and the package falls off the path.

## Code conventions

- Comments give the reason in a line and point to a doc. A measurement write-up in a comment
  is a record in the wrong place — put it in `docs/results/`.
- `+cnngvf`: one stage per file, every function takes `cfg`. `cnngvf.defaults()` is the
  current configuration; `cnngvf.arm` holds only the frozen arms.
- Runners take nothing (current configuration), a frozen arm's name, or a `cfg`.
- Vectorize: images are 1024×1024.

## When sources disagree

- Code vs SPEC: the code is the truth about behaviour — fix SPEC in the same change, or ask
  if you cannot tell whether the code is the bug.
- A new result contradicts a record: write a new record that says what it supersedes.
- A brief contradicts SPEC: stop and ask.

## Link check

Run from the repo root before committing doc changes; it must print nothing:

```bash
for f in $(git ls-files '*.md'); do d=$(dirname "$f"); grep -oE '\]\([^)#]+' "$f" | cut -c3- | grep -v '^http' | sort -u | while read l; do [ -e "$d/$l" ] || echo "BROKEN $f -> $l"; done; done
```

Note Windows is case-insensitive: a file renamed only by case (`spec.md` → `SPEC.md`) needs
`git rm --cached` on the old name, or git keeps the old casing.
