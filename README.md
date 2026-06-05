Supplementary materials for the REBEL-inspired reward-gap regression loss
PR in OpenRLHF. The code PR lives at:
https://github.com/LeoPhilly/OpenRLHF/tree/feat/rebel-loss-dpo

## Contents

- **design_doc.md** — full writeup: motivation, implementation choices,
  experiment setup, and results comparing DPO / IPO / REBEL on
  Qwen2.5-1.5B-Instruct + UltraFeedback.
- **wandb_charts.pdf** — supplementary figures: train loss, grad norm,
  eval accuracy, REBEL residual MSE.
- **preprocessing.py** — preprocessing script used to filter the
  dataset (drops ties and overlong sequences, computes `margin` column).

## Reproducibility

- Branch: `feat/rebel-loss-dpo` on `LeoPhilly/OpenRLHF`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Dataset: UltraFeedback (HuggingFaceH4/ultrafeedback_binarized), filtered
  via the script in this repo.
- Hardware: 1× H100 SXM5 (Lambda Labs)
- Seed: 42 (single seed)