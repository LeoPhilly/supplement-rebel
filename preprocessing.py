"""Prep UltraFeedback-binarized for the REBEL/DPO/IPO comparison.

Loads HuggingFaceH4/ultrafeedback_binarized, filters ties + overlong pairs,
computes margin = score_chosen - score_rejected, saves the result to disk,
and prints the IPO-anchored eta for use as --model.eta.

"""

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main(split="train_prefs"):
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  vocab_size={tokenizer.vocab_size}  eos={tokenizer.eos_token!r}")

    print(f"\nLoading dataset: HuggingFaceH4/ultrafeedback_binarized ({split})")
    train = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    print(f"  rows={len(train)}  columns={train.column_names}")

    # Sanity-check the chat-template renders look right on one row.
    sample_chosen = tokenizer.apply_chat_template(train[0]["chosen"], tokenize=False)
    print("\n--- Rendered chosen (first 400 chars) ---")
    print(sample_chosen[:400])
    
    # --- Filter 1: ties ---
    # Strict > enforces margin > 0 for every remaining pair. A margin == 0 pair
    # would tell REBEL "make chosen/rejected equiprobable" while DPO/IPO push to
    # separate — a cross-method confound.
    before = len(train)
    train = train.filter(lambda ex: ex["score_chosen"] > ex["score_rejected"])
    after = len(train)
    print(f"\nFilter ties: {before} -> {after}  (dropped {before - after}, "
          f"{100*(before-after)/before:.1f}%)")
    
    # --- Filter 2: overlong pairs ---
    # Truncation corrupts magnitude (REBEL's signal) while only mildly affecting
    # sign (DPO/IPO). Dropping overlong pairs ensures the loss-comparison isn't
    # confounded by REBEL-disadvantaging label noise.
    MAX_LEN = 1024  # must match --data.max_len at training time

    def add_lengths(ex):
        c_ids = tokenizer.apply_chat_template(ex["chosen"], tokenize=True)["input_ids"]
        r_ids = tokenizer.apply_chat_template(ex["rejected"], tokenize=True)["input_ids"]
        return {"chosen_len": len(c_ids), "rejected_len": len(r_ids)}

    train = train.map(add_lengths, num_proc=4, load_from_cache_file=False)
    
    # Sanity check the lengths look right before filtering
    import numpy as np
    c_lens = np.array(train["chosen_len"])
    r_lens = np.array(train["rejected_len"])
    print(f"\nchosen_len   min={c_lens.min()}  max={c_lens.max()}  median={int(np.median(c_lens))}  p90={int(np.percentile(c_lens, 90))}  p99={int(np.percentile(c_lens, 99))}")
    print(f"rejected_len min={r_lens.min()}  max={r_lens.max()}  median={int(np.median(r_lens))}  p90={int(np.percentile(r_lens, 90))}  p99={int(np.percentile(r_lens, 99))}")
    print(f"# chosen   > 512: {int((c_lens > 512).sum())}")
    print(f"# rejected > 512: {int((r_lens > 512).sum())}")
    print(f"# chosen   > 1024: {int((c_lens > 1024).sum())}")
    print(f"# rejected > 1024: {int((r_lens > 1024).sum())}")

    before = len(train)
    train = train.filter(lambda ex: ex["chosen_len"] <= MAX_LEN and ex["rejected_len"] <= MAX_LEN)
    after = len(train)
    print(f"\nFilter overlong (max_len={MAX_LEN}): {before} -> {after}  "
          f"(dropped {before - after}, {100*(before-after)/before:.1f}%)")
    
    # --- Compute the margin column ---
    # margin = score_chosen - score_rejected, raw (no normalization).
    # The dataset code (reward_dataset.py) reads a column literally named `margin`.
    train = train.map(lambda ex: {"margin": ex["score_chosen"] - ex["score_rejected"]})

    margins = np.array(train["margin"])
    print(f"\nMargin distribution:")
    print(f"  min={margins.min():.2f}  max={margins.max():.2f}  median={float(np.median(margins)):.2f}  "
          f"mean={margins.mean():.2f}")
    print(f"  p10={float(np.percentile(margins, 10)):.2f}  "
          f"p50={float(np.percentile(margins, 50)):.2f}  "
          f"p90={float(np.percentile(margins, 90)):.2f}")
    print(f"  # margin == 0 (should be 0 — ties filtered): {int((margins == 0).sum())}")
    print(f"  # margin <  0 (should be 0 — strict > filter): {int((margins <  0).sum())}")

    # --- Compute IPO-anchored eta ---
    # eta = (1 / (2*beta)) / median(margin), so REBEL's typical target
    # eta * median(margin) equals IPO's target margin 1/(2*beta). This makes
    # all three losses (DPO, IPO, REBEL) request comparable log-ratio separation.
    BETA = 0.1  # must match --model.beta at training time

    median_margin = float(np.median(margins))
    ipo_target = 1.0 / (2.0 * BETA)
    eta_anchor = ipo_target / median_margin

    print(f"\neta calculation (IPO-anchored to beta={BETA}):")
    print(f"  IPO target margin (1/(2*beta)) = {ipo_target}")
    print(f"  median(margin) = {median_margin}")
    print(f"  --> --model.eta {eta_anchor:.4f}")
    print(f"  suggested sweep: {0.5*eta_anchor:.3f}  {eta_anchor:.3f}  {3*eta_anchor:.3f}")

    # --- Save ---
    # Drop the intermediate length columns to keep the saved dataset clean.
    # The dataset code only needs: prompt, chosen, rejected, margin (and score_* for traceability).
    OUTPUT_DIR = f"ultrafeedback-prepped-{split.replace('_prefs', '')}"
    train = train.remove_columns(["chosen_len", "rejected_len"])
    train.save_to_disk(OUTPUT_DIR)
    print(f"\nSaved {len(train)} rows to ./{OUTPUT_DIR}")
    print(f"  columns: {train.column_names}")
    
if __name__ == "__main__":
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else "train_prefs"
    main(split)