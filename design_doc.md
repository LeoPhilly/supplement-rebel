# Design Doc: REBEL-inspired reward-gap regression loss in OpenRLHF

**Paper:** REBEL — Reinforcement Learning via Regressing Relative Rewards

**Citation:** https://arxiv.org/abs/2404.16767

**PR Link**: https://github.com/OpenRLHF/OpenRLHF/pull/1247

**Summary:** Implemented an offline reward-gap regression loss inspired by REBEL (citation above) into OpenRLHF and compared it to DPO & IPO on Qwen2.5-1.5B-Instruct + UltraFeedback dataset. Note that this is not a paper-faithful implementation of the full REBEL algorithm; it is an offline, one-step proxy inspired by REBEL’s reward-gap regression objective.

## Contents

- [Design Doc: REBEL-inspired reward-gap regression loss in OpenRLHF](#design-doc-rebel-inspired-reward-gap-regression-loss-in-openrlhf)
  - [Contents](#contents)
  - [Why REBEL inspired loss and not the actual paper faithful REBEL?](#why-rebel-inspired-loss-and-not-the-actual-paper-faithful-rebel)
  - [How is this loss different from DPO \& IPO?](#how-is-this-loss-different-from-dpo--ipo)
  - [Implementation Details in OpenRLHF](#implementation-details-in-openrlhf)
    - [Why a new REBEL Loss class instead of branching from the DPO class like IPO implementation did?](#why-a-new-rebel-loss-class-instead-of-branching-from-the-dpo-class-like-ipo-implementation-did)
    - [Why is your Loss calculation `(logits - eta * rewards_margin)**2` instead of the following the paper's `((1/eta) * logits - rewards_margin)**2`](#why-is-your-loss-calculation-logits---eta--rewards_margin2-instead-of-the-following-the-papers-1eta--logits---rewards_margin2)
    - [What testing did you do of your implementation?](#what-testing-did-you-do-of-your-implementation)
  - [Experiment Details](#experiment-details)
    - [How was the initial eta of 3.33 chosen?](#how-was-the-initial-eta-of-333-chosen)
    - [Preprocessing](#preprocessing)
    - [Key Caveats](#key-caveats)
    - [Experiment Config:](#experiment-config)
    - [Wandb Results -- DPO vs IPO vs REBEL:](#wandb-results----dpo-vs-ipo-vs-rebel)
    - [How was AI used and what did it correct/wrong?](#how-was-ai-used-and-what-did-it-correctwrong)

## Why REBEL inspired loss and not the actual paper faithful REBEL?

The goal is to test how the REBEL inspired reward-difference regression loss does compared to DPO & IPO inside OpenRLHF training set-up. Note that this is not the actual REBEL algorithm as described in the paper. At an approximation, the paper describes the actual loss via online (both generations by the current policy) and hybrid (one policy from dataset; the other via generation). This implementation would be a close proxy to what an offline version would look like. However, the guarantees demonstrated in the paper (for example, the convergence guarantee) do not hold in this case, as even in this approx offline implementation, there is no policy iteration. Note that I'm referring to this rebel inspired loss when I say "REBEL loss" in this document.

## How is this loss different from DPO & IPO?

The main difference is that DPO & IPO are preference losses and hence only factor in the direction of the rewards signal via chosen/rejected pairs, whereas this REBEL loss additionally factors in the magnitude of the difference in rewards as well. Conceptually, this can provide a richer training signal, especially for pairs where the chosen/rejected reward difference is large.  

## Implementation Details in OpenRLHF

### Why a new REBEL Loss class instead of branching from the DPO class like IPO implementation did?

Three main reasons:

1. Both DPO and IPO take the same input arguments, and most notably require reference model logits. REBEL, however, requires `old_policy_logits`.
2. IPO additionally needed no extra inputs as compared to DPO, whereas REBEL requires an extra `rewards_margin` (difference of rewards) argument. Secondly, REBEL has an eta parameter, which is separate from the beta parameter that both DPO & IPO use.
3. Conceptually, IPO is a preference loss and hence similar to DPO, whereas REBEL regresses onto rewards magnitude differences.

The above conditions made the current design of adding a new REBEL loss more appropriate from a readability & implementation perspective as opposed to adding 3 extra arguments with defaults to the DPO implementation.

Note that in this iter-1 offline proxy, the old policy is the frozen (SFT) initial model, so these coincide with the reference logps that are used in DPO. This distinction is mostly theoretical here, but would matter more for an online/hybrid version of REBEL (as mentioned in the paper). 

### Why is your Loss calculation `(logits - eta * rewards_margin)**2` instead of the following the paper's `((1/eta) * logits - rewards_margin)**2`

First, note that the optimum/minimizer is the same for both since one is a scalar multiple of the other (specifically scaled by `1/eta**2`). The difference is to be consistent with DPO & IPO's learning rate. If I implemented the paper's loss, the loss magnitude would be deflated by `eta**2` and gradients would be small for a large eta. Hence, REBEL would converge slower than DPO/IPO under identical learning rates. Additionally, this should keep the loss and gradient scale comparable to DPO/IPO. (Empirically, REBEL's gradients still end up ~240× larger than DPO's even with this normalization — see results below) 

### What testing did you do of your implementation?

I added the following 6 unit tests directly into OpenRLHF in the `test_rebel_loss` file. The tests are:

1. Zero loss when prediction matches target
2. Positive loss when off-target
3. Gradient step moves logits toward target
4. Returned rewards-difference equals logits
5. Acc reduces to sign of logits
6. Invalid eta raises `ValueError`

Furthermore, I did a smoke test with Qwen2.5-1.5B-Instruct and ultrafeedback dataset to make sure that the OpenRLHF implementation works on a small scale model. Lastly, I also compared it to a full run with DPO & IPO (more in the experiment section below) on the same model & dataset. The results of this can be seen in the wandb report linked.

## Experiment Details

### How was the initial eta of 3.33 chosen?

η = 3.333, anchored to IPO. IPO's constant target is $\frac{1}{2\beta} = 5.0$ at $\beta = 0.1$. Setting REBEL's target at the *median* reward diff (margin in the code) equal to IPO's target:

$$\eta = \frac{1/(2\beta)}{\text{median}(\Delta r)} = \frac{5.0}{1.5} = 3.333$$

### Preprocessing

Using the default `max_len == 512` would have thrown away ~43% of the data, which is quite high. Therefore, I used `max_len == 1024`, which only drops 6% of the data. Moreover, the marginal cost of running with `max_len = 1024` was minimal, and the results would be more accurate too since I didn't want to disproportionately test on smaller length sequences. Ultrafeeback also has a well known length bias which clearly shows here. One additional aspect of preprocessing – I also dropped ties from the data (about 12% of the data) since REBEL loss needs a reward difference in order to meaningfully work.

The full preprocessing code is available on GitHub: [`preprocessing.py`](https://github.com/LeoPhilly/supplement-rebel/blob/main/preprocessing.py).

### Key Caveats

- Reward diff (margin) was calculated as `score_chosen - score_rejected` with no additional normalization or transformation.
- Note that this is not the full REBEL implementation as mentioned in the paper, and is only an inspired loss version that could be used as 1-iter offline version of the paper's loss.
-  The comparison run on REBEL vs DPO vs IPO is intended to show correct implementation, and is not a full scale controlled experiment or benchmark.
- This was a single seed run. 

### Experiment Config:

- **Branch:** `feat/rebel-loss-dpo` (pushed to LeoPhilly/OpenRLHF)
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Dataset:** `ultrafeedback-prepped/` (50387 rows, includes margin column)
- **beta** = `0.1` (all three runs)
- **eta** = `3.3333` (REBEL only; IPO-anchored to beta)
- **max_len** = `1024` (matches prep filter)
- **Shared across runs:** seed, LR, LoRA config, batch size, data
- **DPO:** `--model.beta 0.1`
- **IPO:** `--model.beta 0.1 --model.ipo_enable`
- **REBEL:** `--model.beta 0.1 --model.rebel_enable --model.eta 3.3333`

### Wandb Results -- DPO vs IPO vs REBEL:

Link to other wandb charts: [`wandb_charts.pdf`](https://github.com/LeoPhilly/supplement-rebel/blob/main/Wandb_charts.pdf).

**Setup:** 4096 train samples (512 steps), 1631 held-out eval examples, seed 42 & identical config across runs except for loss. 

**Headline (eval acc on held-out 1631):**

| Loss | eval/acc_mean | eval/loss | final train/loss | final train/grad_norm |
|---|---|---|---|---|
| DPO | 0.730 | 0.538 | 0.516 | 7.81 |
| IPO | 0.686 | 20.68 | 17.34 | 1102.46 |
| REBEL (η=3.333) | 0.719 | 54.02 | 11.31 | 1868.37 |

**Training dynamics (first-half vs second-half mean train loss):**

- DPO: 0.570 → 0.597 
- IPO: 22.44 → 22.47 
- REBEL: 59.62 → 51.72

**Notes:**

- REBEL's final grad norm is roughly 240× larger than DPO's and 1.7×  larger than IPO's. 
- Loss magnitudes are not directly comparable across the three as these are different objectives(DPO is logistic; IPO and REBEL are squared but on different targets).
- Eval acc places REBEL (71.9%) between DPO (73.0%) and IPO (68.6%) and hence is comparable.
- Note that since DPO & IPO have preference based objectives, the loss values are not directly comparable. 


### How was AI used and what did it correct/wrong?

AI was very useful for writing basic code integration, tests and experiments setup (such as preprocessing and GPU debugging), but needed a few corrections for the actual algorithm writing. For example, I had to correct the distinction between the full online/hybrid REBEL algorithm as mentioned in the paper and this offline REBEL-inspired proxy, and clarify that this implementation uses frozen reference-model logprobs as the old-policy baseline. 
