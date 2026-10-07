# Verification of saved ChipQC predictions

Run `python3 -m src.verify_metrics --output reports/metrics-verification.json`
from the project directory. This uses only Python's standard library and the
published CSV and evaluation JSON. It checks their full SHA-256 hashes and
compares recomputed metrics with the saved evaluation at a tolerance of 1e-12.
The [JSON receipt](metrics-verification.json) records the source hash and results.

The October 7 run passed in **0.047 seconds** for **3,072 frames / 59 acquisition
dates**. All five models' date-held-out pooled AUC, average precision, balanced
accuracy, Brier score and confusion counts match. The selected MobileNet logistic
model's AUC is **0.7476** on date-held-out predictions and **0.8159** on its saved
random-fold predictions. The illustrative 0.2/0.8 score bands also match the
original report. Of the 793 frames with P(good) ≥ 0.8, **16.9%** have the bad label;
the score is unsuitable for automatic acceptance without scientist review.

## Retrospective review-queue comparison

Both queues rank ascending P(good), then image ID, using no labels to select
frames. Each reviews **922/3,072 frames**: ceil(30% of all frames). The per-date
allocation starts with floor(30% of each date's size), then distributes the
remaining slots by largest fractional remainder, with date-ID ties.

| Queue | Bad frames found / 1,345 | Bad-frame recall | Bad share of reviewed frames |
|---|---:|---:|---:|
| Rank within each acquisition date | 508 | 37.8% | 55.1% |
| Random review with the same per-date quotas, expectation | 404.45 | 30.1% | — |
| Rank a pooled backlog across all dates | 645 | 48.0% | 70.0% |

The **37.8%** result corresponds to reviewing a quota within each imaging date.
The **48.0%** result describes a pooled backlog, whose scores come from several
fold models. These are descriptive analyses of saved out-of-fold predictions.
They do not measure prospective workflow benefit, time saved, or a deployed
batch-review tool. No confidence interval was computed for these queue results.

## Verification scope

No original model or fold was retrained, and bootstrap confidence intervals were
not recomputed. The original evaluation and limitations remain in the
[technical report](TECHNICAL_REPORT.md). Acquisition date is an image-ID proxy,
not an independently confirmed chip, donor or site identifier. Agreement with
the expert image-quality label does not establish biological function or safety.
