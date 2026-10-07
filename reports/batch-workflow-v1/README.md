# Batch review workflow

The folder tool reuses the upload application's frozen MobileNet/logistic prediction function. It exports a CSV, standalone HTML review queue with embedded previews, and source/model/input provenance. Rank is lowest P(good), then filename; `ceil(n × review_fraction)` sets an illustrative budget. Session names are user supplied. The selected classifier and accuracy evaluation are unchanged.

## Actual functional checks

The final October 7 run processed three selected recorded illustrations with the checkpoint that excludes their acquisition dates. On the development Apple M2 Max using MPS, it took **1.190327958 seconds after imports**. Maximum disagreement with saved example probabilities was **1.661127718×10⁻⁷**, below the frozen **10⁻⁵** tolerance. The report covers every input and orders poor, review, acceptable, with one frame in the illustrative budget. No fitting or download occurred.

The [manifest](final-run-manifest.json), [verification receipt](final-verification.json), [CSV](final-demo-report/predictions.csv), [HTML](final-demo-report/index.html) and [provenance](final-demo-report/provenance.json) retain exact hashes and results. Shared illustrations visibly credit Movčana et al., the Zenodo source, CC BY 4.0, and resizing/compression/annotation changes. These selected examples establish functional parity, not accuracy or representative throughput.

Three actual uploads to the existing HTTP route passed, and invalid image bytes were rejected. The owned temporary server shut down cleanly after SIGTERM: [HTTP receipt](web-route-verification.json). These checks ran once against the unchanged current `app.py`; they were not repeated. The initial [protocol](protocol.json) is preserved.

## Reproduce

Use a new output directory:

```bash
CHIPQC_MODEL_PATH=models/chipqc-demo-heldout.joblib \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m src.batch_review demo/capture/images \
  --output reports/a-new-attributed-demo-report \
  --session "Recorded illustrations from excluded dates" --review-fraction 0.3 \
  --image-credit "Movčana et al., Organ-on-a-Chip (OOC) Image Dataset, Zenodo 2023, CC BY 4.0" \
  --image-source-url https://doi.org/10.5281/zenodo.10203721 \
  --image-license-url https://creativecommons.org/licenses/by/4.0/

python3 -m src.verify_batch_review \
  --report reports/batch-workflow-v1/final-demo-report \
  --manifest reports/batch-workflow-v1/final-run-manifest.json \
  --output reports/batch-workflow-v1/reproduced-verification.json
```

The standard-library verifier checks current source/model bytes, saved output hashes, coverage, priority, attribution and recorded-example tolerance without inference or downloads. Add `--inputs demo/capture/images` to check the actual cached image bytes. Without that option, it checks recorded input-digest consistency. Changed source/model bytes deliberately refuse this frozen verification.

Appearance flags remain an unvalidated novelty heuristic. Unreviewed frames are not marked safe. Reviewer-decision logging and prospective evaluation remain future work; tissue function, drug response and clinical safety are not measured.
