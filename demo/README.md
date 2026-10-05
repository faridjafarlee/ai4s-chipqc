# Demonstration evidence

The video demonstrates the running FastAPI application using the first
acquisition-date-held-out fold's model, not the classifier refitted on the full
dataset. Generate its cases with `python -m src.prepare_demo` after evaluation.

`examples.json` records the selected source image IDs, expert labels, model
probabilities, and acquisition-date prefixes. The model's training dates and
image IDs exclude every recorded example. These cases illustrate the UI;
performance claims come from all out-of-fold predictions in the report.

The completed 1 minute 55 second recording shows an acceptable estimate, a poor estimate, an
uncertain estimate requiring human review, the validation comparison, and
the reproduction commands. It credits the source microscopy dataset
(Movčana et al., Zenodo 2023, DOI 10.5281/zenodo.10203721, CC BY 4.0).

Start the recording checkpoint with:

```bash
CHIPQC_MODEL_PATH=models/chipqc-demo-heldout.joblib .venv/bin/uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Raw captures and copied microscopy frames are excluded from Git. The final
video and its captions are provided as demonstration artifacts.

Watch `chipqc-demo.mp4`; the captions are burned into the application capture.
`chipqc-demo.srt` is also available. The video is silent. Rendering uses Pillow
and a locally installed FFmpeg; rerender with `python -m src.render_demo` when
the ignored `capture/` source frames are available. Reproducing the model and
application does not require FFmpeg or the original recording frames.
