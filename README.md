# Edge AI Model Compression — Disaster Classification

**Compressing MobileNetV3-Small for low-power edge inference, targeting 3nm-class AI
accelerators, applied to disaster-scene classification.**

A disaster-response camera runs on a battery in a place where the network may be gone.
That constraint drives everything here: the model has to be small enough to fit, fast
enough to be useful, and cheap enough in power to run all day.

## What's here

| File | Role |
|---|---|
| `disaster_dataset.py` | Dataset loading, augmentation and preprocessing |
| `training_pipeline.py` | Trains the ResNet50 teacher, then the distilled student |
| `model_compression.py` | `ModelCompressor` — magnitude pruning, int8 quantisation (dynamic and static), ONNX export, local benchmarking |
| `inference_engine.py` | `ONNXInferenceEngine` — batched inference and latency reporting |
| `smart_dashboard.py` | Streamlit dashboard for comparing variants |

The compression order matters and is deliberate: distil first so the student learns the
teacher's soft targets while still dense, then prune, then quantise. Pruning a
distillation student loses less than distilling an already-sparse one.

## Design targets

- Roughly 8× smaller than the uncompressed baseline
- Accuracy held in the mid-nineties after compression
- Well under 100 ms per inference on the target accelerator
- Power budget around 5 W
- ONNX output, so the runtime is not tied to the training framework

## What it actually achieves today

Running `model_compression.py` on the untrained student, measured on this machine:

```
original (fp32)      5.81 MB   1,521,956 parameters
ONNX export          6.17 MB   (0.36 MB graph + 6.09 MB external weights)
size reduction        -6.3%    i.e. slightly larger
achieved sparsity     50.0%
latency (CPU)        11.7 ms mean, 13.0 ms p95
```

**The pipeline does not hit the 8x target, and it is worth being precise about why.**

Unstructured magnitude pruning sets weights to zero but leaves the tensor dense, so 50%
sparsity costs exactly as many bytes as 0% sparsity. Dynamic quantisation only covers
`Linear` layers, and MobileNetV3 is almost entirely convolutional — so the int8 conversion
touches a small fraction of the weights. ONNX then adds a little container overhead on top,
which is where the negative number comes from.

Closing that gap needs two changes: **static** quantisation with a calibration pass, which
does cover the convolutions and is worth close to 4x on their weights, and **structured**
pruning that removes whole channels so the tensors genuinely shrink.
`quantize_model_static()` implements the first; the second is not done here.

## Running it

```bash
pip install -r requirements.txt
python training_pipeline.py      # teacher, then distilled student
python model_compression.py      # prune, quantise, export ONNX
python inference_engine.py       # benchmark the exported model
streamlit run smart_dashboard.py
```

Outputs land in `models/` and are not committed: `compressed_mobilenet_v3.onnx` plus its
`.onnx.data` weight sidecar, and `benchmark_results.json`.

Note that `torch.onnx.export` needs `onnxscript` on torch 2.6 and later — it is in
`requirements.txt`, and the export step fails without it.

## Status

Runs end to end: training, pruning, quantisation, ONNX export and benchmarking. No trained
weights or benchmark results are bundled — run the pipeline to produce them.

Two things this repository deliberately does not do. There is no structured pruning, so
sparsity does not translate into a smaller file. And `channel_width_plan()` only proposes
narrower convolutions; it does not rebuild them, because doing so invalidates the
pretrained weights and requires a retraining search.

## Licence

Licensed under the GNU Affero General Public License v3.0. See `LICENSE`.

In short: you may use, modify and redistribute this, including over a network,
provided your derivative is released under the same licence.
