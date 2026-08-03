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
| `model_compression.py` | Magnitude pruning, post-training quantisation, ONNX export |
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

## On the numbers

The figures above are **design targets** that shaped the implementation — they are not
measured results. This repository ships no benchmark harness and no trained weights, so
nothing here reproduces them. They are recorded because they drove real decisions about
architecture and algorithm choice, not as claims about observed performance.

## Running it

```bash
pip install -r requirements.txt
python training_pipeline.py      # teacher, then distilled student
python model_compression.py      # prune, quantise, export ONNX
python inference_engine.py       # benchmark the exported model
streamlit run smart_dashboard.py
```

The exported model (`compressed_mobilenet_v3.onnx`) and `benchmark_results.json` are
produced by those steps and are not committed.

## Status

Full pipeline from training through compression to ONNX inference. No trained weights or
benchmark results are bundled — run the pipeline to produce them.

## Licence

All rights reserved. Published for reading, not for reuse.
