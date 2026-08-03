# AI Model Compression for TSMC 3nm Chips - Disaster Classification

Optimizing MobileNetV3-Small for TSMC's 3nm AI accelerators using advanced compression techniques to enable low-power disaster classification in edge environments.

## Features

- **Model Sparsity**: 50% reduction via magnitude pruning
- **Quantization**: 8-bit post-training quantization
- **Knowledge Distillation**: ResNet50 teacher model guidance
- **Architecture Search**: Neural Architecture Search (NAS) optimization
- **Performance**: <100ms latency at ~5W power consumption
- **Compatibility**: ONNX runtime support for TSMC AI accelerators

## Files

- `disaster_dataset.py` - Dataset handling and preprocessing
- `training_pipeline.py` - Model training workflow
- `model_compression.py` - Compression techniques implementation
- `inference_engine.py` - Optimized inference engine
- `smart_dashboard.py` - Monitoring and visualization


*Designed for edge AI disaster response systems on TSMC 3nm technology*