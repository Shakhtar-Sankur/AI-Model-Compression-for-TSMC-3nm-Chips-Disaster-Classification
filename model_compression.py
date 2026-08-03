import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune
import torchvision.models as models
from torch.quantization import quantize_dynamic
import onnx
import onnxruntime as ort
import numpy as np
from typing import Tuple, Dict, Any
import time
import os

class MobileNetV3Compressed(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)

class ResNet50Teacher(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.backbone(x)

class ModelCompressor:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.student_model = None
        self.teacher_model = None
        self.compressed_model = None

    def create_models(self, num_classes: int = 4) -> Tuple[nn.Module, nn.Module]:
        self.student_model = MobileNetV3Compressed(num_classes).to(self.device)
        self.teacher_model = ResNet50Teacher(num_classes).to(self.device)
        return self.student_model, self.teacher_model

    def magnitude_pruning(self, model: nn.Module, sparsity: float = 0.5,
                          permanent: bool = False) -> nn.Module:
        """Globally prune the smallest-magnitude weights.

        `permanent=False` keeps torch's reparametrisation in place, so the mask is
        re-applied on every forward pass and the zeros survive training. Calling
        `prune.remove` straight away — as this used to — bakes the zeros into the
        weight and drops the mask, after which the very next optimiser step makes
        those weights non-zero again and the sparsity is silently lost.

        Call once with `permanent=True` at the end of training to strip the
        reparametrisation before export.
        """
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )

        if permanent:
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)

        return model

    @staticmethod
    def measure_sparsity(model: nn.Module) -> float:
        """Fraction of weights in Conv2d/Linear layers that are exactly zero."""
        zeros = total = 0
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                w = module.weight
                zeros += int((w == 0).sum().item())
                total += w.numel()
        return zeros / total if total else 0.0

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Dynamically quantise the Linear layers to int8.

        Note what this does and does not do. PyTorch's dynamic quantisation
        supports Linear, LSTM, GRU and RNN — it does *not* support Conv2d, and
        passing Conv2d in the spec is ignored without warning. MobileNetV3 is
        almost entirely convolutional, so dynamic quantisation alone moves very
        little of this model.

        Quantising the convolutions needs static (post-training) quantisation
        with a calibration pass over real data, or quantisation-aware training.
        Use `quantize_model_static` below for that.
        """
        model.eval()
        return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    def quantize_model_static(self, model: nn.Module, calibration_loader,
                              backend: str = 'fbgemm') -> nn.Module:
        """Post-training static quantisation, which does cover the convolutions.

        Runs a calibration pass so the observers can pick activation ranges.
        `calibration_loader` needs to yield batches from the training
        distribution; a few hundred images is usually enough.
        """
        import torch.ao.quantization as tq

        model.eval()
        model.qconfig = tq.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend

        prepared = tq.prepare(model, inplace=False)
        with torch.no_grad():
            for batch in calibration_loader:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                prepared(images)
        return tq.convert(prepared, inplace=False)

    def knowledge_distillation_loss(self, student_outputs, teacher_outputs, labels,
                                  temperature: float = 3.0, alpha: float = 0.7):
        soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
        soft_prob = F.log_softmax(student_outputs / temperature, dim=1)
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
        hard_loss = F.cross_entropy(student_outputs, labels)
        return alpha * soft_loss + (1 - alpha) * hard_loss

    def channel_width_plan(self, model: nn.Module, factor: float = 0.75,
                           min_channels: int = 16) -> Dict[str, int]:
        """Propose narrower channel counts for each convolution.

        This returns a *plan* and changes nothing. An earlier version assigned
        `module.out_channels = n` directly, which does not resize
        `module.weight` — the convolution kept computing at full width while the
        attribute reported the narrower one, so the model was unchanged and its
        metadata was wrong.

        Actually narrowing a network means rebuilding each layer at the new width
        and retraining, because the pretrained weights no longer fit. That is a
        search, not an in-place edit, and it is out of scope here.
        """
        plan = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > 32:
                proposed = int(module.out_channels * factor)
                if proposed >= min_channels:
                    plan[name] = proposed
        return plan

    def export_to_onnx(self, model: nn.Module, filepath: str, input_shape: Tuple = (1, 3, 224, 224)):
        model.eval()
        dummy_input = torch.randn(input_shape).to(self.device)
        torch.onnx.export(
            model,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        onnx_model = onnx.load(filepath)
        onnx.checker.check_model(onnx_model)

    def compress_pipeline(self, model: nn.Module, save_path: str = "models/",
                          sparsity: float = 0.5) -> Dict[str, Any]:
        """Prune, then quantise, then export — measuring the result at each step."""
        os.makedirs(save_path, exist_ok=True)

        original_parameters = sum(p.numel() for p in model.parameters())
        original_size_mb = self.get_model_info(model)["model_size_mb"]

        model = self.magnitude_pruning(model, sparsity=sparsity, permanent=True)
        achieved_sparsity = self.measure_sparsity(model)

        quantized_model = self.quantize_model(model)

        onnx_path = os.path.join(save_path, "compressed_mobilenet_v3.onnx")
        self.export_to_onnx(model, onnx_path)
        onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

        self.compressed_model = quantized_model
        return {
            "original_parameters": original_parameters,
            "original_size_mb": round(original_size_mb, 2),
            "onnx_size_mb": round(onnx_size_mb, 2),
            "size_reduction": round(1 - onnx_size_mb / original_size_mb, 3),
            "requested_sparsity": sparsity,
            "achieved_sparsity": round(achieved_sparsity, 3),
            "quantization": "int8 dynamic (Linear layers only)",
            "onnx_path": onnx_path,
            "model": quantized_model,
        }

    def benchmark_local(self, model: nn.Module, input_shape: Tuple = (1, 3, 224, 224)) -> Dict[str, float]:
        """Measure inference latency on *this* machine.

        This is not a 3nm accelerator and the numbers do not transfer to one.
        It is useful for comparing model variants against each other on identical
        hardware, which is what it is used for here.
        """
        model.eval()
        dummy_input = torch.randn(input_shape).to(self.device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        times = []
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        parameters = sum(p.numel() for p in model.parameters())
        return {
            "latency_ms_mean": float(np.mean(times)),
            "latency_ms_p95": float(np.percentile(times, 95)),
            "device": str(self.device),
            "parameters": parameters,
        }

    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "architecture": str(model.__class__.__name__)
        }

if __name__ == "__main__":
    compressor = ModelCompressor()
    student, teacher = compressor.create_models(num_classes=4)

    print("before:", compressor.get_model_info(student))
    results = compressor.compress_pipeline(student)
    for key, value in results.items():
        if key != "model":
            print(f"  {key}: {value}")
    print("latency:", compressor.benchmark_local(compressor.compressed_model))
