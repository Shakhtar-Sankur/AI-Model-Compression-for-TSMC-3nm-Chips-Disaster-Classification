import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
import os
from pathlib import Path
import json
from datetime import datetime

from model_compression import ModelCompressor, MobileNetV3Compressed, ResNet50Teacher
from disaster_dataset import DatasetManager

class TSMCPerformanceSimulator:
    def __init__(self):
        self.base_latency = 2.5
        self.base_power = 0.8
        self.memory_bandwidth = 1024
        
    def estimate_performance(self, model: nn.Module, input_shape: Tuple = (1, 3, 224, 224)) -> Dict[str, float]:
        total_params = sum(p.numel() for p in model.parameters())
        flops = self._estimate_flops(model, input_shape)
        latency_ms = (flops / 1e9) * self.base_latency
        power_w = (total_params / 1e6) * self.base_power
        if hasattr(model, 'compression_ratio'):
            latency_ms *= (1 - model.compression_ratio * 0.3)
            power_w *= (1 - model.compression_ratio * 0.4)
        return {
            'latency_ms': latency_ms,
            'power_w': power_w,
            'flops': flops,
            'parameters': total_params,
            'meets_targets': latency_ms < 100 and power_w < 5.0
        }
    
    def _estimate_flops(self, model: nn.Module, input_shape: Tuple) -> float:
        total_flops = 0
        def hook(module, input, output):
            nonlocal total_flops
            if isinstance(module, nn.Conv2d):
                output_elements = output.numel()
                kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                total_flops += output_elements * kernel_flops
            elif isinstance(module, nn.Linear):
                total_flops += module.in_features * module.out_features
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(hook))
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(input_shape)
            if next(model.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()
            _ = model(dummy_input)
        for hook in hooks:
            hook.remove()
        return total_flops

class KnowledgeDistillationTrainer:
    def __init__(self, student: nn.Module, teacher: nn.Module, device: str):
        self.student = student
        self.teacher = teacher
        self.device = device
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def distillation_loss(self, student_outputs, teacher_outputs, labels, 
                         temperature: float = 3.0, alpha: float = 0.7):
        soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
        soft_prob = F.log_softmax(student_outputs / temperature, dim=1)
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
        hard_loss = F.cross_entropy(student_outputs, labels)
        total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
        return total_loss, soft_loss, hard_loss

class TrainingPipeline:
    def __init__(self, output_dir: str = "results", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.compressor = ModelCompressor(self.device)
        self.dataset_manager = DatasetManager()
        self.performance_simulator = TSMCPerformanceSimulator()
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'distillation_loss': [],
            'latency_ms': [],
            'power_w': []
        }
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        
    def setup_models_and_data(self, num_classes: int = 4, batch_size: int = 32) -> Dict:
        self.student_model, self.teacher_model = self.compressor.create_models(num_classes)
        self.dataloaders = self.dataset_manager.create_dataloaders(
            batch_size=batch_size, 
            num_workers=2
        )
        self.kd_trainer = KnowledgeDistillationTrainer(
            self.student_model, self.teacher_model, self.device
        )
        return {
            'student_params': sum(p.numel() for p in self.student_model.parameters()),
            'teacher_params': sum(p.numel() for p in self.teacher_model.parameters()),
            'train_samples': len(self.dataloaders['train'].dataset),
            'val_samples': len(self.dataloaders['val'].dataset)
        }
    
    def train_teacher_model(self, epochs: int = 10, lr: float = 0.001) -> Dict[str, float]:
        optimizer = optim.Adam(self.teacher_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        for epoch in range(epochs):
            self.teacher_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for batch_idx, (data, targets) in enumerate(self.dataloaders['train']):
                data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.teacher_model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            self.teacher_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, targets in self.dataloaders['val']:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = self.teacher_model(data)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.teacher_model.state_dict(), 
                          self.output_dir / 'best_teacher_model.pth')
            scheduler.step()
        return {'best_accuracy': best_acc, 'final_accuracy': val_acc}
    
    def train_student_with_distillation(self, epochs: int = 20, lr: float = 0.001,
                                       temperature: float = 3.0, alpha: float = 0.7) -> Dict:
        optimizer = optim.Adam(self.student_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best_acc = 0.0
        compression_schedule = np.linspace(0.0, 0.5, epochs)
        for epoch in range(epochs):
            if epoch > 5:
                current_sparsity = compression_schedule[epoch]
                if current_sparsity > 0.1:
                    self.student_model = self.compressor.magnitude_pruning(
                        self.student_model, sparsity=current_sparsity
                    )
            self.student_model.train()
            train_loss = 0.0
            distill_loss_total = 0.0
            train_correct = 0
            train_total = 0
            for batch_idx, (data, targets) in enumerate(self.dataloaders['train']):
                data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                student_outputs = self.student_model(data)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(data)
                total_loss, soft_loss, hard_loss = self.kd_trainer.distillation_loss(
                    student_outputs, teacher_outputs, targets, temperature, alpha
                )
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()
                distill_loss_total += soft_loss.item()
                _, predicted = student_outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            val_metrics = self.validate_model(self.student_model)
            perf_metrics = self.performance_simulator.estimate_performance(self.student_model)
            train_acc = 100. * train_correct / train_total
            train_loss_avg = train_loss / len(self.dataloaders['train'])
            distill_loss_avg = distill_loss_total / len(self.dataloaders['train'])
            self.training_history['train_loss'].append(train_loss_avg)
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['distillation_loss'].append(distill_loss_avg)
            self.training_history['latency_ms'].append(perf_metrics['latency_ms'])
            self.training_history['power_w'].append(perf_metrics['power_w'])
            self.writer.add_scalar('Loss/Train', train_loss_avg, epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Performance/Latency_ms', perf_metrics['latency_ms'], epoch)
            self.writer.add_scalar('Performance/Power_W', perf_metrics['power_w'], epoch)
            self.writer.add_scalar('Distillation/Soft_Loss', distill_loss_avg, epoch)
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                torch.save({
                    'model_state_dict': self.student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'accuracy': best_acc,
                    'performance': perf_metrics
                }, self.output_dir / 'best_student_model.pth')
            scheduler.step()
        return {
            'best_accuracy': best_acc,
            'final_performance': perf_metrics,
            'compression_achieved': compression_schedule[-1]
        }
    
    def validate_model(self, model: nn.Module) -> Dict[str, float]:
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, targets in self.dataloaders['val']:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        return {
            'loss': val_loss / len(self.dataloaders['val']),
            'accuracy': 100. * val_correct / val_total,
            'total_samples': val_total
        }
    
    def complete_optimization_pipeline(self) -> Dict:
        setup_info = self.setup_models_and_data()
        teacher_results = self.train_teacher_model(epochs=10)
        student_results = self.train_student_with_distillation(epochs=20)
        compression_results = self.compressor.compress_pipeline(
            self.student_model, 
            save_path=str(self.output_dir / "models")
        )
        final_performance = self.compressor.simulate_tsmc_3nm_performance(
            self.compressor.compressed_model
        )
        final_results = {
            'teacher_accuracy': teacher_results['best_accuracy'],
            'student_accuracy': student_results['best_accuracy'],
            'compression_ratio': compression_results['compression_ratio'],
            'final_parameters': final_performance['parameters'],
            'latency_ms': final_performance['latency_ms'],
            'power_w': final_performance['power_consumption_w'],
            'targets_met': final_performance['target_latency_met'] and final_performance['target_power_met'],
            'onnx_model_path': compression_results['onnx_path'],
            'training_history': self.training_history
        }
        self.save_results(final_results)
        self.plot_training_curves()
        return final_results
    
    def save_results(self, results: Dict):
        results_file = self.output_dir / "training_results.json"
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                   for k, v in value.items()}
            else:
                json_results[key] = value
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('TSMC 3nm AI Accelerator Training Results', fontsize=16)
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.training_history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 1].plot(epochs, self.training_history['train_acc'], 'b-', label='Train')
        axes[0, 1].plot(epochs, self.training_history['val_acc'], 'r-', label='Validation')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 2].plot(epochs, self.training_history['distillation_loss'], 'g-')
        axes[0, 2].set_title('Knowledge Distillation Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('KD Loss')
        axes[0, 2].grid(True)
        axes[1, 0].plot(epochs, self.training_history['latency_ms'], 'purple', linewidth=2)
        axes[1, 0].axhline(y=100, color='red', linestyle='--', label='Target (100ms)')
        axes[1, 0].set_title('TSMC 3nm Latency')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Latency (ms)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 1].plot(epochs, self.training_history['power_w'], 'orange', linewidth=2)
        axes[1, 1].axhline(y=5.0, color='red', linestyle='--', label='Target (5W)')
        axes[1, 1].set_title('Power Consumption')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Power (W)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 2].scatter(self.training_history['latency_ms'], 
                          self.training_history['val_acc'], 
                          c=epochs, cmap='viridis', s=50)
        axes[1, 2].set_title('Accuracy vs Latency')
        axes[1, 2].set_xlabel('Latency (ms)')
        axes[1, 2].set_ylabel('Validation Accuracy (%)')
        axes[1, 2].grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def cleanup(self):
        if hasattr(self, 'writer'):
            self.writer.close()

if __name__ == "__main__":
    pipeline = TrainingPipeline(output_dir="results_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    try:
        results = pipeline.complete_optimization_pipeline()
        for key, value in results.items():
            if key != 'training_history':
                print(f"{key}: {value}")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        pass
    finally:
        pipeline.cleanup()