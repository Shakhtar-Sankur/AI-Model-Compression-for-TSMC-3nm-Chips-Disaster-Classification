import onnxruntime as ort
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from collections import deque

class TSMCAcceleratorSimulator:
    def __init__(self):
        self.specs = {
            'peak_tops': 50,
            'memory_bandwidth': 1024,
            'base_power': 2.5,
            'efficiency': 10,
            'clock_speed': 2000,
            'cache_size': 32,
        }
        self.performance_counters = {
            'total_inferences': 0,
            'total_latency': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'power_consumed': 0,
        }
    
    def simulate_inference(self, model_complexity: float, batch_size: int = 1) -> Dict[str, float]:
        ops_per_inference = model_complexity * batch_size
        theoretical_latency = (ops_per_inference / (self.specs['peak_tops'] * 1e12)) * 1000
        cache_efficiency = min(0.95, self.performance_counters['cache_hits'] / 
                              max(1, self.performance_counters['cache_hits'] + self.performance_counters['cache_misses']))
        actual_latency = theoretical_latency * (1.2 - 0.2 * cache_efficiency)
        actual_latency += np.random.normal(0, 0.1)
        actual_latency = max(0.5, actual_latency)
        dynamic_power = (ops_per_inference / 1e12) * (self.specs['efficiency'] ** -1)
        total_power = self.specs['base_power'] + dynamic_power
        self.performance_counters['total_inferences'] += 1
        self.performance_counters['total_latency'] += actual_latency
        if np.random.random() < 0.8:
            self.performance_counters['cache_hits'] += 1
        else:
            self.performance_counters['cache_misses'] += 1
        self.performance_counters['power_consumed'] += total_power * (actual_latency / 1000)
        return {
            'latency_ms': actual_latency,
            'power_w': total_power,
            'throughput_fps': 1000 / actual_latency,
            'energy_mj': total_power * (actual_latency / 1000) * 1000,
            'cache_hit_rate': cache_efficiency
        }

class ONNXInferenceEngine:
    def __init__(self, model_path: str, device: str = 'cpu', optimization_level: str = 'all'):
        self.model_path = Path(model_path)
        self.device = device
        self.classes = ['fire', 'flood', 'earthquake', 'hurricane']
        self.class_colors = {
            'fire': (255, 69, 0),
            'flood': (0, 191, 255),
            'earthquake': (139, 69, 19),
            'hurricane': (128, 128, 128)
        }
        self.inference_times = deque(maxlen=1000)
        self.batch_times = deque(maxlen=100)
        self.power_consumption = deque(maxlen=1000)
        self.tsmc_simulator = TSMCAcceleratorSimulator()
        self._setup_onnx_runtime(optimization_level)
        self._setup_preprocessing()
    
    def _setup_onnx_runtime(self, optimization_level: str):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel, 
            optimization_level.upper(), 
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), 
                sess_options=sess_options,
                providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def _setup_preprocessing(self):
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.preprocess_viz = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be path, numpy array, or PIL Image")
        tensor = self.preprocess(image)
        return tensor.unsqueeze(0).numpy()
    
    def predict_single(self, image: Union[str, np.ndarray, Image.Image], 
                      return_probabilities: bool = False) -> Dict:
        input_data = self.preprocess_image(image)
        model_complexity = np.prod(self.input_shape) * 50e6
        tsmc_metrics = self.tsmc_simulator.simulate_inference(model_complexity)
        start_time = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: input_data})[0]
        end_time = time.perf_counter()
        actual_latency = (end_time - start_time) * 1000
        probabilities = self._softmax(outputs[0])
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        self.inference_times.append(actual_latency)
        self.power_consumption.append(tsmc_metrics['power_w'])
        result = {
            'class': self.classes[predicted_class],
            'class_index': predicted_class,
            'confidence': float(confidence),
            'actual_latency_ms': actual_latency,
            'tsmc_metrics': tsmc_metrics,
        }
        if return_probabilities:
            result['probabilities'] = {
                self.classes[i]: float(probabilities[i]) 
                for i in range(len(self.classes))
            }
        return result
    
    def predict_batch(self, images: List[Union[str, np.ndarray, Image.Image]], 
                     batch_size: int = 8) -> List[Dict]:
        results = []
        total_images = len(images)
        for i in range(0, total_images, batch_size):
            batch = images[i:i + batch_size]
            current_batch_size = len(batch)
            batch_data = np.concatenate([
                self.preprocess_image(img) for img in batch
            ], axis=0)
            model_complexity = np.prod(self.input_shape) * 50e6
            tsmc_metrics = self.tsmc_simulator.simulate_inference(
                model_complexity, current_batch_size
            )
            start_time = time.perf_counter()
            batch_outputs = self.session.run(
                [self.output_name], 
                {self.input_name: batch_data}
            )[0]
            end_time = time.perf_counter()
            batch_latency = (end_time - start_time) * 1000
            avg_latency_per_image = batch_latency / current_batch_size
            for j in range(current_batch_size):
                probabilities = self._softmax(batch_outputs[j])
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]
                results.append({
                    'class': self.classes[predicted_class],
                    'class_index': predicted_class,
                    'confidence': float(confidence),
                    'batch_latency_ms': batch_latency,
                    'avg_latency_ms': avg_latency_per_image,
                    'tsmc_metrics': {
                        **tsmc_metrics,
                        'latency_ms': avg_latency_per_image
                    }
                })
            self.batch_times.append(batch_latency)
        return results
    
    def predict_streaming(self, image_queue: queue.Queue, result_queue: queue.Queue,
                         max_workers: int = 2, stop_event: Optional[threading.Event] = None):
        def worker():
            while True:
                if stop_event and stop_event.is_set():
                    break
                try:
                    image_data = image_queue.get(timeout=1.0)
                    if image_data is None:
                        break
                    result = self.predict_single(image_data['image'])
                    result['metadata'] = image_data.get('metadata', {})
                    result['timestamp'] = time.time()
                    result_queue.put(result)
                    image_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    error_result = {
                        'error': str(e),
                        'timestamp': time.time(),
                        'metadata': image_data.get('metadata', {}) if 'image_data' in locals() else {}
                    }
                    result_queue.put(error_result)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker) for _ in range(max_workers)]
            for future in futures:
                future.result()
    
    def visualize_prediction(self, image: Union[str, np.ndarray, Image.Image], 
                           prediction_result: Optional[Dict] = None, 
                           save_path: Optional[str] = None) -> plt.Figure:
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
            title_suffix = Path(image).name
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            title_suffix = "Input Image"
        else:
            img = image
            title_suffix = "Input Image"
        if prediction_result is None:
            prediction_result = self.predict_single(image, return_probabilities=True)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Disaster Classification Results - {title_suffix}', fontsize=16, fontweight='bold')
        ax1.imshow(img)
        ax1.set_title(f'Input Image\nPredicted: {prediction_result["class"].title()}')
        ax1.axis('off')
        class_color = self.class_colors[prediction_result['class']]
        rect = plt.Rectangle((10, 10), 100, 30, linewidth=3, 
                           edgecolor=[c/255 for c in class_color], facecolor='none')
        ax1.add_patch(rect)
        ax1.text(15, 25, f'{prediction_result["class"].title()}\n{prediction_result["confidence"]:.1%}', 
                fontsize=12, fontweight='bold', color='white',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=[c/255 for c in class_color], alpha=0.8))
        if 'probabilities' in prediction_result:
            classes = list(prediction_result['probabilities'].keys())
            probs = list(prediction_result['probabilities'].values())
            colors = [self.class_colors[cls] for cls in classes]
            colors = [[c/255 for c in color] for color in colors]
            bars = ax2.barh(classes, probs, color=colors, alpha=0.7)
            ax2.set_xlabel('Confidence Score')
            ax2.set_title('Class Probabilities')
            ax2.set_xlim(0, 1)
            for bar, prob in zip(bars, probs):
                width = bar.get_width()
                ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.1%}', ha='left', va='center', fontweight='bold')
        tsmc_metrics = prediction_result.get('tsmc_metrics', {})
        perf_data = {
            'Latency (ms)': prediction_result.get('actual_latency_ms', 0),
            'TSMC Latency (ms)': tsmc_metrics.get('latency_ms', 0),
            'Power (W)': tsmc_metrics.get('power_w', 0),
            'Energy (mJ)': tsmc_metrics.get('energy_mj', 0),
            'Throughput (FPS)': tsmc_metrics.get('throughput_fps', 0),
            'Cache Hit Rate': tsmc_metrics.get('cache_hit_rate', 0)
        }
        metrics = list(perf_data.keys())
        values = list(perf_data.values())
        ax3.barh(metrics, values, color='steelblue', alpha=0.7)
        ax3.set_title('Performance Metrics')
        ax3.set_xlabel('Value')
        for i, (metric, value) in enumerate(perf_data.items()):
            if 'Rate' in metric:
                label = f'{value:.1%}'
            elif 'FPS' in metric:
                label = f'{value:.1f}'
            else:
                label = f'{value:.2f}'
            ax3.text(value + max(values) * 0.02, i, label, 
                    ha='left', va='center', fontweight='bold')
        sys_info = [
            f"Model: {self.model_path.name}",
            f"Device: {self.device.upper()}",
            f"Input Shape: {self.input_shape}",
            f"Providers: {', '.join(self.session.get_providers())}",
            f"Total Inferences: {self.tsmc_simulator.performance_counters['total_inferences']}",
            f"Avg Latency: {np.mean(list(self.inference_times)):.2f}ms" if self.inference_times else "Avg Latency: N/A",
            f"Avg Power: {np.mean(list(self.power_consumption)):.2f}W" if self.power_consumption else "Avg Power: N/A"
        ]
        ax4.text(0.05, 0.95, '\n'.join(sys_info), transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        ax4.set_title('System Information')
        ax4.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def benchmark(self, test_images: List[Union[str, np.ndarray, Image.Image]], 
                 batch_sizes: List[int] = [1, 4, 8, 16], 
                 num_runs: int = 5) -> Dict:
        benchmark_results = {
            'single_inference': {},
            'batch_inference': {},
            'system_metrics': {},
            'timestamps': {
                'start': time.time(),
                'end': None
            }
        }
        single_times = []
        single_powers = []
        for run in range(num_runs):
            run_times = []
            run_powers = []
            for img in test_images[:10]:
                result = self.predict_single(img)
                run_times.append(result['actual_latency_ms'])
                run_powers.append(result['tsmc_metrics']['power_w'])
            single_times.extend(run_times)
            single_powers.extend(run_powers)
        benchmark_results['single_inference'] = {
            'avg_latency_ms': np.mean(single_times),
            'std_latency_ms': np.std(single_times),
            'min_latency_ms': np.min(single_times),
            'max_latency_ms': np.max(single_times),
            'p95_latency_ms': np.percentile(single_times, 95),
            'p99_latency_ms': np.percentile(single_times, 99),
            'avg_power_w': np.mean(single_powers),
            'avg_throughput_fps': 1000 / np.mean(single_times)
        }
        batch_results = {}
        for batch_size in batch_sizes:
            if len(test_images) < batch_size:
                continue
            batch_times = []
            for run in range(num_runs):
                batch_imgs = test_images[:batch_size]
                start_time = time.perf_counter()
                self.predict_batch(batch_imgs, batch_size=batch_size)
                end_time = time.perf_counter()
                total_time = (end_time - start_time) * 1000
                batch_times.append(total_time)
            avg_batch_time = np.mean(batch_times)
            avg_per_image = avg_batch_time / batch_size
            throughput = (batch_size * 1000) / avg_batch_time
            batch_results[batch_size] = {
                'avg_batch_time_ms': avg_batch_time,
                'std_batch_time_ms': np.std(batch_times),
                'avg_per_image_ms': avg_per_image,
                'throughput_fps': throughput,
                'efficiency_gain': benchmark_results['single_inference']['avg_latency_ms'] / avg_per_image
            }
        benchmark_results['batch_inference'] = batch_results
        benchmark_results['system_metrics'] = {
            'tsmc_simulator': self.tsmc_simulator.performance_counters.copy(),
            'cache_hit_rate': (self.tsmc_simulator.performance_counters['cache_hits'] / 
                              max(1, self.tsmc_simulator.performance_counters['cache_hits'] + 
                                  self.tsmc_simulator.performance_counters['cache_misses'])),
            'avg_power_efficiency': (benchmark_results['single_inference']['avg_throughput_fps'] / 
                                   benchmark_results['single_inference']['avg_power_w']),
            'model_info': {
                'input_shape': self.input_shape,
                'providers': self.session.get_providers(),
                'device': self.device
            }
        }
        benchmark_results['timestamps']['end'] = time.time()
        benchmark_results['total_duration_s'] = (benchmark_results['timestamps']['end'] - 
                                                benchmark_results['timestamps']['start'])
        return benchmark_results
    
    def _print_benchmark_summary(self, results: Dict):
        single = results['single_inference']
        sys_metrics = results['system_metrics']
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        print(f"\nSingle Inference Performance:")
        print(f"   Average Latency: {single['avg_latency_ms']:.2f} ± {single['std_latency_ms']:.2f} ms")
        print(f"   P95 Latency: {single['p95_latency_ms']:.2f} ms")
        print(f"   P99 Latency: {single['p99_latency_ms']:.2f} ms")
        print(f"   Average Power: {single['avg_power_w']:.2f} W")
        print(f"   Throughput: {single['avg_throughput_fps']:.1f} FPS")
        print(f"\nBatch Performance:")
        for batch_size, metrics in results['batch_inference'].items():
            print(f"   Batch Size {batch_size:2d}: {metrics['avg_per_image_ms']:.2f} ms/image, "
                  f"{metrics['throughput_fps']:.1f} FPS, "
                  f"{metrics['efficiency_gain']:.1f}x efficiency gain")
        print(f"\nSystem Performance:")
        print(f"   Cache Hit Rate: {sys_metrics['cache_hit_rate']:.1%}")
        print(f"   Power Efficiency: {sys_metrics['avg_power_efficiency']:.1f} FPS/W")
        print(f"   Total Inferences: {sys_metrics['tsmc_simulator']['total_inferences']}")
        print(f"   Total Energy: {sys_metrics['tsmc_simulator']['power_consumed']:.2f} J")
        print(f"\nTotal Benchmark Duration: {results['total_duration_s']:.1f} seconds")
        print("="*80)
    
    def save_benchmark_report(self, benchmark_results: Dict, output_path: str):
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': str(self.model_path),
            'device': self.device,
            'results': benchmark_results
        }
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def get_performance_stats(self) -> Dict:
        if not self.inference_times:
            return {"error": "No inference data available"}
        return {
            'inference_count': len(self.inference_times),
            'avg_latency_ms': np.mean(list(self.inference_times)),
            'std_latency_ms': np.std(list(self.inference_times)),
            'min_latency_ms': np.min(list(self.inference_times)),
            'max_latency_ms': np.max(list(self.inference_times)),
            'p95_latency_ms': np.percentile(list(self.inference_times), 95),
            'avg_power_w': np.mean(list(self.power_consumption)) if self.power_consumption else 0,
            'tsmc_metrics': self.tsmc_simulator.performance_counters.copy()
        }
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.inference_times.clear()
        self.batch_times.clear()
        self.power_consumption.clear()

def main():
    model_path = "disaster_classifier.onnx"
    try:
        with ONNXInferenceEngine(model_path, device='cpu') as engine:
            test_images = [
                "test_fire.jpg",
                "test_flood.jpg", 
                "test_earthquake.jpg",
                "test_hurricane.jpg"
            ]
            if test_images:
                result = engine.predict_single(test_images[0], return_probabilities=True)
                fig = engine.visualize_prediction(test_images[0], result)
                plt.show()
            batch_results = engine.predict_batch(test_images[:2], batch_size=2)
            benchmark_results = engine.benchmark(test_images, num_runs=3)
            engine.save_benchmark_report(benchmark_results, "benchmark_results.json")
            stats = engine.get_performance_stats()
    except FileNotFoundError:
        pass
    except Exception as e:
        pass

if __name__ == "__main__":
    main()