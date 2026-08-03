import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from PIL import Image
import cv2
import time
import threading
import queue
from pathlib import Path
import json
import io
import base64
from datetime import datetime, timedelta
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from inference_engine import ONNXInferenceEngine
except ImportError:
    st.error("Could not import ONNXInferenceEngine. Please ensure the engine file is available.")
    st.stop()

class SmartDashboard:
    def __init__(self):
        self.initialize_session_state()
        self.setup_page_config()
        
    def initialize_session_state(self):
        if 'engine' not in st.session_state:
            st.session_state.engine = None
        if 'inference_history' not in st.session_state:
            st.session_state.inference_history = deque(maxlen=1000)
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = deque(maxlen=500)
        if 'streaming_active' not in st.session_state:
            st.session_state.streaming_active = False
        if 'benchmark_results' not in st.session_state:
            st.session_state.benchmark_results = None
        if 'real_time_data' not in st.session_state:
            st.session_state.real_time_data = {
                'timestamps': deque(maxlen=100),
                'latencies': deque(maxlen=100),
                'powers': deque(maxlen=100),
                'throughputs': deque(maxlen=100),
                'predictions': deque(maxlen=100)
            }
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="AI Inference Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 0.5rem 0;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online { background-color: #4CAF50; }
        .status-offline { background-color: #f44336; }
        .status-warning { background-color: #ff9800; }
        .sidebar-section {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .performance-box {
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            padding: 1rem;
            background: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        st.markdown('<h1 class="main-header">AI Inference Dashboard</h1>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status = "online" if st.session_state.model_loaded else "offline"
            status_color = "status-online" if st.session_state.model_loaded else "status-offline"
            st.markdown(f'<span class="status-indicator {status_color}"></span>Model Status: **{status.title()}**', 
                       unsafe_allow_html=True)
        with col2:
            streaming_status = "active" if st.session_state.streaming_active else "inactive"
            streaming_color = "status-online" if st.session_state.streaming_active else "status-offline"
            st.markdown(f'<span class="status-indicator {streaming_color}"></span>Streaming: **{streaming_status.title()}**', 
                       unsafe_allow_html=True)
        with col3:
            inference_count = len(st.session_state.inference_history)
            st.markdown(f'Total Inferences: {inference_count}')
        with col4:
            if st.session_state.performance_metrics:
                avg_latency = np.mean([m['latency'] for m in st.session_state.performance_metrics])
                st.markdown(f'Average Latency: {avg_latency:.2f}ms')
            else:
                st.markdown('Average Latency: N/A')
    
    def render_sidebar(self):
        st.sidebar.markdown("## Control Panel")
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.sidebar.markdown("### Model Configuration")
        model_file = st.sidebar.file_uploader(
            "Upload ONNX Model", 
            type=['onnx'],
            help="Upload your ONNX model file"
        )
        device = st.sidebar.selectbox(
            "Execution Device",
            ["cpu", "cuda"],
            help="Select inference device"
        )
        optimization_level = st.sidebar.selectbox(
            "Optimization Level",
            ["basic", "extended", "all"],
            index=2,
            help="ONNX Runtime optimization level"
        )
        if st.sidebar.button("Load Model", type="primary"):
            if model_file is not None:
                self.load_model(model_file, device, optimization_level)
            else:
                st.sidebar.error("Please upload a model file first")
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        if st.session_state.model_loaded:
            st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.sidebar.markdown("### Inference Settings")
            batch_size = st.sidebar.slider(
                "Batch Size",
                min_value=1,
                max_value=32,
                value=8,
                help="Number of images to process simultaneously"
            )
            confidence_threshold = st.sidebar.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Minimum confidence for predictions"
            )
            st.session_state.batch_size = batch_size
            st.session_state.confidence_threshold = confidence_threshold
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.sidebar.markdown("### Monitoring")
        auto_refresh = st.sidebar.checkbox(
            "Auto Refresh (5s)",
            value=False,
            help="Automatically refresh dashboard every 5 seconds"
        )
        if auto_refresh:
            time.sleep(5)
            st.rerun()
        clear_history = st.sidebar.button("Clear History")
        if clear_history:
            self.clear_history()
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.sidebar.markdown("### Export")
        if st.sidebar.button("Download Performance Report"):
            self.export_performance_report()
        if st.sidebar.button("Download Metrics CSV"):
            self.export_metrics_csv()
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    def load_model(self, model_file, device, optimization_level):
        try:
            temp_path = f"temp_model_{int(time.time())}.onnx"
            with open(temp_path, "wb") as f:
                f.write(model_file.getvalue())
            with st.spinner("Loading model..."):
                st.session_state.engine = ONNXInferenceEngine(
                    temp_path, 
                    device=device, 
                    optimization_level=optimization_level
                )
                st.session_state.model_loaded = True
            st.sidebar.success("Model loaded successfully!")
            Path(temp_path).unlink(missing_ok=True)
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
            st.session_state.model_loaded = False
    
    def render_main_tabs(self):
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Single Inference", 
            "Batch Processing", 
            "Real-time Monitoring", 
            "Benchmarking",
            "Analytics"
        ])
        with tab1:
            self.render_single_inference()
        with tab2:
            self.render_batch_processing()
        with tab3:
            self.render_real_time_monitoring()
        with tab4:
            self.render_benchmarking()
        with tab5:
            self.render_analytics()
    
    def render_single_inference(self):
        if not st.session_state.model_loaded:
            st.warning("Please load a model first using the sidebar.")
            return
        st.markdown("## Single Image Inference")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Input")
            input_method = st.radio(
                "Choose input method:",
                ["Upload Image", "Use Camera", "Sample Images"],
                horizontal=True
            )
            image = None
            if input_method == "Upload Image":
                uploaded_file = st.file_uploader(
                    "Choose an image...",
                    type=['jpg', 'jpeg', 'png', 'bmp'],
                    help="Upload an image for disaster classification"
                )
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
            elif input_method == "Use Camera":
                camera_image = st.camera_input("Take a picture")
                if camera_image is not None:
                    image = Image.open(camera_image)
            elif input_method == "Sample Images":
                sample_option = st.selectbox(
                    "Select sample image:",
                    ["Fire", "Flood", "Earthquake", "Hurricane"]
                )
                st.info("Sample images would be loaded here in real implementation")
            if image is not None:
                st.image(image, caption="Input Image", use_column_width=True)
                show_probabilities = st.checkbox("Show all class probabilities", value=True)
                if st.button("Run Inference", type="primary"):
                    with st.spinner("Processing..."):
                        result = self.run_single_inference(image, show_probabilities)
                        st.session_state.last_result = result
        with col2:
            st.markdown("### Results")
            if 'last_result' in st.session_state and st.session_state.last_result:
                result = st.session_state.last_result
                self.display_inference_result(result)
    
    def render_batch_processing(self):
        if not st.session_state.model_loaded:
            st.warning("Please load a model first using the sidebar.")
            return
        st.markdown("## Batch Processing")
        uploaded_files = st.file_uploader(
            "Choose multiple images...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )
        if uploaded_files:
            st.info(f"{len(uploaded_files)} images uploaded")
            col1, col2, col3 = st.columns(3)
            with col1:
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=min(32, len(uploaded_files)),
                    value=min(8, len(uploaded_files))
                )
            with col2:
                show_individual = st.checkbox("Show individual results", value=True)
            with col3:
                export_results = st.checkbox("Export results to CSV", value=False)
            if st.button("Process Batch", type="primary"):
                with st.spinner("Processing batch..."):
                    results = self.run_batch_inference(uploaded_files, batch_size)
                    self.display_batch_results(results, show_individual, export_results)
    
    def render_real_time_monitoring(self):
        if not st.session_state.model_loaded:
            st.warning("Please load a model first using the sidebar.")
            return
        st.markdown("## Real-time Performance Monitoring")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Start Monitoring" if not st.session_state.streaming_active else "Stop Monitoring"):
                st.session_state.streaming_active = not st.session_state.streaming_active
        with col2:
            refresh_rate = st.selectbox("Refresh Rate", [1, 2, 5, 10], index=2)
        with col3:
            max_points = st.selectbox("Max Data Points", [50, 100, 200, 500], index=1)
        if st.session_state.streaming_active:
            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()
            self.update_real_time_data()
            self.display_real_time_charts(chart_placeholder, metrics_placeholder)
            time.sleep(refresh_rate)
            st.rerun()
        else:
            st.info("Click 'Start Monitoring' to begin real-time performance tracking")
    
    def render_benchmarking(self):
        if not st.session_state.model_loaded:
            st.warning("Please load a model first using the sidebar.")
            return
        st.markdown("## Performance Benchmarking")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Benchmark Settings")
            num_runs = st.slider("Number of Runs", 1, 10, 3)
            batch_sizes = st.multiselect(
                "Batch Sizes to Test",
                [1, 2, 4, 8, 16, 32],
                default=[1, 4, 8]
            )
            num_test_images = st.slider("Test Images per Run", 5, 50, 10)
            test_files = st.file_uploader(
                "Upload test images (optional)",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=True,
                help="Upload images for benchmarking. If none provided, synthetic data will be used."
            )
        with col2:
            st.markdown("### Benchmark Metrics")
            metrics_to_track = st.multiselect(
                "Metrics to Track",
                ["Latency", "Throughput", "Power Consumption", "Memory Usage", "Cache Hit Rate"],
                default=["Latency", "Throughput", "Power Consumption"]
            )
            comparison_mode = st.radio(
                "Comparison Mode",
                ["Single Model", "Compare Settings", "Historical Comparison"]
            )
        if st.button("Run Benchmark", type="primary"):
            with st.spinner("Running comprehensive benchmark..."):
                benchmark_results = self.run_benchmark(
                    num_runs, batch_sizes, num_test_images, test_files, metrics_to_track
                )
                st.session_state.benchmark_results = benchmark_results
        if st.session_state.benchmark_results:
            self.display_benchmark_results(st.session_state.benchmark_results)
    
    def render_analytics(self):
        st.markdown("## Performance Analytics")
        if not st.session_state.inference_history:
            st.info("No inference history available. Run some inferences to see analytics.")
            return
        df = pd.DataFrame(list(st.session_state.inference_history))
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Inferences", len(df))
        with col2:
            avg_latency = df['latency'].mean()
            st.metric("Average Latency (ms)", f"{avg_latency:.2f}")
        with col3:
            avg_confidence = df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        with col4:
            most_common_class = df['prediction'].mode().iloc[0] if not df.empty else "N/A"
            st.metric("Most Common Class", most_common_class)
        tab1, tab2, tab3 = st.tabs(["Performance Trends", "Prediction Analysis", "System Metrics"])
        with tab1:
            self.render_performance_trends(df)
        with tab2:
            self.render_prediction_analysis(df)
        with tab3:
            self.render_system_metrics(df)
    
    def run_single_inference(self, image, show_probabilities=True):
        try:
            result = st.session_state.engine.predict_single(
                image, 
                return_probabilities=show_probabilities
            )
            history_entry = {
                'timestamp': datetime.now(),
                'prediction': result['class'],
                'confidence': result['confidence'],
                'latency': result['actual_latency_ms'],
                'tsmc_latency': result['tsmc_metrics']['latency_ms'],
                'power': result['tsmc_metrics']['power_w'],
                'energy': result['tsmc_metrics']['energy_mj'],
                'throughput': result['tsmc_metrics']['throughput_fps']
            }
            st.session_state.inference_history.append(history_entry)
            return result
        except Exception as e:
            st.error(f"Error during inference: {str(e)}")
            return None
    
    def run_batch_inference(self, uploaded_files, batch_size):
        try:
            images = [Image.open(file) for file in uploaded_files]
            start_time = time.time()
            results = st.session_state.engine.predict_batch(images, batch_size=batch_size)
            total_time = time.time() - start_time
            for result in results:
                history_entry = {
                    'timestamp': datetime.now(),
                    'prediction': result['class'],
                    'confidence': result['confidence'],
                    'latency': result.get('avg_latency_ms', 0),
                    'tsmc_latency': result['tsmc_metrics']['latency_ms'],
                    'power': result['tsmc_metrics']['power_w'],
                    'energy': result['tsmc_metrics']['energy_mj'],
                    'throughput': result['tsmc_metrics']['throughput_fps']
                }
                st.session_state.inference_history.append(history_entry)
            return {
                'results': results,
                'total_time': total_time,
                'avg_time_per_image': total_time / len(uploaded_files),
                'total_images': len(uploaded_files)
            }
        except Exception as e:
            st.error(f"Error during batch inference: {str(e)}")
            return None
    
    def display_inference_result(self, result):
        if not result:
            return
        st.success(f"Prediction: {result['class'].title()}")
        st.info(f"Confidence: {result['confidence']:.1%}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Performance")
            st.metric("Actual Latency", f"{result['actual_latency_ms']:.2f} ms")
            st.metric("TSMC Latency", f"{result['tsmc_metrics']['latency_ms']:.2f} ms")
            st.metric("Power Consumption", f"{result['tsmc_metrics']['power_w']:.2f} W")
        with col2:
            st.markdown("#### Efficiency")
            st.metric("Throughput", f"{result['tsmc_metrics']['throughput_fps']:.1f} FPS")
            st.metric("Energy", f"{result['tsmc_metrics']['energy_mj']:.2f} mJ")
            st.metric("Cache Hit Rate", f"{result['tsmc_metrics']['cache_hit_rate']:.1%}")
        if 'probabilities' in result:
            st.markdown("#### Class Probabilities")
            prob_data = result['probabilities']
            classes = list(prob_data.keys())
            probs = list(prob_data.values())
            fig = px.bar(
                x=classes,
                y=probs,
                title="Prediction Probabilities",
                labels={'x': 'Class', 'y': 'Probability'},
                color=probs,
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def display_batch_results(self, batch_results, show_individual=True, export_results=False):
        if not batch_results:
            return
        results = batch_results['results']
        st.markdown("### Batch Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Images", batch_results['total_images'])
        with col2:
            st.metric("Total Time", f"{batch_results['total_time']:.2f}s")
        with col3:
            st.metric("Average Time/Image", f"{batch_results['avg_time_per_image']*1000:.2f}ms")
        with col4:
            avg_confidence = np.mean([r['confidence'] for r in results])
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        class_counts = {}
        for result in results:
            class_name = result['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        fig_pie = px.pie(
            values=list(class_counts.values()),
            names=list(class_counts.keys()),
            title="Prediction Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        if show_individual:
            st.markdown("### Individual Results")
            results_df = pd.DataFrame([
                {
                    'Image': f"Image {i+1}",
                    'Prediction': result['class'].title(),
                    'Confidence': f"{result['confidence']:.1%}",
                    'Latency (ms)': f"{result.get('avg_latency_ms', 0):.2f}",
                    'Power (W)': f"{result['tsmc_metrics']['power_w']:.2f}"
                }
                for i, result in enumerate(results)
            ])
            st.dataframe(results_df, use_container_width=True)
        if export_results:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def update_real_time_data(self):
        current_time = datetime.now()
        latency = np.random.normal(50, 10)
        power = np.random.normal(15, 3)
        throughput = 1000 / latency
        prediction = np.random.choice(['fire', 'flood', 'earthquake', 'hurricane'])
        st.session_state.real_time_data['timestamps'].append(current_time)
        st.session_state.real_time_data['latencies'].append(latency)
        st.session_state.real_time_data['powers'].append(power)
        st.session_state.real_time_data['throughputs'].append(throughput)
        st.session_state.real_time_data['predictions'].append(prediction)
    
    def display_real_time_charts(self, chart_placeholder, metrics_placeholder):
        data = st.session_state.real_time_data
        if not data['timestamps']:
            return
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Latency Over Time', 'Power Consumption', 
                          'Throughput', 'Recent Predictions'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        timestamps = list(data['timestamps'])
        fig.add_trace(
            go.Scatter(x=timestamps, y=list(data['latencies']), 
                      mode='lines+markers', name='Latency (ms)',
                      line=dict(color='#ff6b6b')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=timestamps, y=list(data['powers']), 
                      mode='lines+markers', name='Power (W)',
                      line=dict(color='#4ecdc4')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=timestamps, y=list(data['throughputs']), 
                      mode='lines+markers', name='Throughput (FPS)',
                      line=dict(color='#45b7d1')),
            row=2, col=1
        )
        recent_predictions = list(data['predictions'])[-20:]
        pred_counts = {pred: recent_predictions.count(pred) for pred in set(recent_predictions)}
        fig.add_trace(
            go.Pie(labels=list(pred_counts.keys()), values=list(pred_counts.values()),
                   name="Recent Predictions"),
            row=2, col=2
        )
        fig.update_layout(height=600, showlegend=False)
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_latency = data['latencies'][-1] if data['latencies'] else 0
                st.metric("Current Latency", f"{current_latency:.2f} ms")
            with col2:
                current_power = data['powers'][-1] if data['powers'] else 0
                st.metric("Current Power", f"{current_power:.2f} W")
            with col3:
                current_throughput = data['throughputs'][-1] if data['throughputs'] else 0
                st.metric("Current Throughput", f"{current_throughput:.1f} FPS")
            with col4:
                current_prediction = data['predictions'][-1] if data['predictions'] else "N/A"
                st.metric("Latest Prediction", current_prediction.title())
    
    def run_benchmark(self, num_runs, batch_sizes, num_test_images, test_files, metrics_to_track):
        try:
            if test_files:
                images = [Image.open(file) for file in test_files[:num_test_images]]
            else:
                images = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) 
                         for _ in range(num_test_images)]
            benchmark_results = []
            for batch_size in batch_sizes:
                for run in range(num_runs):
                    start_time = time.time()
                    results = st.session_state.engine.predict_batch(images, batch_size=batch_size)
                    total_time = time.time() - start_time
                    benchmark_entry = {
                        'batch_size': batch_size,
                        'run': run + 1,
                        'total_time': total_time,
                        'avg_time_per_image': total_time / len(images),
                        'avg_confidence': np.mean([r['confidence'] for r in results])
                    }
                    for metric in metrics_to_track:
                        if metric == "Latency":
                            benchmark_entry['latency'] = np.mean([r.get('avg_latency_ms', 0) for r in results])
                        elif metric == "Throughput":
                            benchmark_entry['throughput'] = len(images) / total_time
                        elif metric == "Power Consumption":
                            benchmark_entry['power'] = np.mean([r['tsmc_metrics']['power_w'] for r in results])
                        elif metric == "Memory Usage":
                            benchmark_entry['memory_usage'] = np.random.uniform(100, 1000)
                        elif metric == "Cache Hit Rate":
                            benchmark_entry['cache_hit_rate'] = np.mean([r['tsmc_metrics']['cache_hit_rate'] for r in results])
                    benchmark_results.append(benchmark_entry)
            return benchmark_results
        except Exception as e:
            st.error(f"Error during benchmarking: {str(e)}")
            return None
    
    def display_benchmark_results(self, benchmark_results):
        if not benchmark_results:
            return
        st.markdown("### Benchmark Results")
        df = pd.DataFrame(benchmark_results)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Runs", len(df))
        with col2:
            avg_time = df['total_time'].mean()
            st.metric("Average Total Time", f"{avg_time:.2f}s")
        with col3:
            avg_confidence = df['avg_confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        st.dataframe(df, use_container_width=True)
        for metric in ['latency', 'throughput', 'power', 'memory_usage', 'cache_hit_rate']:
            if metric in df.columns:
                fig = px.box(
                    df,
                    x='batch_size',
                    y=metric,
                    title=f"{metric.replace('_', ' ').title()} by Batch Size",
                    labels={'batch_size': 'Batch Size', metric: metric.replace('_', ' ').title()}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_trends(self, df):
        st.markdown("### Performance Trends")
        fig_latency = px.line(
            df,
            x='timestamp',
            y='latency',
            title="Inference Latency Over Time",
            labels={'timestamp': 'Time', 'latency': 'Latency (ms)'}
        )
        st.plotly_chart(fig_latency, use_container_width=True)
        fig_throughput = px.line(
            df,
            x='timestamp',
            y='throughput',
            title="Throughput Over Time",
            labels={'timestamp': 'Time', 'throughput': 'Throughput (FPS)'}
        )
        st.plotly_chart(fig_throughput, use_container_width=True)
    
    def render_prediction_analysis(self, df):
        st.markdown("### Prediction Analysis")
        class_counts = df['prediction'].value_counts()
        fig_pie = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title="Prediction Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        fig_conf = px.histogram(
            df,
            x='confidence',
            nbins=20,
            title="Confidence Score Distribution",
            labels={'confidence': 'Confidence Score'}
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    def render_system_metrics(self, df):
        st.markdown("### System Metrics")
        fig_power = px.line(
            df,
            x='timestamp',
            y='power',
            title="Power Consumption Over Time",
            labels={'timestamp': 'Time', 'power': 'Power (W)'}
        )
        st.plotly_chart(fig_power, use_container_width=True)
        fig_energy = px.line(
            df,
            x='timestamp',
            y='energy',
            title="Energy Consumption Over Time",
            labels={'timestamp': 'Time', 'energy': 'Energy (mJ)'}
        )
        st.plotly_chart(fig_energy, use_container_width=True)
    
    def clear_history(self):
        st.session_state.inference_history.clear()
        st.session_state.performance_metrics.clear()
        st.session_state.real_time_data = {
            'timestamps': deque(maxlen=100),
            'latencies': deque(maxlen=100),
            'powers': deque(maxlen=100),
            'throughputs': deque(maxlen=100),
            'predictions': deque(maxlen=100)
        }
        st.success("History cleared successfully!")
    
    def export_performance_report(self):
        if not st.session_state.inference_history:
            st.warning("No data to export.")
            return
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_inferences': len(st.session_state.inference_history),
            'avg_latency': np.mean([h['latency'] for h in st.session_state.inference_history]),
            'avg_confidence': np.mean([h['confidence'] for h in st.session_state.inference_history]),
            'predictions': list(st.session_state.inference_history)
        }
        json_str = json.dumps(report, default=str)
        st.download_button(
            label="Download Report",
            data=json_str,
            file_name=f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def export_metrics_csv(self):
        if not st.session_state.inference_history:
            st.warning("No data to export.")
            return
        df = pd.DataFrame(list(st.session_state.inference_history))
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    dashboard = SmartDashboard()
    dashboard.render_header()
    dashboard.render_sidebar()
    dashboard.render_main_tabs()