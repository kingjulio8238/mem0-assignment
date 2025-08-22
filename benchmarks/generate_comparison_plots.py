#!/usr/bin/env python3
"""
Generate Comparative Visualization Plots
Creates charts and dashboard for model comparison analysis
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class ComparisonVisualizer:
    def __init__(self, comparison_dir: str, output_dir: str):
        self.comparison_dir = Path(comparison_dir)
        self.output_dir = Path(output_dir) / "visualizations"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load comparison data
        self.comprehensive_data = self._load_json("comprehensive_model_comparison.json")
        self.inference_data = self._load_json("inference_comparison.json")
        self.memory_data = self._load_json("memory_comparison.json")
        
        # Color scheme for models
        self.colors = {
            "base_model": "#FF6B6B",      # Red
            "base_bf16": "#9B59B6",      # Purple
            "finetuned_bf16": "#4ECDC4",  # Teal  
            "finetuned_q4km": "#45B7D1"  # Blue
        }
        
        self.model_labels = {
            "base_model": "Base Model\n(4-bit)",
            "base_bf16": "Base Model\n(bf16)",
            "finetuned_bf16": "Fine-tuned\nBF16",
            "finetuned_q4km": "Fine-tuned\nQ4_K_M"
        }
    
    def _load_json(self, filename: str):
        """Load JSON file from comparison directory"""
        file_path = self.comparison_dir / filename
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def create_inference_performance_charts(self):
        """Create inference performance comparison charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🚀 Inference Performance Comparison', fontsize=16, fontweight='bold')
        
        models = list(self.inference_data["models"].keys())
        model_names = [self.model_labels[m] for m in models]
        
        # Extract metrics
        latencies = [self.inference_data["models"][m]["metrics"]["average_latency"] for m in models]
        throughputs = [self.inference_data["models"][m]["metrics"]["average_throughput"] for m in models]
        p95_latencies = [self.inference_data["models"][m]["metrics"]["p95_latency"] for m in models]
        tokens_generated = [self.inference_data["models"][m]["metrics"]["average_tokens_generated"] for m in models]
        
        # 1. Average Latency
        bars1 = ax1.bar(model_names, latencies, color=[self.colors[m] for m in models], alpha=0.8)
        ax1.set_title('Average Latency', fontweight='bold')
        ax1.set_ylabel('Latency (seconds)')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars1, latencies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # 2. Throughput
        bars2 = ax2.bar(model_names, throughputs, color=[self.colors[m] for m in models], alpha=0.8)
        ax2.set_title('Average Throughput', fontweight='bold')
        ax2.set_ylabel('Tokens per Second')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars2, throughputs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. P95 Latency
        bars3 = ax3.bar(model_names, p95_latencies, color=[self.colors[m] for m in models], alpha=0.8)
        ax3.set_title('P95 Latency', fontweight='bold')
        ax3.set_ylabel('P95 Latency (seconds)')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars3, p95_latencies):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # 4. Average Tokens Generated
        bars4 = ax4.bar(model_names, tokens_generated, color=[self.colors[m] for m in models], alpha=0.8)
        ax4.set_title('Average Tokens Generated', fontweight='bold')
        ax4.set_ylabel('Tokens per Response')
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars4, tokens_generated):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / "inference_performance_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Saved inference performance chart: {output_file}")
        plt.close()
    
    def create_memory_performance_charts(self):
        """Create memory retrieval performance comparison charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🧠 Memory Retrieval Performance Comparison', fontsize=16, fontweight='bold')
        
        models = list(self.memory_data["models"].keys())
        model_names = [self.model_labels[m] for m in models]
        
        # Extract metrics
        precisions = [self.memory_data["models"][m]["metrics"]["average_precision_at_5"] for m in models]
        retrieval_times = [self.memory_data["models"][m]["metrics"]["average_retrieval_time"] for m in models]
        perfect_queries = [self.memory_data["models"][m]["metrics"]["perfect_precision_queries"] for m in models]
        zero_queries = [self.memory_data["models"][m]["metrics"]["zero_precision_queries"] for m in models]
        
        # 1. Precision@5
        bars1 = ax1.bar(model_names, precisions, color=[self.colors[m] for m in models], alpha=0.8)
        ax1.set_title('Average Precision@5', fontweight='bold')
        ax1.set_ylabel('Precision@5 Score')
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Target (0.6)')
        ax1.legend()
        
        for bar, val in zip(bars1, precisions):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Retrieval Time
        bars2 = ax2.bar(model_names, retrieval_times, color=[self.colors[m] for m in models], alpha=0.8)
        ax2.set_title('Average Retrieval Time', fontweight='bold')
        ax2.set_ylabel('Time (seconds)')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars2, retrieval_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        # 3. Perfect vs Zero Precision Queries
        x = np.arange(len(model_names))
        width = 0.35
        
        bars3a = ax3.bar(x - width/2, perfect_queries, width, label='Perfect (1.0)', 
                        color='green', alpha=0.7)
        bars3b = ax3.bar(x + width/2, zero_queries, width, label='Zero (0.0)', 
                        color='red', alpha=0.7)
        
        ax3.set_title('Query Performance Distribution', fontweight='bold')
        ax3.set_ylabel('Number of Queries')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars3a, bars3b]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Precision Improvement vs Base
        base_precision = precisions[0]  # Assuming first model is base
        improvements = [(p - base_precision) * 100 for p in precisions]
        
        bars4 = ax4.bar(model_names, improvements, color=[self.colors[m] for m in models], alpha=0.8)
        ax4.set_title('Precision Improvement vs Base Model', fontweight='bold')
        ax4.set_ylabel('Improvement (%)')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars4, improvements):
            ax4.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.2 if val >= 0 else -0.5),
                    f'{val:+.1f}%', ha='center', 
                    va='bottom' if val >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / "memory_performance_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"🧠 Saved memory performance chart: {output_file}")
        plt.close()
    
    def create_performance_trade_off_chart(self):
        """Create performance trade-off scatter plot"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        models = list(self.inference_data["models"].keys())
        
        # Extract data for scatter plot
        latencies = [self.inference_data["models"][m]["metrics"]["average_latency"] for m in models]
        precisions = [self.memory_data["models"][m]["metrics"]["average_precision_at_5"] for m in models]
        throughputs = [self.inference_data["models"][m]["metrics"]["average_throughput"] for m in models]
        
        # Create scatter plot with bubble size based on throughput
        scatter = ax.scatter(latencies, precisions, 
                           s=[t*20 for t in throughputs],  # Scale bubble size
                           c=[self.colors[m] for m in models],
                           alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add model labels
        for i, model in enumerate(models):
            ax.annotate(self.model_labels[model], 
                       (latencies[i], precisions[i]),
                       xytext=(10, 10), textcoords='offset points',
                       fontweight='bold', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor=self.colors[model], alpha=0.3))
        
        ax.set_xlabel('Average Latency (seconds)', fontweight='bold')
        ax.set_ylabel('Memory Precision@5', fontweight='bold')
        ax.set_title('🎯 Performance Trade-off Analysis\n(Bubble size = Throughput)', 
                    fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add ideal region
        ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Target Precision')
        ax.axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='Target Latency')
        
        # Add legend for bubble sizes
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='gray', markersize=8, alpha=0.7,
                      label=f'Throughput: {min(throughputs):.1f}-{max(throughputs):.1f} tok/s')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / "performance_tradeoff_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"🎯 Saved trade-off analysis chart: {output_file}")
        plt.close()
    
    def create_comprehensive_dashboard(self):
        """Create comprehensive dashboard with key metrics"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('📊 Comprehensive Model Comparison Dashboard', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        models = list(self.inference_data["models"].keys())
        model_names = [self.model_labels[m] for m in models]
        
        # 1. Latency Comparison (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        latencies = [self.inference_data["models"][m]["metrics"]["average_latency"] for m in models]
        bars1 = ax1.bar(model_names, latencies, color=[self.colors[m] for m in models], alpha=0.8)
        ax1.set_title('Inference Latency', fontweight='bold')
        ax1.set_ylabel('Seconds')
        for bar, val in zip(bars1, latencies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. Throughput Comparison (Top Center-Left)
        ax2 = fig.add_subplot(gs[0, 1])
        throughputs = [self.inference_data["models"][m]["metrics"]["average_throughput"] for m in models]
        bars2 = ax2.bar(model_names, throughputs, color=[self.colors[m] for m in models], alpha=0.8)
        ax2.set_title('Throughput', fontweight='bold')
        ax2.set_ylabel('Tokens/sec')
        for bar, val in zip(bars2, throughputs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 3. Memory Precision (Top Center-Right)
        ax3 = fig.add_subplot(gs[0, 2])
        precisions = [self.memory_data["models"][m]["metrics"]["average_precision_at_5"] for m in models]
        bars3 = ax3.bar(model_names, precisions, color=[self.colors[m] for m in models], alpha=0.8)
        ax3.set_title('Memory Precision@5', fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, 1.0)
        ax3.axhline(y=0.6, color='red', linestyle='--', alpha=0.7)
        for bar, val in zip(bars3, precisions):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 4. Success Metrics (Top Right)
        ax4 = fig.add_subplot(gs[0, 3])
        perfect_queries = [self.memory_data["models"][m]["metrics"]["perfect_precision_queries"] for m in models]
        ax4.pie(perfect_queries, labels=model_names, autopct='%1.0f', 
               colors=[self.colors[m] for m in models], startangle=90)
        ax4.set_title('Perfect Precision\nQueries Distribution', fontweight='bold')
        
        # 5. Performance Trend (Middle, spanning 2 columns)
        ax5 = fig.add_subplot(gs[1, :2])
        x_pos = np.arange(len(models))
        
        # Normalize metrics for comparison (0-1 scale)
        norm_latency = [(max(latencies) - l) / (max(latencies) - min(latencies)) for l in latencies]  # Invert (lower is better)
        norm_throughput = [(t - min(throughputs)) / (max(throughputs) - min(throughputs)) for t in throughputs]
        norm_precision = [(p - min(precisions)) / (max(precisions) - min(precisions)) if max(precisions) > min(precisions) else 0.5 for p in precisions]
        
        width = 0.25
        ax5.bar(x_pos - width, norm_latency, width, label='Speed (inv. latency)', alpha=0.8, color='#FF6B6B')
        ax5.bar(x_pos, norm_throughput, width, label='Throughput', alpha=0.8, color='#4ECDC4')
        ax5.bar(x_pos + width, norm_precision, width, label='Memory Precision', alpha=0.8, color='#45B7D1')
        
        ax5.set_title('Normalized Performance Comparison', fontweight='bold')
        ax5.set_ylabel('Normalized Score (0-1)')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(model_names)
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Model Efficiency Radar (Middle Right, spanning 2 columns)
        ax6 = fig.add_subplot(gs[1, 2:], projection='polar')
        
        # Radar chart metrics
        metrics = ['Speed\n(inv. latency)', 'Throughput', 'Memory\nPrecision', 'Efficiency\n(P@5/latency)']
        num_metrics = len(metrics)
        
        # Calculate efficiency metric
        efficiency = [p/l for p, l in zip(precisions, latencies)]
        norm_efficiency = [(e - min(efficiency)) / (max(efficiency) - min(efficiency)) if max(efficiency) > min(efficiency) else 0.5 for e in efficiency]
        
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, model in enumerate(models):
            values = [norm_latency[i], norm_throughput[i], norm_precision[i], norm_efficiency[i]]
            values += values[:1]  # Complete the circle
            
            ax6.plot(angles, values, 'o-', linewidth=2, 
                    label=self.model_labels[model], color=self.colors[model])
            ax6.fill(angles, values, alpha=0.1, color=self.colors[model])
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(metrics)
        ax6.set_ylim(0, 1)
        ax6.set_title('Model Performance Radar', fontweight='bold', pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        # 7. Summary Statistics (Bottom, spanning all columns)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Create summary table
        summary_data = []
        for i, model in enumerate(models):
            summary_data.append([
                self.model_labels[model],
                f"{latencies[i]:.2f}s",
                f"{throughputs[i]:.1f} tok/s",
                f"{precisions[i]:.3f}",
                f"{self.memory_data['models'][model]['metrics']['average_retrieval_time']:.3f}s",
                "✅" if precisions[i] > 0.6 else "❌"
            ])
        
        table = ax7.table(cellText=summary_data,
                         colLabels=['Model', 'Avg Latency', 'Throughput', 'Precision@5', 'Retrieval Time', 'Target Met'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code the table
        for i in range(len(models)):
            table[(i+1, 0)].set_facecolor(self.colors[models[i]])
            table[(i+1, 0)].set_alpha(0.3)
        
        ax7.set_title('📋 Summary Statistics', fontweight='bold', fontsize=14, pad=20)
        
        # Save dashboard
        output_file = self.output_dir / "comprehensive_dashboard.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Saved comprehensive dashboard: {output_file}")
        plt.close()
    
    def create_improvement_analysis_chart(self):
        """Create improvement analysis chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📈 Fine-tuning Improvement Analysis', fontsize=16, fontweight='bold')
        
        # 1. Precision Improvement vs Base
        models = ["finetuned_bf16", "finetuned_q4km"]
        model_names = [self.model_labels[m] for m in models]
        
        base_precision = self.memory_data["models"]["base_model"]["metrics"]["average_precision_at_5"]
        improvements = []
        for model in models:
            model_precision = self.memory_data["models"][model]["metrics"]["average_precision_at_5"]
            improvement = ((model_precision - base_precision) / base_precision) * 100
            improvements.append(improvement)
        
        bars1 = ax1.bar(model_names, improvements, color=[self.colors[m] for m in models], alpha=0.8)
        ax1.set_title('Memory Precision Improvement', fontweight='bold')
        ax1.set_ylabel('Improvement (%)')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.axhline(y=25, color='green', linestyle='--', alpha=0.7, label='Good Improvement (25%)')
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend()
        
        for bar, val in zip(bars1, improvements):
            ax1.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.5 if val >= 0 else -1),
                    f'{val:+.1f}%', ha='center', 
                    va='bottom' if val >= 0 else 'top', fontweight='bold')
        
        # 2. Speed vs Quality Trade-off
        latencies = [self.inference_data["models"][m]["metrics"]["average_latency"] for m in ["base_model"] + models]
        precisions = [self.memory_data["models"][m]["metrics"]["average_precision_at_5"] for m in ["base_model"] + models]
        all_models = ["base_model"] + models
        all_names = [self.model_labels[m] for m in all_models]
        
        for i, model in enumerate(all_models):
            ax2.scatter(latencies[i], precisions[i], 
                       s=200, c=self.colors[model], alpha=0.8, edgecolors='black')
            ax2.annotate(all_names[i], (latencies[i], precisions[i]),
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax2.set_xlabel('Latency (seconds)')
        ax2.set_ylabel('Memory Precision@5')
        ax2.set_title('Speed vs Quality Trade-off', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.6, color='green', linestyle='--', alpha=0.7, label='Target Precision')
        ax2.legend()
        
        # 3. Quantization Impact
        bf16_precision = self.memory_data["models"]["finetuned_bf16"]["metrics"]["average_precision_at_5"]
        q4km_precision = self.memory_data["models"]["finetuned_q4km"]["metrics"]["average_precision_at_5"]
        
        bf16_latency = self.inference_data["models"]["finetuned_bf16"]["metrics"]["average_latency"]
        q4km_latency = self.inference_data["models"]["finetuned_q4km"]["metrics"]["average_latency"]
        
        metrics = ['Memory\nPrecision', 'Inference\nSpeed']
        bf16_values = [bf16_precision, 1/bf16_latency]  # Invert latency for speed
        q4km_values = [q4km_precision, 1/q4km_latency]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Normalize for comparison
        max_precision = max(bf16_precision, q4km_precision)
        max_speed = max(1/bf16_latency, 1/q4km_latency)
        
        bf16_norm = [bf16_precision/max_precision, (1/bf16_latency)/max_speed]
        q4km_norm = [q4km_precision/max_precision, (1/q4km_latency)/max_speed]
        
        bars3a = ax3.bar(x - width/2, bf16_norm, width, label='BF16', 
                        color=self.colors["finetuned_bf16"], alpha=0.8)
        bars3b = ax3.bar(x + width/2, q4km_norm, width, label='Q4_K_M', 
                        color=self.colors["finetuned_q4km"], alpha=0.8)
        
        ax3.set_title('Quantization Impact (Normalized)', fontweight='bold')
        ax3.set_ylabel('Normalized Performance')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Success Rate Analysis
        success_metrics = ['Perfect\nQueries', 'Zero\nQueries', 'Success\nRate']
        
        base_perfect = self.memory_data["models"]["base_model"]["metrics"]["perfect_precision_queries"]
        base_zero = self.memory_data["models"]["base_model"]["metrics"]["zero_precision_queries"]
        base_success = self.memory_data["models"]["base_model"]["metrics"]["success_rate"]
        
        bf16_perfect = self.memory_data["models"]["finetuned_bf16"]["metrics"]["perfect_precision_queries"]
        bf16_zero = self.memory_data["models"]["finetuned_bf16"]["metrics"]["zero_precision_queries"]
        bf16_success = self.memory_data["models"]["finetuned_bf16"]["metrics"]["success_rate"]
        
        q4km_perfect = self.memory_data["models"]["finetuned_q4km"]["metrics"]["perfect_precision_queries"]
        q4km_zero = self.memory_data["models"]["finetuned_q4km"]["metrics"]["zero_precision_queries"]
        q4km_success = self.memory_data["models"]["finetuned_q4km"]["metrics"]["success_rate"]
        
        x = np.arange(len(success_metrics))
        width = 0.25
        
        base_values = [base_perfect, 21-base_zero, base_success*21]  # Convert success rate back to count
        bf16_values = [bf16_perfect, 21-bf16_zero, bf16_success*21]
        q4km_values = [q4km_perfect, 21-q4km_zero, q4km_success*21]
        
        ax4.bar(x - width, base_values, width, label='Base', color=self.colors["base_model"], alpha=0.8)
        ax4.bar(x, bf16_values, width, label='BF16', color=self.colors["finetuned_bf16"], alpha=0.8)
        ax4.bar(x + width, q4km_values, width, label='Q4_K_M', color=self.colors["finetuned_q4km"], alpha=0.8)
        
        ax4.set_title('Success Metrics Comparison', fontweight='bold')
        ax4.set_ylabel('Count / Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(success_metrics)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / "improvement_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📈 Saved improvement analysis chart: {output_file}")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualization plots"""
        print("🎨 Generating visualization plots...")
        
        # Create individual charts
        self.create_inference_performance_charts()
        self.create_memory_performance_charts()
        self.create_performance_trade_off_chart()
        self.create_improvement_analysis_chart()
        
        # Create comprehensive dashboard
        self.create_comprehensive_dashboard()
        
        # Generate summary
        self._generate_visualization_summary()
        
        print(f"\n🎉 All visualizations complete! Check {self.output_dir}")
    
    def _generate_visualization_summary(self):
        """Generate summary of all visualizations"""
        summary = f"""
# 📊 Visualization Summary
Generated: {datetime.now().isoformat()}

## 📈 Generated Charts:

1. **inference_performance_comparison.png** - Detailed inference metrics comparison
2. **memory_performance_comparison.png** - Memory retrieval performance analysis  
3. **performance_tradeoff_analysis.png** - Speed vs quality trade-off scatter plot
4. **improvement_analysis.png** - Fine-tuning improvement analysis
5. **comprehensive_dashboard.png** - Complete overview dashboard

## 🎯 Key Insights from Visualizations:

### Performance Rankings:
- **Fastest Inference:** {self.comprehensive_data['inference_analysis']['statistical_analysis']['latency']['fastest_model']}
- **Best Memory:** {self.comprehensive_data['memory_analysis']['statistical_analysis']['precision']['best_model']}
- **Most Efficient:** {self.comprehensive_data['trade_off_analysis']['model_efficiency']['most_efficient_overall']}

### Fine-tuning Success:
- **Memory Improvement:** {'✅ Achieved' if self.comprehensive_data['memory_analysis']['improvement_analysis']['overall_assessment']['precision_improvement_achieved'] else '❌ Not achieved'}
- **Best Fine-tuned Model:** {self.comprehensive_data['memory_analysis']['improvement_analysis']['overall_assessment']['best_fine_tuned_model']}

### Recommendations:
- **Production Use:** {self.comprehensive_data['recommendations']['use_case_recommendations']['production_deployment']['recommended_model']}
- **Memory-Critical:** {self.comprehensive_data['recommendations']['use_case_recommendations']['memory_critical_applications']['recommended_model']}
- **Speed-Critical:** {self.comprehensive_data['recommendations']['use_case_recommendations']['speed_critical_applications']['recommended_model']}

## 📁 Files Location:
All charts saved to: {self.output_dir}
"""
        
        summary_file = self.output_dir / "visualization_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"📝 Saved visualization summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison visualization plots")
    parser.add_argument("--comparison-results", required=True, 
                       help="Path to comparison results directory")
    parser.add_argument("--output-dir", required=True, 
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    print("🎨 Starting visualization generation...")
    print(f"📂 Comparison results: {args.comparison_results}")
    print(f"📂 Output directory: {args.output_dir}")
    
    visualizer = ComparisonVisualizer(
        comparison_dir=args.comparison_results,
        output_dir=args.output_dir
    )
    
    visualizer.generate_all_visualizations()
    
    print("\n🎉 Visualization generation complete!")


if __name__ == "__main__":
    main()
