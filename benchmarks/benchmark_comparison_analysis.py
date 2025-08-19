#!/usr/bin/env python3
"""
Benchmark Comparison Analysis Script
Compares performance across Base, Fine-tuned BF16, and Fine-tuned Q4_K_M models
"""

import json
import argparse
import os
import numpy as np
from datetime import datetime
from pathlib import Path
import statistics
from typing import Dict, List, Any


class BenchmarkComparator:
    def __init__(self, base_dir: str, bf16_dir: str, q4km_dir: str, output_dir: str):
        self.base_dir = Path(base_dir)
        self.bf16_dir = Path(bf16_dir)
        self.q4km_dir = Path(q4km_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load all benchmark data
        self.data = self._load_all_benchmarks()
    
    def _load_all_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Load all benchmark JSON files"""
        data = {}
        
        # Load base model results
        base_inference = self._find_and_load_file(self.base_dir, "inference_benchmark")
        base_memory = self._find_and_load_file(self.base_dir, "memory_benchmark")
        
        # Load BF16 model results
        bf16_inference = self._find_and_load_file(self.bf16_dir, "inference_benchmark")
        bf16_memory = self._find_and_load_file(self.bf16_dir, "memory_benchmark")
        
        # Load Q4_K_M model results
        q4km_inference = self._find_and_load_file(self.q4km_dir, "inference_benchmark")
        q4km_memory = self._find_and_load_file(self.q4km_dir, "memory_benchmark")
        
        data = {
            "base_model": {
                "inference": base_inference,
                "memory": base_memory,
                "name": "Base Model (4-bit)",
                "model_path": base_inference.get("model_info", {}).get("model_path", "Unknown")
            },
            "finetuned_bf16": {
                "inference": bf16_inference,
                "memory": bf16_memory,
                "name": "Fine-tuned BF16",
                "model_path": bf16_inference.get("model_info", {}).get("model_path", "Unknown")
            },
            "finetuned_q4km": {
                "inference": q4km_inference,
                "memory": q4km_memory,
                "name": "Fine-tuned Q4_K_M",
                "model_path": q4km_inference.get("model_info", {}).get("model_path", "Unknown")
            }
        }
        
        return data
    
    def _find_and_load_file(self, directory: Path, file_prefix: str) -> Dict[str, Any]:
        """Find and load a JSON file with given prefix in directory"""
        files = list(directory.glob(f"{file_prefix}*.json"))
        if not files:
            raise FileNotFoundError(f"No {file_prefix} file found in {directory}")
        
        # Take the most recent file if multiple exist
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def compare_inference_performance(self) -> Dict[str, Any]:
        """Compare inference performance across all models"""
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "relative_performance": {},
            "statistical_analysis": {}
        }
        
        # Extract inference metrics for each model
        for model_key, model_data in self.data.items():
            inference_data = model_data["inference"]
            summary = inference_data["summary"]
            
            comparison["models"][model_key] = {
                "name": model_data["name"],
                "model_path": model_data["model_path"],
                "metrics": {
                    "average_latency": summary["average_latency"],
                    "median_latency": summary["median_latency"],
                    "p95_latency": summary["p95_latency"],
                    "average_throughput": summary["average_throughput"],
                    "median_throughput": summary["median_throughput"],
                    "average_tokens_generated": summary["average_tokens_generated"],
                    "total_prompts": summary["num_prompts"],
                    "success_rate": summary["successful_prompts"] / summary["num_prompts"]
                }
            }
        
        # Calculate relative performance (base model as reference)
        base_metrics = comparison["models"]["base_model"]["metrics"]
        
        for model_key in ["finetuned_bf16", "finetuned_q4km"]:
            model_metrics = comparison["models"][model_key]["metrics"]
            
            comparison["relative_performance"][model_key] = {
                "latency_ratio": model_metrics["average_latency"] / base_metrics["average_latency"],
                "throughput_ratio": model_metrics["average_throughput"] / base_metrics["average_throughput"],
                "latency_change_percent": ((model_metrics["average_latency"] - base_metrics["average_latency"]) / base_metrics["average_latency"]) * 100,
                "throughput_change_percent": ((model_metrics["average_throughput"] - base_metrics["average_throughput"]) / base_metrics["average_throughput"]) * 100
            }
        
        # Statistical analysis
        latencies = {k: v["metrics"]["average_latency"] for k, v in comparison["models"].items()}
        throughputs = {k: v["metrics"]["average_throughput"] for k, v in comparison["models"].items()}
        
        comparison["statistical_analysis"] = {
            "latency": {
                "fastest_model": min(latencies, key=latencies.get),
                "slowest_model": max(latencies, key=latencies.get),
                "speed_range": max(latencies.values()) - min(latencies.values()),
                "coefficient_of_variation": np.std(list(latencies.values())) / np.mean(list(latencies.values()))
            },
            "throughput": {
                "highest_throughput": max(throughputs, key=throughputs.get),
                "lowest_throughput": min(throughputs, key=throughputs.get),
                "throughput_range": max(throughputs.values()) - min(throughputs.values()),
                "coefficient_of_variation": np.std(list(throughputs.values())) / np.mean(list(throughputs.values()))
            }
        }
        
        return comparison
    
    def compare_memory_performance(self) -> Dict[str, Any]:
        """Compare memory retrieval performance across all models"""
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "relative_performance": {},
            "statistical_analysis": {},
            "improvement_analysis": {}
        }
        
        # Extract memory metrics for each model
        for model_key, model_data in self.data.items():
            memory_data = model_data["memory"]
            summary = memory_data["summary"]
            
            comparison["models"][model_key] = {
                "name": model_data["name"],
                "model_path": model_data["model_path"],
                "metrics": {
                    "average_precision_at_5": summary["average_precision_at_5"],
                    "median_precision_at_5": summary["median_precision_at_5"],
                    "average_retrieval_time": summary["average_retrieval_time"],
                    "median_retrieval_time": summary["median_retrieval_time"],
                    "perfect_precision_queries": summary["perfect_precision_queries"],
                    "zero_precision_queries": summary["zero_precision_queries"],
                    "total_queries": summary["num_queries"],
                    "success_rate": summary["successful_queries"] / summary["num_queries"],
                    "precision_p95": summary["precision_distribution"]["p95"]
                }
            }
        
        # Calculate relative performance improvements
        base_metrics = comparison["models"]["base_model"]["metrics"]
        
        for model_key in ["finetuned_bf16", "finetuned_q4km"]:
            model_metrics = comparison["models"][model_key]["metrics"]
            
            comparison["relative_performance"][model_key] = {
                "precision_improvement": model_metrics["average_precision_at_5"] - base_metrics["average_precision_at_5"],
                "precision_improvement_percent": ((model_metrics["average_precision_at_5"] - base_metrics["average_precision_at_5"]) / base_metrics["average_precision_at_5"]) * 100,
                "retrieval_time_ratio": model_metrics["average_retrieval_time"] / base_metrics["average_retrieval_time"],
                "retrieval_time_change_percent": ((model_metrics["average_retrieval_time"] - base_metrics["average_retrieval_time"]) / base_metrics["average_retrieval_time"]) * 100,
                "perfect_queries_improvement": model_metrics["perfect_precision_queries"] - base_metrics["perfect_precision_queries"],
                "zero_queries_change": model_metrics["zero_precision_queries"] - base_metrics["zero_precision_queries"]
            }
        
        # Statistical analysis
        precisions = {k: v["metrics"]["average_precision_at_5"] for k, v in comparison["models"].items()}
        retrieval_times = {k: v["metrics"]["average_retrieval_time"] for k, v in comparison["models"].items()}
        
        comparison["statistical_analysis"] = {
            "precision": {
                "best_model": max(precisions, key=precisions.get),
                "worst_model": min(precisions, key=precisions.get),
                "precision_range": max(precisions.values()) - min(precisions.values()),
                "mean_precision": np.mean(list(precisions.values())),
                "std_precision": np.std(list(precisions.values()))
            },
            "retrieval_time": {
                "fastest_model": min(retrieval_times, key=retrieval_times.get),
                "slowest_model": max(retrieval_times, key=retrieval_times.get),
                "time_range": max(retrieval_times.values()) - min(retrieval_times.values())
            }
        }
        
        # Fine-tuning improvement analysis
        comparison["improvement_analysis"] = {
            "expected_vs_actual": {
                "expected_precision_threshold": 0.6,
                "base_precision": base_metrics["average_precision_at_5"],
                "bf16_precision": comparison["models"]["finetuned_bf16"]["metrics"]["average_precision_at_5"],
                "q4km_precision": comparison["models"]["finetuned_q4km"]["metrics"]["average_precision_at_5"],
                "bf16_meets_expectation": comparison["models"]["finetuned_bf16"]["metrics"]["average_precision_at_5"] > 0.6,
                "q4km_meets_expectation": comparison["models"]["finetuned_q4km"]["metrics"]["average_precision_at_5"] > 0.6
            },
            "overall_assessment": self._assess_fine_tuning_success(comparison)
        }
        
        return comparison
    
    def _assess_fine_tuning_success(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall fine-tuning success"""
        base_precision = comparison["models"]["base_model"]["metrics"]["average_precision_at_5"]
        bf16_precision = comparison["models"]["finetuned_bf16"]["metrics"]["average_precision_at_5"]
        q4km_precision = comparison["models"]["finetuned_q4km"]["metrics"]["average_precision_at_5"]
        
        return {
            "fine_tuning_effective": bf16_precision > base_precision or q4km_precision > base_precision,
            "best_fine_tuned_model": "bf16" if bf16_precision > q4km_precision else "q4km",
            "precision_improvement_achieved": max(bf16_precision, q4km_precision) > base_precision,
            "quantization_impact": abs(bf16_precision - q4km_precision),
            "recommendation": self._generate_recommendation(base_precision, bf16_precision, q4km_precision)
        }
    
    def _generate_recommendation(self, base: float, bf16: float, q4km: float) -> str:
        """Generate model recommendation based on performance"""
        if bf16 > base and q4km > base:
            if abs(bf16 - q4km) < 0.05:  # Similar performance
                return "Q4_K_M recommended: Similar memory performance with better inference speed"
            elif bf16 > q4km:
                return "BF16 recommended: Best memory performance despite slower inference"
            else:
                return "Q4_K_M recommended: Better memory performance with good inference speed"
        elif bf16 > base:
            return "BF16 recommended: Only model showing memory improvement"
        elif q4km > base:
            return "Q4_K_M recommended: Only model showing memory improvement"
        else:
            return "Base model recommended: Fine-tuning did not improve memory performance"
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        inference_comparison = self.compare_inference_performance()
        memory_comparison = self.compare_memory_performance()
        
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "comparison_type": "comprehensive_model_analysis",
                "models_compared": list(self.data.keys()),
                "benchmark_types": ["inference", "memory_retrieval"]
            },
            "executive_summary": self._generate_executive_summary(inference_comparison, memory_comparison),
            "inference_analysis": inference_comparison,
            "memory_analysis": memory_comparison,
            "trade_off_analysis": self._analyze_trade_offs(inference_comparison, memory_comparison),
            "recommendations": self._generate_final_recommendations(inference_comparison, memory_comparison)
        }
        
        return report
    
    def _generate_executive_summary(self, inference_comp: Dict, memory_comp: Dict) -> Dict[str, Any]:
        """Generate executive summary of all comparisons"""
        base_inf = inference_comp["models"]["base_model"]["metrics"]
        bf16_inf = inference_comp["models"]["finetuned_bf16"]["metrics"] 
        q4km_inf = inference_comp["models"]["finetuned_q4km"]["metrics"]
        
        base_mem = memory_comp["models"]["base_model"]["metrics"]
        bf16_mem = memory_comp["models"]["finetuned_bf16"]["metrics"]
        q4km_mem = memory_comp["models"]["finetuned_q4km"]["metrics"]
        
        return {
            "key_findings": {
                "fastest_inference": inference_comp["statistical_analysis"]["latency"]["fastest_model"],
                "best_memory_precision": memory_comp["statistical_analysis"]["precision"]["best_model"],
                "fine_tuning_improved_memory": memory_comp["improvement_analysis"]["overall_assessment"]["precision_improvement_achieved"],
                "quantization_preserves_quality": abs(bf16_mem["average_precision_at_5"] - q4km_mem["average_precision_at_5"]) < 0.05
            },
            "performance_summary": {
                "base_model": {
                    "inference_throughput": f"{base_inf['average_throughput']:.2f} tok/s",
                    "memory_precision": f"{base_mem['average_precision_at_5']:.3f}",
                    "inference_latency": f"{base_inf['average_latency']:.2f}s"
                },
                "finetuned_bf16": {
                    "inference_throughput": f"{bf16_inf['average_throughput']:.2f} tok/s",
                    "memory_precision": f"{bf16_mem['average_precision_at_5']:.3f}",
                    "inference_latency": f"{bf16_inf['average_latency']:.2f}s",
                    "vs_base_precision": f"{((bf16_mem['average_precision_at_5'] - base_mem['average_precision_at_5']) / base_mem['average_precision_at_5'] * 100):+.1f}%"
                },
                "finetuned_q4km": {
                    "inference_throughput": f"{q4km_inf['average_throughput']:.2f} tok/s", 
                    "memory_precision": f"{q4km_mem['average_precision_at_5']:.3f}",
                    "inference_latency": f"{q4km_inf['average_latency']:.2f}s",
                    "vs_base_precision": f"{((q4km_mem['average_precision_at_5'] - base_mem['average_precision_at_5']) / base_mem['average_precision_at_5'] * 100):+.1f}%"
                }
            }
        }
    
    def _analyze_trade_offs(self, inference_comp: Dict, memory_comp: Dict) -> Dict[str, Any]:
        """Analyze trade-offs between models"""
        return {
            "speed_vs_memory_quality": {
                "bf16": {
                    "inference_speed_rank": 3,  # Slowest
                    "memory_quality_rank": self._rank_memory_quality("finetuned_bf16", memory_comp),
                    "trade_off_ratio": memory_comp["models"]["finetuned_bf16"]["metrics"]["average_precision_at_5"] / inference_comp["models"]["finetuned_bf16"]["metrics"]["average_throughput"]
                },
                "q4km": {
                    "inference_speed_rank": self._rank_inference_speed("finetuned_q4km", inference_comp),
                    "memory_quality_rank": self._rank_memory_quality("finetuned_q4km", memory_comp),
                    "trade_off_ratio": memory_comp["models"]["finetuned_q4km"]["metrics"]["average_precision_at_5"] / inference_comp["models"]["finetuned_q4km"]["metrics"]["average_throughput"]
                },
                "base": {
                    "inference_speed_rank": self._rank_inference_speed("base_model", inference_comp),
                    "memory_quality_rank": self._rank_memory_quality("base_model", memory_comp),
                    "trade_off_ratio": memory_comp["models"]["base_model"]["metrics"]["average_precision_at_5"] / inference_comp["models"]["base_model"]["metrics"]["average_throughput"]
                }
            },
            "model_efficiency": {
                "most_efficient_overall": self._determine_most_efficient(inference_comp, memory_comp),
                "best_for_memory_tasks": memory_comp["statistical_analysis"]["precision"]["best_model"],
                "best_for_speed_tasks": inference_comp["statistical_analysis"]["latency"]["fastest_model"]
            }
        }
    
    def _rank_inference_speed(self, model_key: str, inference_comp: Dict) -> int:
        """Rank model by inference speed (1 = fastest)"""
        throughputs = {k: v["metrics"]["average_throughput"] for k, v in inference_comp["models"].items()}
        sorted_models = sorted(throughputs.items(), key=lambda x: x[1], reverse=True)
        for rank, (model, _) in enumerate(sorted_models, 1):
            if model == model_key:
                return rank
        return len(sorted_models)
    
    def _rank_memory_quality(self, model_key: str, memory_comp: Dict) -> int:
        """Rank model by memory quality (1 = best)"""
        precisions = {k: v["metrics"]["average_precision_at_5"] for k, v in memory_comp["models"].items()}
        sorted_models = sorted(precisions.items(), key=lambda x: x[1], reverse=True)
        for rank, (model, _) in enumerate(sorted_models, 1):
            if model == model_key:
                return rank
        return len(sorted_models)
    
    def _determine_most_efficient(self, inference_comp: Dict, memory_comp: Dict) -> str:
        """Determine most efficient model overall"""
        # Simple efficiency score: memory_precision / inference_latency
        scores = {}
        for model_key in inference_comp["models"].keys():
            precision = memory_comp["models"][model_key]["metrics"]["average_precision_at_5"]
            latency = inference_comp["models"][model_key]["metrics"]["average_latency"]
            scores[model_key] = precision / latency
        
        return max(scores, key=scores.get)
    
    def _generate_final_recommendations(self, inference_comp: Dict, memory_comp: Dict) -> Dict[str, Any]:
        """Generate final recommendations"""
        return {
            "use_case_recommendations": {
                "production_deployment": {
                    "recommended_model": "finetuned_q4km",
                    "reason": "Best balance of memory performance and inference speed"
                },
                "memory_critical_applications": {
                    "recommended_model": memory_comp["statistical_analysis"]["precision"]["best_model"],
                    "reason": "Highest memory retrieval precision"
                },
                "speed_critical_applications": {
                    "recommended_model": inference_comp["statistical_analysis"]["latency"]["fastest_model"],
                    "reason": "Fastest inference with acceptable memory performance"
                },
                "resource_constrained_environments": {
                    "recommended_model": "finetuned_q4km",
                    "reason": "Good performance with lower memory footprint"
                }
            },
            "fine_tuning_assessment": {
                "was_successful": memory_comp["improvement_analysis"]["overall_assessment"]["precision_improvement_achieved"],
                "best_fine_tuned_variant": memory_comp["improvement_analysis"]["overall_assessment"]["best_fine_tuned_model"],
                "quantization_impact": "minimal" if memory_comp["improvement_analysis"]["overall_assessment"]["quantization_impact"] < 0.05 else "significant",
                "next_steps": self._suggest_next_steps(memory_comp)
            }
        }
    
    def _suggest_next_steps(self, memory_comp: Dict) -> List[str]:
        """Suggest next steps based on results"""
        suggestions = []
        
        if memory_comp["improvement_analysis"]["overall_assessment"]["precision_improvement_achieved"]:
            suggestions.append("Deploy fine-tuned model to production")
            suggestions.append("Conduct A/B testing with real users")
        else:
            suggestions.append("Investigate fine-tuning data quality")
            suggestions.append("Consider additional training epochs")
            suggestions.append("Experiment with different learning rates")
        
        suggestions.append("Benchmark on domain-specific evaluation sets")
        suggestions.append("Monitor performance in production environment")
        
        return suggestions
    
    def save_analysis(self):
        """Save all analysis results to files"""
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        # Save comprehensive report
        report_file = self.output_dir / "comprehensive_model_comparison.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save individual analysis files
        inference_file = self.output_dir / "inference_comparison.json"
        with open(inference_file, 'w') as f:
            json.dump(report["inference_analysis"], f, indent=2)
        
        memory_file = self.output_dir / "memory_comparison.json"
        with open(memory_file, 'w') as f:
            json.dump(report["memory_analysis"], f, indent=2)
        
        # Generate summary report
        summary = self._generate_text_summary(report)
        summary_file = self.output_dir / "model_comparison_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"✅ Analysis complete! Results saved to {self.output_dir}")
        print(f"📊 Comprehensive report: {report_file}")
        print(f"📈 Inference analysis: {inference_file}")
        print(f"🧠 Memory analysis: {memory_file}")
        print(f"📝 Summary report: {summary_file}")
        
        return report
    
    def _generate_text_summary(self, report: Dict[str, Any]) -> str:
        """Generate human-readable text summary"""
        summary = f"""
# 🚀 Model Comparison Analysis Summary
Generated: {report['metadata']['generated_at']}

## 📊 Executive Summary

### Key Findings:
- **Fastest Inference:** {report['executive_summary']['key_findings']['fastest_inference']}
- **Best Memory Precision:** {report['executive_summary']['key_findings']['best_memory_precision']}
- **Fine-tuning Improved Memory:** {'✅ Yes' if report['executive_summary']['key_findings']['fine_tuning_improved_memory'] else '❌ No'}
- **Quantization Preserves Quality:** {'✅ Yes' if report['executive_summary']['key_findings']['quantization_preserves_quality'] else '❌ No'}

## 🎯 Performance Summary

### Base Model (4-bit)
- **Inference:** {report['executive_summary']['performance_summary']['base_model']['inference_throughput']} | {report['executive_summary']['performance_summary']['base_model']['inference_latency']}
- **Memory Precision:** {report['executive_summary']['performance_summary']['base_model']['memory_precision']}

### Fine-tuned BF16
- **Inference:** {report['executive_summary']['performance_summary']['finetuned_bf16']['inference_throughput']} | {report['executive_summary']['performance_summary']['finetuned_bf16']['inference_latency']}
- **Memory Precision:** {report['executive_summary']['performance_summary']['finetuned_bf16']['memory_precision']} ({report['executive_summary']['performance_summary']['finetuned_bf16']['vs_base_precision']} vs base)

### Fine-tuned Q4_K_M
- **Inference:** {report['executive_summary']['performance_summary']['finetuned_q4km']['inference_throughput']} | {report['executive_summary']['performance_summary']['finetuned_q4km']['inference_latency']}
- **Memory Precision:** {report['executive_summary']['performance_summary']['finetuned_q4km']['memory_precision']} ({report['executive_summary']['performance_summary']['finetuned_q4km']['vs_base_precision']} vs base)

## 🎯 Recommendations

### Production Deployment
**Recommended:** {report['recommendations']['use_case_recommendations']['production_deployment']['recommended_model']}
**Reason:** {report['recommendations']['use_case_recommendations']['production_deployment']['reason']}

### Memory-Critical Applications
**Recommended:** {report['recommendations']['use_case_recommendations']['memory_critical_applications']['recommended_model']}
**Reason:** {report['recommendations']['use_case_recommendations']['memory_critical_applications']['reason']}

### Speed-Critical Applications
**Recommended:** {report['recommendations']['use_case_recommendations']['speed_critical_applications']['recommended_model']}
**Reason:** {report['recommendations']['use_case_recommendations']['speed_critical_applications']['reason']}

## 🔬 Fine-tuning Assessment

**Success:** {'✅ Yes' if report['recommendations']['fine_tuning_assessment']['was_successful'] else '❌ No'}
**Best Variant:** {report['recommendations']['fine_tuning_assessment']['best_fine_tuned_variant']}
**Quantization Impact:** {report['recommendations']['fine_tuning_assessment']['quantization_impact']}

### Next Steps:
"""
        for step in report['recommendations']['fine_tuning_assessment']['next_steps']:
            summary += f"- {step}\n"
        
        summary += f"""
## 📈 Detailed Metrics

### Inference Performance
- **Fastest Model:** {report['inference_analysis']['statistical_analysis']['latency']['fastest_model']}
- **Highest Throughput:** {report['inference_analysis']['statistical_analysis']['throughput']['highest_throughput']}

### Memory Performance  
- **Best Precision:** {report['memory_analysis']['statistical_analysis']['precision']['best_model']}
- **Fastest Retrieval:** {report['memory_analysis']['statistical_analysis']['retrieval_time']['fastest_model']}

---
📁 Full analysis data available in JSON files.
"""
        return summary


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results across models")
    parser.add_argument("--base-results", required=True, help="Path to base model results directory")
    parser.add_argument("--bf16-results", required=True, help="Path to BF16 model results directory")
    parser.add_argument("--q4km-results", required=True, help="Path to Q4_K_M model results directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    print("🚀 Starting benchmark comparison analysis...")
    print(f"📂 Base results: {args.base_results}")
    print(f"📂 BF16 results: {args.bf16_results}")
    print(f"📂 Q4_K_M results: {args.q4km_results}")
    print(f"📂 Output directory: {args.output_dir}")
    
    comparator = BenchmarkComparator(
        base_dir=args.base_results,
        bf16_dir=args.bf16_results,
        q4km_dir=args.q4km_results,
        output_dir=args.output_dir
    )
    
    report = comparator.save_analysis()
    
    print("\n🎉 Comparison analysis complete!")
    print(f"📊 Check {args.output_dir} for detailed results")


if __name__ == "__main__":
    main()
