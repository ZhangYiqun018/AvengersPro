"""
Cost/Performance Weight Ablation Study

Conducts systematic ablation studies on cost_sensitivity and performance_weight 
parameters while keeping their sum equal to 1.0. Analyzes the Pareto frontier
of performance vs cost trade-offs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

# Import internal modules
from .data_collector import AblationDataCollector
from .visualization import AblationVisualizer


class WeightAblation:
    """
    Conducts cost/performance weight ablation experiments for the Balance Cluster Router.
    
    This class systematically varies the balance between performance and cost optimization
    by adjusting performance_weight and cost_sensitivity parameters while maintaining 
    their sum equal to 1.0.
    """
    
    def __init__(self, output_dir: str = "ablation"):
        """
        Initialize weight ablation experiment.
        
        Args:
            output_dir: Directory to save results and figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_collector = AblationDataCollector(cache_dir=str(self.output_dir / "results"))
        self.visualizer = AblationVisualizer(output_dir=str(self.output_dir / "figures"))
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup logging if not configured
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def generate_weight_configurations(self, 
                                     performance_weight_range: Tuple[float, float] = (0.0, 1.0),
                                     step_size: float = 0.1,
                                     quiet: bool = False) -> List[Tuple[float, float]]:
        """
        Generate list of (performance_weight, cost_sensitivity) pairs to test.
        
        Args:
            performance_weight_range: Tuple of (min_performance_weight, max_performance_weight)
            step_size: Step size for performance_weight iteration (e.g., 0.01, 0.1)
            quiet: Whether to minimize logging output
            
        Returns:
            List of (performance_weight, cost_sensitivity) tuples
        """
        min_perf_weight, max_perf_weight = performance_weight_range
        
        # Generate performance weights with specified step size
        # Use linspace for more precise handling of endpoints
        num_steps = int(round((max_perf_weight - min_perf_weight) / step_size)) + 1
        performance_weights = np.linspace(min_perf_weight, max_perf_weight, num_steps)
        
        # Calculate corresponding cost sensitivities (cost_sensitivity = 1 - performance_weight)
        weight_pairs = []
        for perf_weight in performance_weights:
            cost_sens = 1.0 - perf_weight
            weight_pairs.append((float(perf_weight), float(cost_sens)))
        
        if not quiet:
            self.logger.info(f"Generated {len(weight_pairs)} weight configurations (step_size={step_size}):")
            for i, (pw, cs) in enumerate(weight_pairs):
                self.logger.info(f"  {i+1:2d}: performance_weight={pw:.3f}, cost_sensitivity={cs:.3f}")
        
        return weight_pairs
    
    def run_weight_ablation(self, base_config: Dict[str, Any],
                          performance_weight_range: Tuple[float, float] = (0.0, 1.0),
                          step_size: float = 0.1,
                          load_baseline: bool = True,
                          parallel: bool = False,
                          max_workers: Optional[int] = None,
                          quiet: bool = False) -> Dict[str, Any]:
        """
        Run complete weight ablation study.
        
        Args:
            base_config: Base configuration for experiments
            performance_weight_range: Tuple of (min_performance_weight, max_performance_weight)
            step_size: Step size for performance_weight iteration (e.g., 0.01, 0.1)
            load_baseline: Whether to load baseline performance data
            parallel: Whether to run experiments in parallel
            max_workers: Maximum number of parallel workers (default: CPU count - 1)
            quiet: Whether to minimize output and show progress bar
            
        Returns:
            Dictionary containing all experimental results and analysis
        """
        if not quiet:
            self.logger.info("Starting cost/performance weight ablation study")
        
        # Generate weight configurations with constraint: performance_weight + cost_sensitivity = 1.0
        weight_pairs = self.generate_weight_configurations(
            performance_weight_range=performance_weight_range,
            step_size=step_size,
            quiet=quiet
        )
        
        # Create parameter configurations as constrained pairs (not all combinations)
        # Each experiment uses a specific (performance_weight, cost_sensitivity) pair where sum = 1.0
        parameter_configs = []
        for perf_weight, cost_sens in weight_pairs:
            parameter_configs.append({
                'performance_weight': perf_weight,
                'cost_sensitivity': cost_sens
            })
        
        # Run parameter sweep for constrained weight pairs
        if parallel:
            if not quiet:
                self.logger.info(f"Running {len(weight_pairs)} weight experiments in parallel with {max_workers or 'auto'} workers")
            results = self._run_weight_experiments_parallel(
                base_config=base_config,
                parameter_configs=parameter_configs,
                max_workers=max_workers,
                quiet=quiet
            )
        else:
            if not quiet:
                self.logger.info(f"Running experiments for {len(weight_pairs)} weight combinations")
            results = self._run_weight_experiments_sequential(
                base_config=base_config,
                parameter_configs=parameter_configs
            )
        
        # Load baseline data if requested
        baseline_data = None
        if load_baseline:
            baseline_data = self._load_baseline_data()
        
        # Analyze results
        analysis = self._analyze_weight_results(results, baseline_data)
        
        # Generate visualizations
        if not quiet:
            self.logger.info("Generating visualizations")
        figure_paths = self._create_visualizations(results, baseline_data)
        
        # Compile final results
        final_results = {
            'experiment_type': 'weight_ablation',
            'timestamp': datetime.now().isoformat(),
            'base_config': base_config,
            'weight_configurations': weight_pairs,
            'total_experiments': len(results),
            'successful_experiments': len([r for r in results if not r.get('failed', False)]),
            'results': results,
            'analysis': analysis,
            'baseline_data': baseline_data,
            'figure_paths': figure_paths
        }
        
        # Save results
        results_path = self.output_dir / "results" / "weight_ablation_complete.json"
        self.data_collector.save_sweep_results(results, "weight_ablation_sweep.json")
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        # Create concise results export
        concise_results_path = self._export_concise_results(results, baseline_data, base_config)
        if not quiet:
            self.logger.info(f"Concise results exported to {concise_results_path}")
        
        if not quiet:
            self.logger.info(f"Weight ablation study completed. Results saved to {results_path}")
        return final_results
    
    def _analyze_weight_results(self, results: List[Dict[str, Any]], 
                              baseline_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze weight ablation results.
        
        Args:
            results: List of experimental results
            baseline_data: Optional baseline performance data
            
        Returns:
            Analysis dictionary with insights and statistics
        """
        analysis = {
            'pareto_analysis': {},
            'trade_off_analysis': {},
            'optimal_configurations': {},
            'sensitivity_analysis': {},
            'baseline_comparison': {}
        }
        
        # Filter successful results
        successful_results = [r for r in results if not r.get('failed', False)]
        
        if not successful_results:
            self.logger.warning("No successful results to analyze")
            return analysis
        
        # Extract key metrics
        performance_weights = []
        cost_sensitivities = []
        accuracies = []
        total_costs = []
        avg_costs = []
        cost_efficiencies = []
        
        for result in successful_results:
            param_combo = result.get('parameter_combination', {})
            perf_weight = param_combo.get('performance_weight')
            cost_sens = param_combo.get('cost_sensitivity')
            
            if perf_weight is not None and cost_sens is not None:
                performance_weights.append(perf_weight)
                cost_sensitivities.append(cost_sens)
                accuracies.append(result.get('accuracy', 0.0))
                
                cost_analysis = result.get('cost_analysis', {})
                total_costs.append(cost_analysis.get('total_cost', 0.0))
                avg_costs.append(cost_analysis.get('avg_cost_per_query', 0.0))
                cost_efficiencies.append(cost_analysis.get('cost_efficiency', 0.0))
        
        if not performance_weights:
            return analysis
        
        # Pareto frontier analysis
        pareto_points = self._find_pareto_frontier(accuracies, avg_costs, maximize_y=True, minimize_x=True)
        pareto_indices = self._find_pareto_indices(successful_results, accuracies, avg_costs)
        
        analysis['pareto_analysis'] = {
            'pareto_frontier_points': len(pareto_points),
            'pareto_configurations': [
                {
                    'performance_weight': performance_weights[i],
                    'cost_sensitivity': cost_sensitivities[i],
                    'accuracy': accuracies[i],
                    'avg_cost': avg_costs[i],
                    'cost_efficiency': cost_efficiencies[i]
                }
                for i in pareto_indices
            ]
        }
        
        # Trade-off analysis
        analysis['trade_off_analysis'] = self._analyze_tradeoffs(
            performance_weights, cost_sensitivities, accuracies, avg_costs
        )
        
        # Find optimal configurations
        analysis['optimal_configurations'] = self._find_optimal_weight_configs(successful_results)
        
        # Sensitivity analysis
        analysis['sensitivity_analysis'] = self._analyze_weight_sensitivity(
            performance_weights, cost_sensitivities, accuracies, avg_costs, cost_efficiencies
        )
        
        # Baseline comparison
        if baseline_data:
            analysis['baseline_comparison'] = self._compare_with_baselines(
                successful_results, baseline_data
            )
        
        return analysis
    
    def _find_pareto_frontier(self, accuracies: List[float], costs: List[float],
                            maximize_y: bool = True, minimize_x: bool = True) -> List[Tuple[float, float]]:
        """
        Find Pareto frontier points in accuracy-cost space.
        
        Args:
            accuracies: List of accuracy values
            costs: List of cost values
            maximize_y: Whether to maximize accuracy
            minimize_x: Whether to minimize cost
            
        Returns:
            List of (cost, accuracy) Pareto optimal points
        """
        points = list(zip(costs, accuracies))
        pareto_points = []
        
        for i, (cost1, acc1) in enumerate(points):
            is_pareto = True
            
            for j, (cost2, acc2) in enumerate(points):
                if i != j:
                    # Check if point j dominates point i
                    cost_better = (cost2 <= cost1) if minimize_x else (cost2 >= cost1)
                    acc_better = (acc2 >= acc1) if maximize_y else (acc2 <= acc1)
                    
                    cost_strictly_better = (cost2 < cost1) if minimize_x else (cost2 > cost1)
                    acc_strictly_better = (acc2 > acc1) if maximize_y else (acc2 < acc1)
                    
                    if (cost_better and acc_better) and (cost_strictly_better or acc_strictly_better):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_points.append((cost1, acc1))
        
        return sorted(pareto_points)
    
    def _find_pareto_indices(self, results: List[Dict[str, Any]], 
                           accuracies: List[float], costs: List[float]) -> List[int]:
        """
        Find indices of Pareto optimal configurations.
        
        Args:
            results: List of experimental results
            accuracies: List of accuracy values
            costs: List of cost values
            
        Returns:
            List of indices corresponding to Pareto optimal points
        """
        pareto_points = self._find_pareto_frontier(accuracies, costs)
        pareto_indices = []
        
        for i, (cost, acc) in enumerate(zip(costs, accuracies)):
            if (cost, acc) in pareto_points:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def _analyze_tradeoffs(self, performance_weights: List[float], cost_sensitivities: List[float],
                         accuracies: List[float], costs: List[float]) -> Dict[str, Any]:
        """
        Analyze trade-offs between performance and cost.
        
        Args:
            performance_weights: List of performance weight values
            cost_sensitivities: List of cost sensitivity values  
            accuracies: List of accuracy values
            costs: List of cost values
            
        Returns:
            Trade-off analysis dictionary
        """
        trade_off_analysis = {}
        
        # Correlation analysis
        if len(performance_weights) > 2:
            corr_perf_acc = np.corrcoef(performance_weights, accuracies)[0, 1]
            corr_cost_sens_cost = np.corrcoef(cost_sensitivities, costs)[0, 1]
            
            if not np.isnan(corr_perf_acc):
                trade_off_analysis['performance_weight_accuracy_correlation'] = float(corr_perf_acc)
            if not np.isnan(corr_cost_sens_cost):
                trade_off_analysis['cost_sensitivity_cost_correlation'] = float(corr_cost_sens_cost)
        
        # Find extreme configurations
        max_perf_weight_idx = np.argmax(performance_weights)
        max_cost_sens_idx = np.argmax(cost_sensitivities)
        
        trade_off_analysis['extreme_configurations'] = {
            'max_performance_focus': {
                'performance_weight': performance_weights[max_perf_weight_idx],
                'cost_sensitivity': cost_sensitivities[max_perf_weight_idx],
                'accuracy': accuracies[max_perf_weight_idx],
                'cost': costs[max_perf_weight_idx]
            },
            'max_cost_focus': {
                'performance_weight': performance_weights[max_cost_sens_idx],
                'cost_sensitivity': cost_sensitivities[max_cost_sens_idx],
                'accuracy': accuracies[max_cost_sens_idx],
                'cost': costs[max_cost_sens_idx]
            }
        }
        
        # Calculate trade-off efficiency
        acc_range = max(accuracies) - min(accuracies)
        cost_range = max(costs) - min(costs) if max(costs) > 0 else 1.0
        
        trade_off_analysis['trade_off_efficiency'] = {
            'accuracy_range': float(acc_range),
            'cost_range': float(cost_range),
            'normalized_trade_off_ratio': float(acc_range / cost_range) if cost_range > 0 else 0.0
        }
        
        return trade_off_analysis
    
    def _find_optimal_weight_configs(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Find optimal weight configurations based on different criteria.
        
        Args:
            results: List of experimental results
            
        Returns:
            Dictionary of optimal configurations
        """
        optimal_configs = {}
        
        if not results:
            return optimal_configs
        
        # Best accuracy
        best_acc_result = max(results, key=lambda x: x.get('accuracy', 0.0))
        param_combo = best_acc_result.get('parameter_combination', {})
        
        optimal_configs['best_accuracy'] = {
            'performance_weight': param_combo.get('performance_weight'),
            'cost_sensitivity': param_combo.get('cost_sensitivity'),
            'accuracy': best_acc_result.get('accuracy', 0.0),
            'cost_analysis': best_acc_result.get('cost_analysis', {})
        }
        
        # Lowest cost (with reasonable accuracy > 0.3)
        reasonable_results = [r for r in results if r.get('accuracy', 0.0) > 0.3]
        if reasonable_results:
            lowest_cost_result = min(reasonable_results, 
                                   key=lambda x: x.get('cost_analysis', {}).get('avg_cost_per_query', float('inf')))
            param_combo = lowest_cost_result.get('parameter_combination', {})
            
            optimal_configs['lowest_cost'] = {
                'performance_weight': param_combo.get('performance_weight'),
                'cost_sensitivity': param_combo.get('cost_sensitivity'),
                'accuracy': lowest_cost_result.get('accuracy', 0.0),
                'cost_analysis': lowest_cost_result.get('cost_analysis', {})
            }
        
        # Best cost efficiency
        cost_efficiencies = [(r, r.get('cost_analysis', {}).get('cost_efficiency', 0.0)) for r in results]
        best_eff_result = max(cost_efficiencies, key=lambda x: x[1])[0]
        param_combo = best_eff_result.get('parameter_combination', {})
        
        optimal_configs['best_cost_efficiency'] = {
            'performance_weight': param_combo.get('performance_weight'),
            'cost_sensitivity': param_combo.get('cost_sensitivity'),
            'accuracy': best_eff_result.get('accuracy', 0.0),
            'cost_efficiency': best_eff_result.get('cost_analysis', {}).get('cost_efficiency', 0.0),
            'cost_analysis': best_eff_result.get('cost_analysis', {})
        }
        
        # Balanced configuration (equal weights)
        balanced_results = [(r, abs(r.get('parameter_combination', {}).get('performance_weight', 0.5) - 0.5)) 
                          for r in results]
        most_balanced_result = min(balanced_results, key=lambda x: x[1])[0]
        param_combo = most_balanced_result.get('parameter_combination', {})
        
        optimal_configs['most_balanced'] = {
            'performance_weight': param_combo.get('performance_weight'),
            'cost_sensitivity': param_combo.get('cost_sensitivity'),
            'accuracy': most_balanced_result.get('accuracy', 0.0),
            'cost_analysis': most_balanced_result.get('cost_analysis', {})
        }
        
        return optimal_configs
    
    def _analyze_weight_sensitivity(self, performance_weights: List[float], cost_sensitivities: List[float],
                                  accuracies: List[float], costs: List[float], 
                                  cost_efficiencies: List[float]) -> Dict[str, Any]:
        """
        Analyze sensitivity of metrics to weight changes.
        
        Args:
            performance_weights: List of performance weight values
            cost_sensitivities: List of cost sensitivity values
            accuracies: List of accuracy values
            costs: List of cost values
            cost_efficiencies: List of cost efficiency values
            
        Returns:
            Sensitivity analysis dictionary
        """
        sensitivity = {}
        
        if len(performance_weights) < 3:
            return sensitivity
        
        # Calculate derivatives (gradients) using finite differences
        def calculate_gradient(x_vals, y_vals):
            if len(x_vals) != len(y_vals) or len(x_vals) < 2:
                return []
            gradients = []
            for i in range(1, len(x_vals)):
                dx = x_vals[i] - x_vals[i-1]
                dy = y_vals[i] - y_vals[i-1]
                if dx != 0:
                    gradients.append(dy / dx)
            return gradients
        
        # Sort by performance weight for gradient calculation
        sorted_data = sorted(zip(performance_weights, accuracies, costs, cost_efficiencies))
        sorted_perf_weights, sorted_accs, sorted_costs, sorted_effs = zip(*sorted_data)
        
        # Calculate gradients
        acc_gradients = calculate_gradient(sorted_perf_weights, sorted_accs)
        cost_gradients = calculate_gradient(sorted_perf_weights, sorted_costs)
        eff_gradients = calculate_gradient(sorted_perf_weights, sorted_effs)
        
        if acc_gradients:
            sensitivity['accuracy_sensitivity'] = {
                'mean_gradient': float(np.mean(acc_gradients)),
                'std_gradient': float(np.std(acc_gradients)),
                'max_gradient': float(max(acc_gradients)),
                'min_gradient': float(min(acc_gradients))
            }
        
        if cost_gradients:
            sensitivity['cost_sensitivity'] = {
                'mean_gradient': float(np.mean(cost_gradients)),
                'std_gradient': float(np.std(cost_gradients)),
                'max_gradient': float(max(cost_gradients)),
                'min_gradient': float(min(cost_gradients))
            }
        
        if eff_gradients:
            sensitivity['efficiency_sensitivity'] = {
                'mean_gradient': float(np.mean(eff_gradients)),
                'std_gradient': float(np.std(eff_gradients)),
                'max_gradient': float(max(eff_gradients)),
                'min_gradient': float(min(eff_gradients))
            }
        
        return sensitivity
    
    def _compare_with_baselines(self, results: List[Dict[str, Any]], 
                              baseline_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare weight ablation results with baseline models.
        
        Args:
            results: Experimental results
            baseline_data: Baseline model data
            
        Returns:
            Baseline comparison dictionary
        """
        comparison = {}
        
        if not results or not baseline_data:
            return comparison
        
        # Get best router performance
        best_result = max(results, key=lambda x: x.get('accuracy', 0.0))
        best_accuracy = best_result.get('accuracy', 0.0)
        best_cost = best_result.get('cost_analysis', {}).get('avg_cost_per_query', 0.0)
        
        # Get baseline performance
        best_baseline = baseline_data.get('best_overall_baseline')
        if best_baseline:
            baseline_acc = best_baseline.get('avg_score', 0.0)
            
            comparison['best_router_vs_best_baseline'] = {
                'router_accuracy': best_accuracy,
                'baseline_accuracy': baseline_acc,
                'improvement': best_accuracy - baseline_acc,
                'improvement_percentage': ((best_accuracy - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0,
                'router_cost': best_cost
            }
        
        # Count how many router configs beat baseline
        model_summaries = baseline_data.get('model_summaries', [])
        if model_summaries:
            baseline_accuracies = [m.get('avg_score', 0.0) for m in model_summaries]
            max_baseline_acc = max(baseline_accuracies)
            
            better_than_baseline = [r for r in results if r.get('accuracy', 0.0) > max_baseline_acc]
            
            comparison['configurations_beating_baseline'] = {
                'total_configs': len(results),
                'configs_beating_best_baseline': len(better_than_baseline),
                'percentage': (len(better_than_baseline) / len(results) * 100) if results else 0
            }
        
        return comparison
    
    def _load_baseline_data(self) -> Optional[Dict[str, Any]]:
        """
        Load baseline performance data from configuration.
        
        Returns:
            Baseline data dictionary or None if not found
        """
        try:
            baseline_path = Path(__file__).parent.parent / "config" / "baseline.json"
            if baseline_path.exists():
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    baseline_scores = json.load(f)
                
                # Process baseline data similar to balance_cluster_router
                model_summaries = []
                for model, scores in baseline_scores.items():
                    valid_scores = [score / 100.0 if score > 1.0 else score 
                                  for score in scores.values() if score is not None]
                    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                    
                    model_summaries.append({
                        'model': model,
                        'avg_score': avg_score,
                        'total_cost': 0.0,  # Would need actual cost data
                        'dataset_coverage': f"{len(valid_scores)}/{len(scores)}"
                    })
                
                model_summaries.sort(key=lambda x: x['avg_score'], reverse=True)
                
                return {
                    'model_summaries': model_summaries,
                    'best_overall_baseline': model_summaries[0] if model_summaries else None
                }
        except Exception as e:
            self.logger.warning(f"Failed to load baseline data: {e}")
        
        return None
    
    def _create_visualizations(self, results: List[Dict[str, Any]], 
                             baseline_data: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Create all visualizations for weight ablation.
        
        Args:
            results: Experimental results
            baseline_data: Optional baseline data
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        figure_paths = {}
        
        try:
            # Weight ablation plot
            weight_path = self.visualizer.plot_weight_ablation(results, baseline_data)
            figure_paths['weight_ablation'] = weight_path
        except Exception as e:
            self.logger.error(f"Failed to create weight ablation plot: {e}")
        
        try:
            # Pareto frontier plot
            pareto_path = self.visualizer.plot_pareto_frontier(results, baseline_data)
            figure_paths['pareto_frontier'] = pareto_path
        except Exception as e:
            self.logger.error(f"Failed to create Pareto frontier plot: {e}")
        
        return figure_paths
    
    def _run_weight_experiments_sequential(self, base_config: Dict[str, Any], 
                                          parameter_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run weight experiments sequentially with constrained parameter pairs.
        
        Args:
            base_config: Base configuration dictionary
            parameter_configs: List of parameter dictionaries (each with performance_weight + cost_sensitivity = 1.0)
            
        Returns:
            List of experimental results
        """
        results = []
        
        for i, param_config in enumerate(parameter_configs):
            self.logger.info(f"Progress: {i+1}/{len(parameter_configs)} ({(i+1)/len(parameter_configs)*100:.1f}%)")
            
            # Create experiment config by merging base_config with specific parameter values
            experiment_config = base_config.copy()
            experiment_config.update(param_config)
            
            # Run single experiment
            result = self.data_collector.run_single_experiment(
                config_dict=experiment_config,
                experiment_type='weight_ablation'
            )
            
            # Add parameter combination info to result
            result['parameter_combination'] = param_config
            results.append(result)
        
        return results
    
    def _run_weight_experiments_parallel(self, base_config: Dict[str, Any], 
                                        parameter_configs: List[Dict[str, Any]],
                                        max_workers: Optional[int] = None,
                                        quiet: bool = False) -> List[Dict[str, Any]]:
        """
        Run weight experiments in parallel with constrained parameter pairs.
        
        Args:
            base_config: Base configuration dictionary
            parameter_configs: List of parameter dictionaries (each with performance_weight + cost_sensitivity = 1.0)
            max_workers: Maximum number of parallel workers
            quiet: Whether to show progress bar
            
        Returns:
            List of experimental results
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm
        
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
        
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_config = {}
            for param_config in parameter_configs:
                experiment_config = base_config.copy()
                experiment_config.update(param_config)
                
                future = executor.submit(
                    self.data_collector.run_single_experiment,
                    experiment_config,
                    'weight_ablation'
                )
                future_to_config[future] = param_config
            
            # Collect results with progress bar
            if quiet:
                progress_bar = tqdm(total=len(parameter_configs), desc="Weight experiments")
            
            for future in as_completed(future_to_config):
                param_config = future_to_config[future]
                try:
                    result = future.result()
                    # Add parameter combination info to result
                    result['parameter_combination'] = param_config
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Experiment failed for {param_config}: {e}")
                    results.append({
                        'parameter_combination': param_config,
                        'failed': True,
                        'error': str(e)
                    })
                
                if quiet:
                    progress_bar.update(1)
            
            if quiet:
                progress_bar.close()
        
        return results
    
    def _export_concise_results(self, results: List[Dict[str, Any]], 
                               baseline_data: Optional[Dict[str, Any]] = None,
                               base_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Export concise weight ablation results to a separate file.
        
        Args:
            results: List of experimental results
            baseline_data: Optional baseline model data
            base_config: Optional base configuration containing seed info
            
        Returns:
            Path to exported concise results file
        """
        concise_data = {
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'weight_ablation',
            'router_results': [],
            'baseline_results': []
        }
        
        # Process router experiment results
        for result in results:
            if result.get('failed', False):
                continue
                
            param_combo = result.get('parameter_combination', {})
            cost_analysis = result.get('cost_analysis', {})
            
            concise_data['router_results'].append({
                'performance_weight': param_combo.get('performance_weight'),
                'cost_sensitivity': param_combo.get('cost_sensitivity'),
                'accuracy': result.get('accuracy', 0.0),
                'total_cost': cost_analysis.get('total_cost', 0.0),
                'avg_cost_per_query': cost_analysis.get('avg_cost_per_query', 0.0),
                'cost_efficiency': cost_analysis.get('cost_efficiency', 0.0)
            })
        
        # Process baseline results if available
        if baseline_data:
            model_summaries = baseline_data.get('model_summaries', [])
            for model_info in model_summaries:
                concise_data['baseline_results'].append({
                    'model': model_info.get('model', 'Unknown'),
                    'accuracy': model_info.get('avg_score', 0.0),
                    'total_cost': model_info.get('total_cost', 0.0),
                    'cost_per_query': 0.0,  # Baseline models don't have per-query cost data
                    'dataset_coverage': model_info.get('dataset_coverage', 'Unknown')
                })
        
        # Sort router results by performance_weight
        concise_data['router_results'].sort(key=lambda x: x.get('performance_weight', 0))
        
        # Sort baseline results by accuracy (descending)
        concise_data['baseline_results'].sort(key=lambda x: x.get('accuracy', 0), reverse=True)
        
        # Save concise results
        concise_path = self.output_dir / "results" / "weight_ablation_concise.json"
        concise_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(concise_path, 'w', encoding='utf-8') as f:
            json.dump(concise_data, f, indent=2, ensure_ascii=False)
        
        # Also create a CSV version for easy analysis with seed in filename
        seed = base_config.get('seed', 42) if base_config else 42
        csv_filename = f"weight_ablation_results_seed{seed}.csv"
        csv_path = self.output_dir / "results" / csv_filename
        self._export_csv_results(concise_data, csv_path)
        
        return str(concise_path)
    
    def _export_csv_results(self, concise_data: Dict[str, Any], csv_path: Path):
        """
        Export results to CSV format for easy analysis.
        
        Args:
            concise_data: Concise results data
            csv_path: Path to save CSV file
        """
        import csv
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write router results
            writer.writerow(['# Router Experiment Results'])
            writer.writerow(['performance_weight', 'cost_sensitivity', 'accuracy', 'total_cost', 'avg_cost_per_query', 'cost_efficiency'])
            
            for result in concise_data['router_results']:
                writer.writerow([
                    result.get('performance_weight', ''),
                    result.get('cost_sensitivity', ''),
                    result.get('accuracy', ''),
                    result.get('total_cost', ''),
                    result.get('avg_cost_per_query', ''),
                    result.get('cost_efficiency', '')
                ])
            
            # Write baseline results if available
            if concise_data['baseline_results']:
                writer.writerow([])  # Empty row
                writer.writerow(['# Baseline Model Results'])
                writer.writerow(['model', 'accuracy', 'total_cost', 'dataset_coverage'])
                
                for baseline in concise_data['baseline_results']:
                    writer.writerow([
                        baseline.get('model', ''),
                        baseline.get('accuracy', ''),
                        baseline.get('total_cost', ''),
                        baseline.get('dataset_coverage', '')
                    ])