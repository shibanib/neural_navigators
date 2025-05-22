import numpy as np
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from data_loader import SteinmetzDataLoader
from neural_analysis import NeuralAnalyzer
from behavior_analysis import BehaviorAnalyzer
from visualization import NeuralViz

class AnalysisPipeline:
    """Class for automated batch processing of neural data."""
    
    def __init__(self, output_dir: str = 'results'):
        """Initialize pipeline with output directory."""
        self.output_dir = output_dir
        self.loader = SteinmetzDataLoader()
        self.neural_analyzer = NeuralAnalyzer()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.visualizer = NeuralViz()
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        
    def process_session(self, session_idx: int, 
                       analyses: List[str] = None) -> Dict[str, Any]:
        """
        Process a single session with specified analyses.
        
        Args:
            session_idx: Index of session to process
            analyses: List of analyses to perform
        """
        if analyses is None:
            analyses = ['basic', 'lfp', 'population', 'behavior', 'cross_regional']
            
        results = {}
        session_data = self.loader.load_session(session_idx)
        
        # Create session directory
        session_dir = os.path.join(self.output_dir, f'session_{session_idx}')
        os.makedirs(session_dir, exist_ok=True)
        
        try:
            if 'basic' in analyses:
                results['basic'] = self._run_basic_analysis(session_data, session_dir)
            
            if 'lfp' in analyses:
                results['lfp'] = self._run_lfp_analysis(session_data, session_dir)
                
            if 'population' in analyses:
                results['population'] = self._run_population_analysis(session_data, session_dir)
                
            if 'behavior' in analyses:
                results['behavior'] = self._run_behavior_analysis(session_data, session_dir)
                
            if 'cross_regional' in analyses:
                results['cross_regional'] = self._run_cross_regional_analysis(session_data, session_dir)
                
            # Save results
            self._save_results(results, session_dir)
            
        except Exception as e:
            print(f"Error processing session {session_idx}: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def batch_process(self, session_indices: List[int], 
                     analyses: List[str] = None,
                     n_workers: int = 4) -> Dict[int, Dict[str, Any]]:
        """
        Process multiple sessions in parallel.
        
        Args:
            session_indices: List of session indices to process
            analyses: List of analyses to perform
            n_workers: Number of parallel workers
        """
        all_results = {}
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_session = {
                executor.submit(self.process_session, idx, analyses): idx 
                for idx in session_indices
            }
            
            for future in future_to_session:
                session_idx = future_to_session[future]
                try:
                    all_results[session_idx] = future.result()
                except Exception as e:
                    print(f"Error in session {session_idx}: {str(e)}")
                    all_results[session_idx] = {'error': str(e)}
                    
        return all_results
    
    def _run_basic_analysis(self, session_data: Dict[str, Any], 
                          output_dir: str) -> Dict[str, Any]:
        """Run basic neural analysis."""
        results = {}
        
        # Compute PSTHs
        for i, neuron_spikes in enumerate(session_data['spikes'][:5]):  # First 5 neurons
            psth, time_bins = self.neural_analyzer.compute_psth(
                neuron_spikes, (-0.5, 0.5)
            )
            results[f'neuron_{i}_psth'] = psth
            
            # Save PSTH plot
            fig = self.visualizer.plot_raster(
                neuron_spikes[0], neuron_spikes, (-0.5, 0.5),
                title=f'Neuron {i} Raster'
            )
            fig.savefig(os.path.join(output_dir, f'neuron_{i}_raster.png'))
            plt.close(fig)
            
        return results
    
    def _run_lfp_analysis(self, session_data: Dict[str, Any], 
                         output_dir: str) -> Dict[str, Any]:
        """Run LFP analysis."""
        results = {}
        
        # Compute power spectra
        for i in range(min(5, session_data['lfp'].shape[1])):  # First 5 channels
            freqs, power = self.neural_analyzer.compute_lfp_power(
                session_data['lfp'][:, i],
                freq_range=(1, 100)
            )
            results[f'channel_{i}_power'] = power
            
            # Save power spectrum plot
            fig, ax = plt.subplots()
            ax.semilogy(freqs, power)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power')
            fig.savefig(os.path.join(output_dir, f'channel_{i}_power.png'))
            plt.close(fig)
            
        return results
    
    def _run_population_analysis(self, session_data: Dict[str, Any], 
                               output_dir: str) -> Dict[str, Any]:
        """Run population analysis."""
        results = {}
        
        # Compute population activity
        time_bins = np.arange(-0.5, 0.5, 0.01)
        firing_rates = self.loader.compute_firing_rates(
            session_data['spikes'],
            time_bins
        )
        
        # Run PCA
        projected_data, pca = self.neural_analyzer.compute_population_dynamics(firing_rates)
        results['pca_components'] = projected_data
        results['explained_variance'] = pca.explained_variance_ratio_
        
        # Save population activity plot
        fig = self.visualizer.plot_population_activity(
            firing_rates, time_bins, sort_by='rate'
        )
        fig.savefig(os.path.join(output_dir, 'population_activity.png'))
        plt.close(fig)
        
        return results
    
    def _run_behavior_analysis(self, session_data: Dict[str, Any], 
                             output_dir: str) -> Dict[str, Any]:
        """Run behavioral analysis."""
        results = {}
        
        # Generate example behavioral data (replace with actual data when available)
        n_trials = len(session_data['spikes'][0])
        example_choices = np.random.randint(0, 2, n_trials)
        example_outcomes = np.random.randint(0, 2, n_trials)
        
        # Analyze sequential effects
        seq_effects = self.behavior_analyzer.compute_sequential_effects(
            example_choices, example_outcomes
        )
        results['sequential_effects'] = seq_effects
        
        return results
    
    def _run_cross_regional_analysis(self, session_data: Dict[str, Any], 
                                   output_dir: str) -> Dict[str, Any]:
        """Run cross-regional analysis."""
        results = {}
        
        # Compute coherence between regions
        if 'brain_area_lfp' in session_data:
            unique_areas = np.unique(session_data['brain_area_lfp'])
            n_areas = len(unique_areas)
            coherence_matrix = np.zeros((n_areas, n_areas))
            
            for i, area1 in enumerate(unique_areas):
                for j, area2 in enumerate(unique_areas):
                    if i != j:
                        # Get LFP from both regions
                        lfp1 = session_data['lfp'][:, session_data['brain_area_lfp'] == area1].mean(axis=1)
                        lfp2 = session_data['lfp'][:, session_data['brain_area_lfp'] == area2].mean(axis=1)
                        
                        # Compute coherence
                        f, Cxy = signal.coherence(lfp1, lfp2, fs=100)
                        coherence_matrix[i, j] = np.mean(Cxy)
            
            results['coherence_matrix'] = coherence_matrix
            
            # Save coherence matrix plot
            fig = self.visualizer.plot_connectivity_matrix(
                coherence_matrix, unique_areas, 
                title='Inter-regional Coherence'
            )
            fig.savefig(os.path.join(output_dir, 'coherence_matrix.png'))
            plt.close(fig)
        
        return results
    
    def _save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """Save analysis results."""
        # Save numerical results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f'results_{timestamp}.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value.tolist() if isinstance(value, np.ndarray) else value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f) 