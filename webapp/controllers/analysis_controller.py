import sys
import os

# Robustly add src directory to sys.path
_CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_CONTROLLER_DIR, '..', '..', 'src'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import json
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from neural_analysis import NeuralAnalyzer
from behavior_analysis import BehaviorAnalyzer
from .data_controller import DataController
import logging  # Add logging import

# Set up basic logging 
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger('analysis_controller')

class AnalysisController:
    """Controller for managing analysis operations"""
    
    def __init__(self):
        logger.info("Initializing AnalysisController")
        self.neural_analyzer = NeuralAnalyzer()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.data_controller = DataController()
        
        # Define available analyses
        self.available_analyses = {
            'basic': self._run_basic_analysis,
            'lfp': self._run_lfp_analysis,
            'population': self._run_population_analysis,
            'behavior': self._run_behavior_analysis,
            'cross_regional': self._run_cross_regional_analysis,
            'choice_aligned': self._run_choice_aligned_analysis
        }
        logger.info(f"Available analyses: {list(self.available_analyses.keys())}")
        
        # Default configurations
        self.default_configs = {
            'basic': {
                'time_window': (-0.5, 0.5),
                'bin_size': 0.01
            },
            'lfp': {
                'freq_range': (1, 100),
                'freq_bands': {
                    'theta': (4, 8),
                    'beta': (13, 30),
                    'gamma': (30, 80)
                }
            },
            'population': {
                'n_components': 10,
                'scale_data': True
            },
            'behavior': {
                'n_lags': 5
            },
            'cross_regional': {
                'method': 'coherence'
            },
            'choice_aligned': {
                'example_param': 'default_value'
            }
        }
    
    def get_available_analyses(self):
        """Get a list of available analysis types"""
        return list(self.available_analyses.keys())
    
    def run_analyses(self, session_indices, analyses, config=None):
        """Run selected analyses on selected sessions"""
        logger.info(f"Running analyses: {analyses} on sessions: {session_indices}")
        
        # Merge with default configs
        merged_config = self._merge_configs(config)
        
        # Process each session
        results = {}
        for session_idx in session_indices:
            try:
                logger.info(f"Processing session {session_idx}")
                session_results = self.process_session(
                    session_idx=session_idx,
                    analyses=analyses,
                    config=merged_config
                )
                results[str(session_idx)] = session_results
                logger.info(f"Session {session_idx} processed with results keys: {list(session_results.keys())}")
            except Exception as e:
                logger.error(f"Error processing session {session_idx}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                results[str(session_idx)] = {'error': str(e)}
        
        return results
    
    def process_session(self, session_idx, analyses, config=None):
        """Process a single session with selected analyses"""
        logger.info(f"Processing session {session_idx} with analyses: {analyses}")
        
        # Merge with default configs
        merged_config = self._merge_configs(config)
        
        # Load session data
        try:
            session_data = self.data_controller.load_session(session_idx)
            logger.info(f"Session {session_idx} data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading session {session_idx}: {str(e)}")
            return {'error': f"Failed to load session data: {str(e)}"}
        
        # Run each analysis
        results = {}
        for analysis in analyses:
            if analysis in self.available_analyses:
                try:
                    logger.info(f"Running {analysis} analysis for session {session_idx}")
                    analysis_fn = self.available_analyses[analysis]
                    analysis_config = merged_config.get(analysis, {})
                    results[analysis] = analysis_fn(session_data, analysis_config)
                    logger.info(f"Analysis {analysis} completed with results keys: {list(results[analysis].keys())}")
                except Exception as e:
                    logger.error(f"Error in {analysis} analysis: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    results[analysis] = {'error': str(e)}
            else:
                logger.warning(f"Analysis type {analysis} not available")
                results[analysis] = {'error': f"Analysis type {analysis} not available"}
                
        return results
    
    def batch_process(self, session_indices, analyses, config=None, n_workers=4):
        """Process multiple sessions in parallel"""
        merged_config = self._merge_configs(config)
        results = {}
        
        # Setup parallel processing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for session_idx in session_indices:
                future = executor.submit(
                    self.process_session,
                    session_idx=session_idx,
                    analyses=analyses,
                    config=merged_config
                )
                futures.append((session_idx, future))
            
            # Collect results
            for session_idx, future in futures:
                results[str(session_idx)] = future.result()
        
        return results
    
    def generate_summary_report(self, results):
        """Create a summary report of analyses across sessions"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_sessions': len(results),
            'session_summaries': {}
        }
        
        for session_idx, session_results in results.items():
            session_summary = {
                'analyses_completed': [k for k in session_results.keys() if k != 'error'],
                'error': session_results.get('error', None)
            }
            
            # Add analysis-specific metrics
            if 'population' in session_results:
                pop_results = session_results['population']
                if 'explained_variance' in pop_results:
                    explained_var = pop_results['explained_variance']
                    n_components_80 = np.where(
                        np.cumsum(explained_var) >= 0.8
                    )[0][0] + 1 if len(explained_var) > 0 else 0
                    session_summary['n_components_80_var'] = int(n_components_80)
            
            summary['session_summaries'][session_idx] = session_summary
        
        return summary
    
    def _merge_configs(self, user_config):
        """Merge user configuration with defaults"""
        merged_config = {}
        
        if user_config is None:
            user_config = {}
        
        # For each analysis type
        for analysis_type, default_config in self.default_configs.items():
            user_analysis_config = user_config.get(analysis_type, {})
            # Merge default with user-provided
            merged_config[analysis_type] = {**default_config, **user_analysis_config}
        
        return merged_config
    
    # Analysis implementation methods
    def _run_basic_analysis(self, session_data, config):
        """Run basic spike-train analysis"""
        time_window = config.get('time_window', (-0.5, 0.5))
        bin_size = config.get('bin_size', 0.01)
        
        # Select first neuron for demo
        example_spikes = session_data['spikes'][0] if len(session_data['spikes']) > 0 else []
        
        # Compute PSTH
        psth, time_bins = self.neural_analyzer.compute_psth(
            example_spikes,
            time_window=time_window,
            bin_size=bin_size
        )
        
        return {
            'psth': psth.tolist(),
            'time_bins': time_bins.tolist(),
            'neuron_id': 0,
            'time_window': time_window,
            'bin_size': bin_size
        }
    
    def _run_lfp_analysis(self, session_data, config):
        """Run LFP analysis"""
        freq_range = config.get('freq_range', (1, 100))
        freq_bands = config.get('freq_bands', {})
        
        # Compute LFP power spectrum for first channel
        freqs, power = self.neural_analyzer.compute_lfp_power(
            session_data['lfp'][:, 0] if 'lfp' in session_data and session_data['lfp'].shape[1] > 0 else [],
            freq_range=freq_range
        )
        
        # Compute band powers
        band_powers = {}
        for band_name, (band_min, band_max) in freq_bands.items():
            mask = (freqs >= band_min) & (freqs <= band_max)
            band_powers[band_name] = np.mean(power[mask]) if np.any(mask) else 0
        
        return {
            'freqs': freqs.tolist(),
            'power': power.tolist(),
            'band_powers': band_powers,
            'channel_id': 0,
            'freq_range': freq_range
        }
    
    def _run_population_analysis(self, session_data, config):
        """Run population analysis"""
        n_components = config.get('n_components', 10)
        scale_data = config.get('scale_data', True)
        
        # Prepare population data
        time_bins = np.arange(-0.5, 0.5, 0.01)
        firing_rates = self.data_controller.compute_firing_rates(
            session_data['spikes'], 
            time_bins
        )
        
        # Run PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        
        # Reshape data for PCA (neurons x timepoints)
        n_neurons, n_trials, n_timepoints = firing_rates.shape
        reshaped_data = firing_rates.reshape(n_neurons, -1)
        
        # Scale data if requested
        if scale_data:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            reshaped_data = scaler.fit_transform(reshaped_data)
        
        # Fit PCA
        pca_result = pca.fit_transform(reshaped_data)
        
        return {
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'n_components': n_components,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'scale_data': scale_data
        }
    
    def _run_behavior_analysis(self, session_data, config):
        """Run behavioral analysis"""
        n_lags = config.get('n_lags', 5)
        
        # Generate example behavioral data
        n_trials = len(session_data['spikes'][0]) if session_data.get('spikes') and len(session_data['spikes']) > 0 else 100
        example_choices = np.random.randint(0, 2, n_trials)
        example_outcomes = np.random.randint(0, 2, n_trials)
        
        # Analyze sequential effects
        seq_effects = self.behavior_analyzer.compute_sequential_effects(
            example_choices, 
            example_outcomes
        )
        
        return {
            'sequential_effects': {key: float(value) for key, value in seq_effects.items()},
            'n_trials': n_trials,
            'choice_bias': float(np.mean(example_choices)),
            'outcome_rate': float(np.mean(example_outcomes))
        }
    
    def _run_cross_regional_analysis(self, session_data, config):
        """Run cross-regional analysis"""
        method = config.get('method', 'coherence')
        
        if method == 'coherence' and 'lfp' in session_data:
            from scipy import signal
            
            # Use first two channels for demo
            if session_data['lfp'].shape[1] >= 2:
                f, Cxy = signal.coherence(
                    session_data['lfp'][:, 0],
                    session_data['lfp'][:, 1],
                    fs=100
                )
                
                return {
                    'freqs': f.tolist(),
                    'coherence': Cxy.tolist(),
                    'channel1': 0,
                    'channel2': 1,
                    'method': method
                }
        
        # Default empty result
        return {
            'freqs': [],
            'coherence': [],
            'method': method,
            'error': 'Insufficient data for cross-regional analysis'
        }

    def _run_choice_aligned_analysis(self, session_data, config):
        """Run choice-aligned activity analysis.
        Roadmap 3.3:
        - Data: dat['spks'], dat['response']
        - Method: Separate trials by choice, calc avg activity, stat tests, FDR.
        - Visualization: Activity diff, heatmap of selectivity.
        - Metrics: Selectivity index, latency, duration.
        """
        logger.info(f"Running choice-aligned analysis with config: {config}")

        spikes = session_data.get('spikes') # Expected shape: (n_neurons, n_trials, n_timebins)
        response = session_data.get('response') # Expected shape: (n_trials,)
        
        if spikes is None or response is None:
            logger.error("Choice-aligned: Missing spikes or response data.")
            return {'error': 'Missing spikes or response data for choice-aligned analysis.'}

        if not isinstance(spikes, np.ndarray) or spikes.ndim != 3:
            logger.error(f"Choice-aligned: Spikes data has incorrect shape or type: {type(spikes)}, ndim: {getattr(spikes, 'ndim', 'N/A')}")
            return {'error': 'Spikes data has incorrect shape or type.'}
            
        if not isinstance(response, np.ndarray) or response.ndim != 1:
            logger.error(f"Choice-aligned: Response data has incorrect shape or type: {type(response)}, ndim: {getattr(response, 'ndim', 'N/A')}")
            return {'error': 'Response data has incorrect shape or type.'}

        n_neurons, n_trials, n_timebins = spikes.shape

        if n_trials != len(response):
            logger.error(f"Choice-aligned: Mismatch between number of trials in spikes ({n_trials}) and response ({len(response)}).")
            return {'error': 'Mismatch in trial numbers between spikes and response.'}

        # Roadmap: Separate trials by choice direction (left vs. right)
        # Response values: -1 (right), 0 (no-go), 1 (left) based on NMA notebook
        
        # For this example, let's define left_choice_trials and right_choice_trials
        # This might need adjustment based on actual response values.
        # Assuming response value 1 for left, -1 for right.
        left_choice_mask = (response == 1)
        right_choice_mask = (response == -1)

        if not np.any(left_choice_mask) or not np.any(right_choice_mask):
            logger.warning("Choice-aligned: Not enough trials for both left and right choices.")
            return {'error': 'Insufficient trials for at least one choice direction.'}

        spikes_left_choice = spikes[:, left_choice_mask, :]    # (n_neurons, n_left_trials, n_timebins)
        spikes_right_choice = spikes[:, right_choice_mask, :] # (n_neurons, n_right_trials, n_timebins)

        # Calculate average activity profiles for each choice
        avg_activity_left = np.mean(spikes_left_choice, axis=1)   # (n_neurons, n_timebins)
        avg_activity_right = np.mean(spikes_right_choice, axis=1) # (n_neurons, n_timebins)
        
        # Activity difference
        activity_diff = avg_activity_left - avg_activity_right # (n_neurons, n_timebins)

        # Placeholder for statistical tests, FDR, selectivity index, latency, duration
        # These would involve more complex calculations, potentially using self.neural_analyzer

        # For now, return the processed data that can be visualized
        # Time bins information is also needed for plotting
        # Assuming bin_size and stim_onset are available as in basic analysis, or fixed
        # From NMA notebook: dt = dat['bin_size'] (0.01s), stim_onset = T0 (0.5s)
        # Number of timebins is spikes.shape[2]. Total duration = n_timebins * dt
        # Time relative to stimulus onset: np.arange(n_timebins) * dt - stim_onset
        
        bin_size = session_data.get('bin_size', 0.01) # Default 10ms
        stim_onset_time = session_data.get('stim_onset', 0.5) # Default 500ms
        
        # Create a time axis: from -stim_onset_time to (n_timebins*bin_size - stim_onset_time)
        # Example: if n_timebins = 250, bin_size = 0.01, stim_onset_time = 0.5
        # Bins are 0 to 249. Times: 0*0.01-0.5 = -0.5;  249*0.01-0.5 = 2.49-0.5 = 1.99
        time_axis = (np.arange(n_timebins) * bin_size) - stim_onset_time


        logger.info(f"Choice-aligned analysis completed for {n_neurons} neurons.")
        
        return {
            'avg_activity_left': avg_activity_left.tolist(),
            'avg_activity_right': avg_activity_right.tolist(),
            'activity_difference': activity_diff.tolist(),
            'time_axis': time_axis.tolist(),
            'n_left_trials': int(np.sum(left_choice_mask)),
            'n_right_trials': int(np.sum(right_choice_mask)),
            'n_neurons': n_neurons,
            # 'choice_selectivity_index': [], # Placeholder
            # 'latency_to_encoding': [], # Placeholder
            # 'duration_of_encoding': [] # Placeholder
        } 