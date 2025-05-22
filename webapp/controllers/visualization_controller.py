import sys
import os

# Robustly add src directory to sys.path
_CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_CONTROLLER_DIR, '..', '..', 'src'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import json
import logging  # Add logging import

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger('visualization_controller')

class VisualizationController:
    """Controller for generating visualizations from analysis results"""
    
    def __init__(self):
        # Set plotting style
        logger.info("Initializing VisualizationController")
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("talk")
        
        # Register available visualization types
        self.visualization_types = {
            'psth': self._generate_psth_plot,
            'lfp_power': self._generate_lfp_power_plot,
            'pca_variance': self._generate_pca_variance_plot,
            'coherence': self._generate_coherence_plot,
            'summary': self._generate_summary_dashboard,
            'comparison': self._generate_comparison_plot,
            'choice_activity': self._generate_choice_activity_plot
        }
        logger.info(f"Registered visualization types: {list(self.visualization_types.keys())}")
    
    def generate_visualization(self, viz_type, results):
        """Generate visualization based on type and results"""
        logger.info(f"Generating visualization of type: {viz_type}")
        logger.info(f"Results keys: {list(results.keys())}")
        
        if viz_type in self.visualization_types:
            viz_fn = self.visualization_types[viz_type]
            try:
                result = viz_fn(results)
                if 'plot' in result:
                    logger.info(f"Successfully generated plot of type {viz_type} with size: {len(result['plot'])} bytes")
                else:
                    logger.error(f"Visualization for {viz_type} did not return a plot. Got: {list(result.keys())}")
                return result
            except Exception as e:
                logger.error(f"Error generating visualization for {viz_type}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return {'error': f'Error generating visualization: {str(e)}'}
        else:
            return {
                'error': f'Visualization type {viz_type} not available',
                'available_types': list(self.visualization_types.keys())
            }
    
    def _figure_to_base64(self, fig):
        """Convert matplotlib figure to base64 encoded string for HTML embedding"""
        logger.info("Converting figure to base64")
        try:
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight')
            img_buf.seek(0)
            img_data = base64.b64encode(img_buf.getvalue()).decode('utf-8')
            plt.close(fig)
            logger.info(f"Successfully converted figure to base64 string of size: {len(img_data)} bytes")
            return img_data
        except Exception as e:
            logger.error(f"Error converting figure to base64: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _generate_psth_plot(self, results):
        """Generate PSTH plot from basic analysis results"""
        logger.info("Generating PSTH plot")
        if 'basic' not in results:
            logger.warning("No basic analysis results found")
            return {'error': 'No basic analysis results found'}
        
        basic_results = results['basic']
        logger.info(f"Basic results keys: {list(basic_results.keys())}")
        
        # Extract data
        psth = np.array(basic_results['psth'])
        time_bins = np.array(basic_results['time_bins'])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_bins, psth)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time from stimulus onset (s)')
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_title('Peri-Stimulus Time Histogram')
        ax.grid(True)
        
        # Convert to base64
        img_data = self._figure_to_base64(fig)
        
        return {
            'plot': img_data,
            'plot_type': 'psth',
            'neuron_id': basic_results.get('neuron_id', 0)
        }
    
    def _generate_lfp_power_plot(self, results):
        """Generate LFP power spectrum plot from LFP analysis results"""
        if 'lfp' not in results:
            return {'error': 'No LFP analysis results found'}
        
        lfp_results = results['lfp']
        
        # Extract data
        freqs = np.array(lfp_results['freqs'])
        power = np.array(lfp_results['power'])
        band_powers = lfp_results.get('band_powers', {})
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(freqs, power)
        
        # Add frequency bands if available
        colors = ['r', 'g', 'b', 'c', 'm']
        for i, (band_name, band_power) in enumerate(band_powers.items()):
            color = colors[i % len(colors)]
            ax.axvspan(*lfp_results['freq_bands'][band_name], alpha=0.2, color=color, label=f'{band_name}')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title('LFP Power Spectrum')
        ax.grid(True)
        if band_powers:
            ax.legend()
        
        # Convert to base64
        img_data = self._figure_to_base64(fig)
        
        return {
            'plot': img_data,
            'plot_type': 'lfp_power',
            'channel_id': lfp_results.get('channel_id', 0)
        }
    
    def _generate_pca_variance_plot(self, results):
        """Generate PCA explained variance plot from population analysis results"""
        if 'population' not in results:
            return {'error': 'No population analysis results found'}
        
        pop_results = results['population']
        
        # Extract data
        explained_var = np.array(pop_results['explained_variance'])
        cumulative_var = np.array(pop_results['cumulative_variance'])
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Individual variance
        components = np.arange(1, len(explained_var) + 1)
        ax1.bar(components, explained_var)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('PCA Explained Variance')
        
        # Cumulative variance
        ax2.plot(components, cumulative_var, 'o-')
        ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% Threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        # Convert to base64
        img_data = self._figure_to_base64(fig)
        
        return {
            'plot': img_data,
            'plot_type': 'pca_variance',
            'n_components': pop_results.get('n_components', 10)
        }
    
    def _generate_coherence_plot(self, results):
        """Generate coherence plot from cross-regional analysis results"""
        if 'cross_regional' not in results:
            return {'error': 'No cross-regional analysis results found'}
        
        cross_results = results['cross_regional']
        
        # Check for error
        if 'error' in cross_results:
            return {'error': cross_results['error']}
        
        # Extract data
        freqs = np.array(cross_results['freqs'])
        coherence = np.array(cross_results['coherence'])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(freqs, coherence)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Coherence')
        ax.set_title(f'LFP Coherence between Channels {cross_results.get("channel1", 0)} and {cross_results.get("channel2", 1)}')
        ax.grid(True)
        
        # Convert to base64
        img_data = self._figure_to_base64(fig)
        
        return {
            'plot': img_data,
            'plot_type': 'coherence',
            'channel1': cross_results.get('channel1', 0),
            'channel2': cross_results.get('channel2', 1)
        }
    
    def _generate_summary_dashboard(self, results):
        """Generate a summary dashboard with multiple plots"""
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Define subplot grid
        gs = fig.add_gridspec(2, 2)
        
        # Track which plots were added
        added_plots = []
        
        # Add PSTH if available
        if 'basic' in results:
            ax1 = fig.add_subplot(gs[0, 0])
            basic_results = results['basic']
            psth = np.array(basic_results['psth'])
            time_bins = np.array(basic_results['time_bins'])
            ax1.plot(time_bins, psth)
            ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Firing rate (Hz)')
            ax1.set_title('PSTH')
            added_plots.append('psth')
        
        # Add LFP power if available
        if 'lfp' in results:
            ax2 = fig.add_subplot(gs[0, 1])
            lfp_results = results['lfp']
            freqs = np.array(lfp_results['freqs'])
            power = np.array(lfp_results['power'])
            ax2.semilogy(freqs, power)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Power')
            ax2.set_title('LFP Power')
            added_plots.append('lfp_power')
        
        # Add PCA variance if available
        if 'population' in results:
            ax3 = fig.add_subplot(gs[1, 0])
            pop_results = results['population']
            cumulative_var = np.array(pop_results['cumulative_variance'])
            components = np.arange(1, len(cumulative_var) + 1)
            ax3.plot(components, cumulative_var, 'o-')
            ax3.axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Components')
            ax3.set_ylabel('Cum. Variance')
            ax3.set_title('PCA')
            added_plots.append('pca_variance')
        
        # Add Coherence if available
        if 'cross_regional' in results and 'error' not in results['cross_regional']:
            ax4 = fig.add_subplot(gs[1, 1])
            cross_results = results['cross_regional']
            freqs = np.array(cross_results['freqs'])
            coherence = np.array(cross_results['coherence'])
            ax4.plot(freqs, coherence)
            ax4.set_xlabel('Frequency (Hz)')
            ax4.set_ylabel('Coherence')
            ax4.set_title('LFP Coherence')
            added_plots.append('coherence')
        
        plt.tight_layout()
        
        # Convert to base64
        img_data = self._figure_to_base64(fig)
        
        return {
            'plot': img_data,
            'plot_type': 'summary_dashboard',
            'included_plots': added_plots
        }
    
    def _generate_comparison_plot(self, results):
        """Generate comparison plots across multiple sessions"""
        if not isinstance(results, dict) or not results:
            return {'error': 'No valid results for comparison'}
        
        # Determine which analyses are available across sessions
        available_analyses = set()
        for session_results in results.values():
            available_analyses.update(session_results.keys())
        available_analyses.discard('error')
        
        # Create comparison plots
        comparison_plots = {}
        
        # Example: Compare PCA explained variance
        if 'population' in available_analyses:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for session_idx, session_results in results.items():
                if 'population' in session_results:
                    explained_var = np.array(session_results['population']['cumulative_variance'])
                    components = np.arange(1, len(explained_var) + 1)
                    ax.plot(components, explained_var, 'o-', label=f'Session {session_idx}')
            
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Cumulative Explained Variance')
            ax.set_title('PCA Variance Across Sessions')
            ax.grid(True)
            ax.legend()
            
            # Convert to base64
            comparison_plots['pca_comparison'] = self._figure_to_base64(fig)
        
        # Example: Compare PSTH across sessions
        if 'basic' in available_analyses:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for session_idx, session_results in results.items():
                if 'basic' in session_results:
                    psth = np.array(session_results['basic']['psth'])
                    time_bins = np.array(session_results['basic']['time_bins'])
                    ax.plot(time_bins, psth, label=f'Session {session_idx}')
            
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time from stimulus onset (s)')
            ax.set_ylabel('Firing rate (Hz)')
            ax.set_title('PSTH Comparison Across Sessions')
            ax.grid(True)
            ax.legend()
            
            # Convert to base64
            comparison_plots['psth_comparison'] = self._figure_to_base64(fig)
        
        return {
            'plots': comparison_plots,
            'plot_type': 'comparison',
            'available_analyses': list(available_analyses)
        }

    def _generate_choice_activity_plot(self, results):
        """Generate plot for choice-aligned activity analysis results."""
        logger.info("Generating choice activity plot")
        # The 'results' here is expected to be the direct output of 
        # AnalysisController._run_choice_aligned_analysis for a single session.
        
        if 'choice_aligned' not in results:
            logger.warning("No choice_aligned analysis results found in the input for visualization.")
            # This check might be tricky if results is directly the output of choice_aligned analysis
            # Let's assume `results` *is* the choice_aligned_results dictionary itself.
            # So, we check for its specific keys directly.
            pass # Re-evaluating structure below

        choice_results = results.get('choice_aligned')
        if not choice_results: # If called with results={session_id: {analysis_name: data}}
             # Or if called with results={analysis_name: data} and analysis_name is not 'choice_aligned'
            if isinstance(results, dict) and 'avg_activity_left' in results: # results is already choice_results
                choice_results = results
            else:
                logger.warning("No choice_aligned analysis results found under 'choice_aligned' key or direct result.")
                return {'error': 'No choice_aligned analysis results found for visualization.'}

        logger.info(f"Choice-aligned results keys for viz: {list(choice_results.keys())}")

        # Extract data
        try:
            avg_activity_left_all_neurons = np.array(choice_results['avg_activity_left']) # (n_neurons, n_timebins)
            avg_activity_right_all_neurons = np.array(choice_results['avg_activity_right'])
            activity_diff_all_neurons = np.array(choice_results['activity_difference'])
            time_axis = np.array(choice_results['time_axis'])
            n_left_trials = choice_results.get('n_left_trials', 'N/A')
            n_right_trials = choice_results.get('n_right_trials', 'N/A')
        except KeyError as e:
            logger.error(f"Missing key in choice_aligned results: {e}")
            return {'error': f'Missing key {e} in choice_aligned results for visualization.'}

        if not avg_activity_left_all_neurons.size or not time_axis.size:
            logger.warning("Choice activity plot: No data to plot (empty arrays).")
            return {'error': 'No data available to plot for choice activity.'}

        # For a summary plot, average across neurons
        avg_left_across_neurons = np.mean(avg_activity_left_all_neurons, axis=0)
        avg_right_across_neurons = np.mean(avg_activity_right_all_neurons, axis=0)
        avg_diff_across_neurons = np.mean(activity_diff_all_neurons, axis=0)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(time_axis, avg_left_across_neurons, label=f'Avg Left Choice (N={n_left_trials} trials)', color='blue')
        ax.plot(time_axis, avg_right_across_neurons, label=f'Avg Right Choice (N={n_right_trials} trials)', color='red')
        ax.plot(time_axis, avg_diff_across_neurons, label='Difference (Left - Right)', color='green', linestyle='--')
        
        ax.axvline(x=0, color='k', linestyle=':', alpha=0.7, label='Stimulus Onset')
        ax.set_xlabel('Time from stimulus onset (s)')
        ax.set_ylabel('Average Firing Rate (population mean, Hz or counts/bin)') # Label needs to be accurate based on input
        ax.set_title('Population Average Activity Aligned to Choice')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        # Convert to base64
        img_data = self._figure_to_base64(fig)
        
        return {
            'plot': img_data,
            'plot_type': 'choice_activity',
            'description': 'Average neural activity for left and right choices, and their difference, across all neurons.'
        } 