import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.gridspec import GridSpec

class NeuralViz:
    """Class for creating standardized visualizations of neural data."""
    
    def __init__(self, style='seaborn-v0_8-whitegrid'):
        """Initialize with a specific style."""
        plt.style.use(style)
        sns.set_context("talk")
        
    def plot_raster(self, spike_times: List[float], trials: List[List[float]], 
                    window: Tuple[float, float], title: str = None) -> plt.Figure:
        """
        Create a raster plot of spike times across trials.
        
        Args:
            spike_times: List of spike times
            trials: List of trial start times
            window: (start, end) time window relative to trial start
            title: Optional plot title
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, trial_start in enumerate(trials):
            trial_spikes = [s - trial_start for s in spike_times 
                          if trial_start + window[0] <= s <= trial_start + window[1]]
            ax.plot(trial_spikes, [i] * len(trial_spikes), '|', color='black')
            
        ax.set_xlabel('Time from trial start (s)')
        ax.set_ylabel('Trial number')
        if title:
            ax.set_title(title)
        
        return fig

    def plot_brain_regions_3d(self, coordinates: np.ndarray, 
                            regions: List[str], values: np.ndarray,
                            title: str = None) -> None:
        """
        Create interactive 3D visualization of brain regions.
        
        Args:
            coordinates: (n_points, 3) array of coordinates
            regions: List of region names
            values: Array of values to plot
            title: Optional plot title
        """
        fig = go.Figure(data=[go.Scatter3d(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=values,
                colorscale='Viridis',
                opacity=0.8
            ),
            text=regions
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='AP',
                yaxis_title='ML',
                zaxis_title='DV'
            )
        )
        
        fig.show()

    def create_summary_plot(self, data_dict: Dict[str, np.ndarray], 
                          plot_type: str = 'grid') -> plt.Figure:
        """
        Create a summary plot combining multiple data types.
        
        Args:
            data_dict: Dictionary of data arrays to plot
            plot_type: Type of summary plot ('grid' or 'dashboard')
        """
        if plot_type == 'grid':
            n_plots = len(data_dict)
            n_cols = min(3, n_plots)
            n_rows = (n_plots - 1) // n_cols + 1
            
            fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
            gs = GridSpec(n_rows, n_cols)
            
            for i, (name, data) in enumerate(data_dict.items()):
                row = i // n_cols
                col = i % n_cols
                ax = fig.add_subplot(gs[row, col])
                
                if data.ndim == 1:
                    ax.plot(data)
                else:
                    ax.imshow(data, aspect='auto')
                    
                ax.set_title(name)
                
        else:  # dashboard
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(2, 2)
            
            for i, (name, data) in enumerate(list(data_dict.items())[:4]):
                row = i // 2
                col = i % 2
                ax = fig.add_subplot(gs[row, col])
                
                if data.ndim == 1:
                    ax.plot(data)
                else:
                    ax.imshow(data, aspect='auto')
                    
                ax.set_title(name)
        
        plt.tight_layout()
        return fig

    def plot_population_activity(self, firing_rates: np.ndarray, 
                               time_bins: np.ndarray,
                               sort_by: str = None) -> plt.Figure:
        """
        Create a heatmap of population activity.
        
        Args:
            firing_rates: Array of shape (n_neurons, n_timepoints)
            time_bins: Array of time points
            sort_by: Method to sort neurons ('rate', 'latency', or None)
        """
        if sort_by == 'rate':
            sort_idx = np.argsort(np.mean(firing_rates, axis=1))
        elif sort_by == 'latency':
            latencies = np.argmax(firing_rates, axis=1)
            sort_idx = np.argsort(latencies)
        else:
            sort_idx = np.arange(firing_rates.shape[0])
            
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(firing_rates[sort_idx], aspect='auto', 
                      extent=[time_bins[0], time_bins[-1], 0, firing_rates.shape[0]])
        
        plt.colorbar(im, ax=ax, label='Firing rate (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron')
        
        return fig

    def plot_connectivity_matrix(self, matrix: np.ndarray, 
                               labels: List[str],
                               title: str = None) -> plt.Figure:
        """
        Create a connectivity matrix visualization.
        
        Args:
            matrix: Square matrix of connectivity values
            labels: List of region/neuron labels
            title: Optional plot title
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        im = sns.heatmap(matrix, xticklabels=labels, yticklabels=labels,
                        cmap='RdBu_r', center=0, ax=ax)
        
        if title:
            ax.set_title(title)
            
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        return fig

    def create_interactive_timeseries(self, time: np.ndarray, 
                                    data: np.ndarray,
                                    labels: List[str] = None) -> None:
        """
        Create an interactive time series plot.
        
        Args:
            time: Time points
            data: Data array of shape (n_signals, n_timepoints)
            labels: Optional list of signal labels
        """
        if labels is None:
            labels = [f'Signal {i+1}' for i in range(data.shape[0])]
            
        fig = go.Figure()
        
        for i in range(data.shape[0]):
            fig.add_trace(go.Scatter(
                x=time,
                y=data[i],
                name=labels[i],
                mode='lines'
            ))
            
        fig.update_layout(
            title='Interactive Time Series',
            xaxis_title='Time (s)',
            yaxis_title='Value',
            hovermode='x'
        )
        
        fig.show() 