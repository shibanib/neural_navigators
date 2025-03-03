import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Any
from sklearn.decomposition import PCA

class NeuralAnalyzer:
    """Class for analyzing neural data from the Steinmetz dataset."""
    
    def __init__(self):
        self.sampling_rate = 100  # 10ms bins = 100 Hz

    def compute_psth(self, spikes: List[List[float]], 
                    time_window: Tuple[float, float],
                    bin_size: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Peri-Stimulus Time Histogram (PSTH).
        
        Args:
            spikes: List of spike times for each trial
            time_window: (start_time, end_time) relative to stimulus
            bin_size: Size of time bins in seconds
            
        Returns:
            tuple: (psth, time_bins)
        """
        time_bins = np.arange(time_window[0], time_window[1] + bin_size, bin_size)
        n_bins = len(time_bins) - 1
        n_trials = len(spikes)
        
        # Compute histogram for each trial
        trial_histograms = np.zeros((n_trials, n_bins))
        for i, trial_spikes in enumerate(spikes):
            trial_histograms[i], _ = np.histogram(trial_spikes, bins=time_bins)
        
        # Average across trials and convert to firing rate
        psth = np.mean(trial_histograms, axis=0) / bin_size
        
        return psth, time_bins[:-1] + bin_size/2

    def compute_lfp_power(self, lfp_data: np.ndarray, 
                         freq_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute LFP power spectrum using Welch's method.
        
        Args:
            lfp_data: LFP time series
            freq_range: (min_freq, max_freq) to analyze
            
        Returns:
            tuple: (frequencies, power_spectrum)
        """
        frequencies, power_spectrum = signal.welch(lfp_data, 
                                                 fs=self.sampling_rate,
                                                 nperseg=256)
        
        # Filter to desired frequency range
        mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        return frequencies[mask], power_spectrum[mask]

    def compute_spike_triggered_lfp(self, spikes: List[float], 
                                  lfp: np.ndarray,
                                  window: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spike-triggered average of LFP.
        
        Args:
            spikes: Spike times
            lfp: LFP time series
            window: (time_before, time_after) spike in seconds
            
        Returns:
            tuple: (time_points, average_lfp)
        """
        window_samples = (int(window[0] * self.sampling_rate),
                        int(window[1] * self.sampling_rate))
        time_points = np.arange(window_samples[0], window_samples[1]) / self.sampling_rate
        
        # Convert spike times to samples
        spike_samples = (np.array(spikes) * self.sampling_rate).astype(int)
        
        # Extract LFP segments around each spike
        segments = []
        for spike in spike_samples:
            if (spike + window_samples[1] < len(lfp) and 
                spike + window_samples[0] >= 0):
                segment = lfp[spike + window_samples[0]:spike + window_samples[1]]
                segments.append(segment)
        
        # Average across segments
        if segments:
            average_lfp = np.mean(segments, axis=0)
        else:
            average_lfp = np.zeros_like(time_points)
            
        return time_points, average_lfp

    def compute_population_dynamics(self, firing_rates: np.ndarray, 
                                 n_components: int = 3) -> Tuple[np.ndarray, PCA]:
        """
        Compute low-dimensional population dynamics using PCA.
        
        Args:
            firing_rates: Array of shape (n_neurons, n_timepoints)
            n_components: Number of PCA components to compute
            
        Returns:
            tuple: (projected_data, pca_object)
        """
        pca = PCA(n_components=n_components)
        projected_data = pca.fit_transform(firing_rates.T).T
        return projected_data, pca

    def compute_cross_correlation(self, spikes1: List[float], 
                                spikes2: List[float],
                                max_lag: float = 0.1,
                                bin_size: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cross-correlation between two spike trains.
        
        Args:
            spikes1: First spike train
            spikes2: Second spike train
            max_lag: Maximum lag time in seconds
            bin_size: Size of time bins in seconds
            
        Returns:
            tuple: (lags, correlation)
        """
        # Create time bins
        lags = np.arange(-max_lag, max_lag + bin_size, bin_size)
        correlation = np.zeros(len(lags) - 1)
        
        # Convert to histograms
        hist1, _ = np.histogram(spikes1, bins=np.arange(min(spikes1), max(spikes1) + bin_size, bin_size))
        hist2, _ = np.histogram(spikes2, bins=np.arange(min(spikes2), max(spikes2) + bin_size, bin_size))
        
        # Compute cross-correlation
        correlation = np.correlate(hist1, hist2, mode='full')
        
        # Center and normalize
        center = len(correlation) // 2
        lags = lags[:-1]
        correlation = correlation[center - len(lags)//2:center + len(lags)//2]
        correlation = correlation / (np.sqrt(np.sum(hist1**2) * np.sum(hist2**2)))
        
        return lags, correlation 