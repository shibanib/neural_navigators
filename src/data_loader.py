import numpy as np
import os
import requests
from typing import List, Dict, Any, Tuple

class SteinmetzDataLoader:
    """Class to handle loading and basic preprocessing of Steinmetz dataset."""
    
    def __init__(self, data_dir: str = '../data'):
        self.data_dir = data_dir
        self.file_urls = {
            'steinmetz_st.npz': 'https://osf.io/4bjns/download',
            'steinmetz_wav.npz': 'https://osf.io/ugm9v/download',
            'steinmetz_lfp.npz': 'https://osf.io/kx3v9/download'
        }
        
        # Brain region groupings
        self.regions = ["vis ctx", "thal", "hipp", "other ctx", "midbrain", 
                       "basal ganglia", "cortical subplate", "other"]
        self.brain_groups = {
            "visual cortex": ["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"],
            "thalamus": ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", 
                        "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"],
            "hippocampus": ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"],
            "non_visual_cortex": ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", 
                                "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP", "TT"]
        }

    def download_data(self) -> None:
        """Download the dataset if not already present."""
        os.makedirs(self.data_dir, exist_ok=True)
        
        for fname, url in self.file_urls.items():
            fpath = os.path.join(self.data_dir, fname)
            if not os.path.isfile(fpath):
                print(f"Downloading {fname}...")
                try:
                    r = requests.get(url)
                    if r.status_code == requests.codes.ok:
                        with open(fpath, "wb") as f:
                            f.write(r.content)
                        print(f"Successfully downloaded {fname}")
                    else:
                        print(f"Failed to download {fname}")
                except requests.ConnectionError:
                    print(f"Connection error while downloading {fname}")

    def load_session(self, session_idx: int) -> Dict[str, Any]:
        """
        Load data for a specific session.
        
        Args:
            session_idx: Index of the session to load
            
        Returns:
            Dictionary containing the session data
        """
        # Load all data types
        dat_LFP = np.load(os.path.join(self.data_dir, 'steinmetz_lfp.npz'), 
                         allow_pickle=True)['dat'][session_idx]
        dat_WAV = np.load(os.path.join(self.data_dir, 'steinmetz_wav.npz'), 
                         allow_pickle=True)['dat'][session_idx]
        dat_ST = np.load(os.path.join(self.data_dir, 'steinmetz_st.npz'), 
                         allow_pickle=True)['dat'][session_idx]
        
        # Combine into single dictionary
        session_data = {
            'lfp': dat_LFP['lfp'],
            'lfp_passive': dat_LFP['lfp_passive'],
            'brain_area_lfp': dat_LFP['brain_area_lfp'],
            'waveform_w': dat_WAV['waveform_w'],
            'waveform_u': dat_WAV['waveform_u'],
            'trough_to_peak': dat_WAV['trough_to_peak'],
            'spikes': dat_ST['ss'],
            'spikes_passive': dat_ST['ss_passive']
        }
        
        return session_data

    def get_fast_spiking_neurons(self, session_data: Dict[str, Any]) -> np.ndarray:
        """
        Identify putative fast-spiking neurons (width <= 10 samples).
        
        Args:
            session_data: Dictionary containing session data
            
        Returns:
            Boolean array indicating which neurons are fast-spiking
        """
        return session_data['trough_to_peak'] <= 10

    def compute_firing_rates(self, spikes: List[List[float]], 
                           time_bins: np.ndarray) -> np.ndarray:
        """
        Compute firing rates from spike times.
        
        Args:
            spikes: List of spike times for each neuron
            time_bins: Array of time bin edges
            
        Returns:
            Array of firing rates for each neuron in each time bin
        """
        n_neurons = len(spikes)
        n_bins = len(time_bins) - 1
        rates = np.zeros((n_neurons, n_bins))
        
        for i, neuron_spikes in enumerate(spikes):
            rates[i], _ = np.histogram(neuron_spikes, bins=time_bins)
            
        # Convert to Hz
        bin_width = time_bins[1] - time_bins[0]
        rates = rates / bin_width
        
        return rates 