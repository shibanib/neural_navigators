import numpy as np
import os
import requests
from typing import List, Dict, Any, Tuple

# Get the absolute path to the directory where this file (data_loader.py) is located
_DATA_LOADER_DIR = os.path.dirname(os.path.abspath(__file__))

class SteinmetzDataLoader:
    """Class to handle loading and basic preprocessing of Steinmetz dataset."""
    
    def __init__(self, data_dir: str = '../data'):
        # Resolve data_dir relative to the location of this data_loader.py file
        self.data_dir = os.path.abspath(os.path.join(_DATA_LOADER_DIR, data_dir))
        # Ensure the path uses OS-specific separators and is cleaned up
        self.data_dir = os.path.normpath(self.data_dir)

        # These URLs might be for the original raw data, not the part*.npz files
        self.file_urls = {
            'steinmetz_part0.npz': None, # Placeholder, assuming these files exist
            'steinmetz_part1.npz': None,
            'steinmetz_part2.npz': None,
            'steinmetz_lfp.npz': 'https://osf.io/kx3v9/download' # Keep for now, might be redundant
        }
        
        # Define how many sessions are in each part file
        # Based on steinmetz_NMA.ipynb: dat[:13], dat[13:26], dat[26:]
        # Assuming 0-indexed alldat length of 39 (0-38)
        # Part 0: sessions 0-12 (13 sessions)
        # Part 1: sessions 13-25 (13 sessions)
        # Part 2: sessions 26-38 (13 sessions)
        self.part_session_counts = [13, 13, 13] 
        self.part_file_names = ['steinmetz_part0.npz', 'steinmetz_part1.npz', 'steinmetz_part2.npz']
        
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
        # This method might need adjustment if part*.npz are not downloadable directly
        # For now, it will only attempt to download steinmetz_lfp.npz if its URL is set
        os.makedirs(self.data_dir, exist_ok=True)
        
        for fname, url in self.file_urls.items():
            if url is None: # Skip files we assume exist locally (part*.npz)
                continue
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
        Load data for a specific session from part*.npz files.
        Session indices are assumed to be 0-based (e.g., 0 to 38 for 39 sessions).
        """
        
        current_offset = 0
        target_part_idx = -1
        idx_in_part_file = -1

        for i, count in enumerate(self.part_session_counts):
            if session_idx < current_offset + count:
                target_part_idx = i
                idx_in_part_file = session_idx - current_offset
                break
            current_offset += count
            
        if target_part_idx == -1:
            raise ValueError(f"Session index {session_idx} is out of range.")

        part_file_name = self.part_file_names[target_part_idx]
        part_file_path = os.path.join(self.data_dir, part_file_name)

        if not os.path.exists(part_file_path):
            # Attempt to download if part files are missing (assuming URLs could be added)
            # For now, just raise an error as URLs are None
            self.download_data() # This will only try LFP for now
            if not os.path.exists(part_file_path):
                 raise FileNotFoundError(f"Data file {part_file_path} not found. "
                                       f"Please ensure it's in {self.data_dir} or can be downloaded.")

        loaded_part_data = np.load(part_file_path, allow_pickle=True)
        
        # The 'dat' key in part*.npz files contains a list/array of session dictionaries
        session_dict_array = loaded_part_data['dat'] 
        session_data = session_dict_array[idx_in_part_file]
        
        # The session_data dictionary from part*.npz should contain all necessary fields.
        # If LFP data is indeed separate in steinmetz_lfp.npz and also structured by session_idx:
        try:
            lfp_file_path = os.path.join(self.data_dir, 'steinmetz_lfp.npz')
            if os.path.exists(lfp_file_path):
                dat_LFP_all_sessions = np.load(lfp_file_path, allow_pickle=True)['dat']
                session_lfp_data = dat_LFP_all_sessions[session_idx]
                # Merge LFP-specific keys if they are not already in session_data or if they are preferred
                # Example: session_data['lfp_from_dedicated_file'] = session_lfp_data.get('lfp')
                # For now, assume part*.npz files are comprehensive as per NMA.ipynb structure
                # If NMA.ipynb showed LFP data being added from separate files *into* the structure that was saved in parts,
                # then the LFP data should already be in session_data.
                # Let's check if 'lfp' is already a key from the part file.
                if 'lfp' not in session_data:
                    print(f"DEBUG: 'lfp' key not found in session_data from {part_file_name}. Checking dedicated LFP file.")
                    session_data['lfp'] = session_lfp_data.get('lfp')
                    session_data['brain_area_lfp'] = session_lfp_data.get('brain_area_lfp')
                else:
                    print(f"DEBUG: 'lfp' key already present in session_data from {part_file_name}.")

            else:
                print(f"DEBUG: Dedicated LFP file {lfp_file_path} not found. Relying on LFP data within part file if present.")
        except Exception as e:
            print(f"DEBUG: Error loading or merging dedicated LFP data: {e}")


        # Standardize key names if necessary, e.g., ensure 'spikes' exists
        if 'spks' in session_data and 'spikes' not in session_data:
            session_data['spikes'] = session_data['spks']
        if 'ss' in session_data and 'spikes' not in session_data: # Older key name
             session_data['spikes'] = session_data['ss']

        # Ensure other required keys for your analyses are present or handled.
        # Example:
        # required_keys = ['response', 'response_time', 'contrast_left', 'contrast_right', 'brain_area']
        # for key in required_keys:
        #     if key not in session_data:
        #         print(f"Warning: Key '{key}' not found in loaded session data for session {session_idx}")
        #         session_data[key] = None # or some default
        
        return session_data

    def get_fast_spiking_neurons(self, session_data: Dict[str, Any]) -> np.ndarray:
        """
        Identify putative fast-spiking neurons (width <= 10 samples).
        Assumes 'trough_to_peak' key exists in session_data.
        """
        if 'trough_to_peak' in session_data:
            return session_data['trough_to_peak'] <= 10
        else:
            print("Warning: 'trough_to_peak' not found in session data. Cannot identify fast-spiking neurons.")
            return np.array([]) # Return empty array or handle as appropriate

    def compute_firing_rates(self, spikes: List[List[float]], 
                           time_bins: np.ndarray) -> np.ndarray:
        """
        Compute firing rates from spike times.
        
        Args:
            spikes: List of spike times for each neuron and trial
                   (structure: neurons x trials x spike_times)
            time_bins: Array of time bin edges
            
        Returns:
            Array of firing rates for each neuron in each time bin
        """
        # This implementation assumes spikes are binned counts or needs adjustment
        # The NMA notebook suggests 'spks' are already (neurons x trials x time_bins)
        # If 'spikes' from session_data is already binned counts (neurons x trials x time_bins):
        if spikes.ndim == 3: # neurons x trials x time_bins
            # Average across trials, then divide by bin width to get Hz
            # Or, if already in Hz, just average across trials.
            # The description in NMA for 'spks': "neurons by trials by time bins" suggests counts.
            # "Time bins for all measurements are 10ms"
            # dt = session_data.get('bin_size', 0.01) # Should be in session_data
            
            # This function might be overly complex if 'spks' is already what's needed.
            # The original compute_firing_rates seemed to take raw spike times.
            # Let's assume 'spikes' from session_data (derived from 'spks') is (neurons, trials, timebins_counts)
            
            # For now, let's assume the AnalysisController will handle how to get rates from this.
            # This method might be better placed in AnalysisController or redefined.
            # Or, it should expect raw spike times if that's what 'ss' (original key for spikes) was.
            
            # The 'spks' in NMA dat[idir]['spks'] = S[:, :ntrials, :]
            # S  = steinmetz_loader.psth(stimes, sclust, visual_times-T0, dT, dt)
            # So 'spks' is ALREADY a PSTH (counts per bin).
            # To get firing rate in Hz, divide by dt.
            # dt = session_data.get('bin_size', 0.01)
            # firing_rates_hz = spikes / dt
            # And often averaged over trials.
            
            # Let's return the raw 'spks' array (neurons x trials x time_bins_counts)
            # and let downstream functions decide on averaging or converting to Hz.
            # The DataController currently calls this.
            # Let's make this method pass-through if spikes are already binned.
            if spikes.ndim == 3: # Presuming it's (neurons, trials, bins)
                print("DEBUG: compute_firing_rates received already binned data (ndim=3). Returning as is or mean over trials.")
                # This is what AnalysisController's _run_population_analysis expects for its PCA
                # It reshapes (n_neurons, n_trials, n_timepoints)
                return spikes # Or np.mean(spikes, axis=1) if trial-averaged rates are needed by default
            else: # Original logic for raw spike times list
                n_neurons = len(spikes)
                n_bins = len(time_bins) - 1
                rates = np.zeros((n_neurons, n_bins))
                
                for i, neuron_spikes in enumerate(spikes):
                    all_spikes_flat = []
                    if isinstance(neuron_spikes, list) and all(isinstance(trial, list) for trial in neuron_spikes):
                        for trial_spikes in neuron_spikes:
                            all_spikes_flat.extend(trial_spikes)
                    else: # If it's already a flat list of spike times for the neuron
                        all_spikes_flat = neuron_spikes

                    if all_spikes_flat:
                        counts, _ = np.histogram(all_spikes_flat, bins=time_bins)
                        rates[i] = counts
                
                bin_width = time_bins[1] - time_bins[0]
                # Need to know number of trials if averaging
                # This part is tricky without knowing the exact structure of 'spikes' list
                # For now, assume counts are returned, and Hz conversion happens later.
                return rates / bin_width # Returns rates in Hz if neuron_spikes was all spikes from all trials


# Example usage (for testing within this file if run directly)
if __name__ == '__main__':
    # Configure data_dir to point to your local 'notebooks/data' relative to this 'src' dir
    loader = SteinmetzDataLoader(data_dir='../notebooks/data') 
    
    # Ensure data files (part0, part1, part2) are present there
    # loader.download_data() # Call this if you want to try downloading missing files (LFP only for now)

    try:
        print(f"Data directory resolved to: {loader.data_dir}")
        if not os.path.exists(loader.data_dir):
            print(f"Error: Data directory does not exist: {loader.data_dir}")
        else:
            print(f"Contents of data directory: {os.listdir(loader.data_dir)}")

        test_session_idx = 0 # Try loading the first session
        print(f"Attempting to load session {test_session_idx}...")
        session_info = loader.load_session(test_session_idx)
        
        print(f"Successfully loaded session {test_session_idx}.")
        print("Available keys in session_info:", list(session_info.keys()))
        
        # Example: Print shape of spikes data
        if 'spikes' in session_info:
            spikes_data = session_info['spikes']
            if hasattr(spikes_data, 'shape'):
                print("Shape of 'spikes' data:", spikes_data.shape)
            else:
                print("'spikes' data is not a numpy array or has no shape attribute.")
        else:
            print("'spikes' key not found in session_info.")

        # Example: Print first few brain areas if available
        if 'brain_area' in session_info:
            brain_areas = session_info['brain_area']
            print("Brain areas (first 5):", brain_areas[:5])
        else:
            print("'brain_area' key not found in session_info.")

    except FileNotFoundError as e:
        print(f"Error during test loading: {e}")
        print("Please ensure the Steinmetz part*.npz files are in the specified data directory.")
    except Exception as e:
        print(f"An unexpected error occurred during test loading: {e}")
        import traceback
        traceback.print_exc() 