import sys
import os
import warnings # Import the standard warnings module

# Get the directory of the current file (data_controller.py)
_CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the 'src' directory
_SRC_DIR = os.path.abspath(os.path.join(_CONTROLLER_DIR, '..', '..', 'src'))

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR) # Insert at the beginning

import numpy as np
from data_loader import SteinmetzDataLoader
from typing import Dict, Any

class DataController:
    """Controller for managing data loading and retrieval"""
    
    def __init__(self):
        # data_dir is relative to src/data_loader.py
        # SteinmetzDataLoader now resolves this robustly
        self.data_loader = SteinmetzDataLoader(data_dir='../notebooks/data')
        self.cached_sessions = {}
    
    def get_available_sessions(self):
        """Get a list of available 0-indexed session indices"""
        # The SteinmetzDataLoader now determines session counts from part files
        # part_session_counts = [13, 13, 13] -> 39 sessions (0-38)
        # This info is in SteinmetzDataLoader. We need to access it or hardcode total here.
        # For now, let's hardcode based on the known structure of part files.
        # A better way would be for DataController to ask SteinmetzDataLoader.
        num_sessions = 39 # sum([13,13,13])
        return list(range(num_sessions)) # Returns 0 to 38
    
    def load_session(self, session_idx):
        """Load a specific session and cache it"""
        if session_idx in self.cached_sessions:
            return self.cached_sessions[session_idx]
        
        # Load the session data
        session_data = self.data_loader.load_session(session_idx)
        
        # Cache the data for future use
        self.cached_sessions[session_idx] = session_data
        
        return session_data
    
    def get_session_info(self, session_idx):
        """Get basic information about a session"""
        session_data = self.load_session(session_idx)
        
        # Extract basic information
        info = {
            'session_idx': session_idx,
            'n_neurons': len(session_data.get('spikes', [])),
            'n_trials': len(session_data.get('spikes', [[]])[0]) if session_data.get('spikes') else 0,
            'brain_regions': session_data.get('brain_regions', []),
            'duration': session_data.get('duration', 0)
        }
        
        return info
    
    def compute_firing_rates(self, spikes, time_bins):
        """Compute firing rates from spike data"""
        return self.data_loader.compute_firing_rates(spikes, time_bins)
    
    def clear_cache(self):
        """Clear the session cache to free memory"""
        self.cached_sessions = {}

    def get_session_data_summary(self, session_idx: int) -> Dict[str, Any]:
        """Load a session and return a summary of its data structure."""
        try:
            session_data = self.load_session(session_idx)
        except Exception as e:
            return {"error": f"Failed to load session {session_idx}: {str(e)}"}

        summary = {}
        # --- DEBUG: Process a limited number of keys ---
        keys_to_process_count = 5 # Start with 5 keys
        processed_count = 0
        # --- END DEBUG ---

        for key, value in session_data.items():
            # --- DEBUG: Limit processing --- 
            if processed_count >= keys_to_process_count:
                break 
            # --- END DEBUG ---

            item_summary = {'type': str(type(value))}
            if isinstance(value, np.ndarray):
                item_summary['shape'] = value.shape
                item_summary['dtype'] = str(value.dtype)
                if value.size > 0 and np.issubdtype(value.dtype, np.number):
                    # Limit stats calculation for very large arrays to avoid performance issues
                    # if value.ndim > 2 or value.size > 100000: # Example threshold
                    #    item_summary['stats_info'] = 'Stats omitted for large/high-dim array'
                    # else:
                    if value.size < 200000 or value.ndim <=2: # Looser constraint for now
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
                            warnings.filterwarnings('ignore', r'Mean of empty slice')
                            warnings.filterwarnings('ignore', r'invalid value encountered in subtract') # For np.std
                            try:
                                item_summary['min'] = float(np.nanmin(value))
                                item_summary['max'] = float(np.nanmax(value))
                                item_summary['mean'] = float(np.nanmean(value))
                                item_summary['std'] = float(np.nanstd(value))
                            except TypeError: 
                                item_summary['stats_error'] = 'Could not compute stats (possibly non-numeric data)'
                    item_summary['has_nan'] = bool(np.isnan(value).any())
                
                # Preview logic (no change, keep it concise)
                if value.size > 0 and value.size < 20:
                    item_summary['preview'] = value.tolist()
                elif value.size > 0:
                    item_summary['preview (first 5)'] = value.flat[:5].tolist()

            elif isinstance(value, list):
                item_summary['length'] = len(value)
                if value:
                    item_summary['first_element_type'] = str(type(value[0]))
                    preview_list = []
                    for i, item in enumerate(value[:5]): # Preview first 5 items
                        if isinstance(item, np.ndarray):
                            preview_list.append(f"ndarray(shape={item.shape}, dtype={item.dtype})")
                        else:
                            preview_list.append(str(item))
                    item_summary['preview (first 5)'] = preview_list 
            elif isinstance(value, dict):
                item_summary['keys'] = list(value.keys())
                item_summary['num_keys'] = len(value.keys())
            else: 
                item_summary['value'] = value # For scalars, keep original value
            summary[key] = item_summary
            processed_count += 1 # Increment processed count
        
        return summary 