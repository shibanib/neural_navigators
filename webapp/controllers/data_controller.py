import sys
sys.path.append('../../src')

import os
import numpy as np
from data_loader import SteinmetzDataLoader

class DataController:
    """Controller for managing data loading and retrieval"""
    
    def __init__(self):
        self.data_loader = SteinmetzDataLoader()
        self.cached_sessions = {}
    
    def get_available_sessions(self):
        """Get a list of available session indices"""
        # This would typically come from scanning the data directory
        # For example purposes, return a fixed list
        return list(range(1, 20))  # Sessions 1-19
    
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