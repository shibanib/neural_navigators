# Advanced Guide for Steinmetz Neural Analysis Pipeline

## Table of Contents
1. [Detailed Analysis Examples](#detailed-analysis-examples)
2. [Troubleshooting Guide](#troubleshooting-guide)
3. [Extending the Pipeline](#extending-the-pipeline)
4. [Advanced Features](#advanced-features)

## Detailed Analysis Examples

### Basic Neural Analysis
```python
from neural_analysis import NeuralAnalyzer
from data_loader import SteinmetzDataLoader

# Load data
loader = SteinmetzDataLoader()
session_data = loader.load_session(11)

# Initialize analyzer
analyzer = NeuralAnalyzer()

# Compute PSTH with different parameters
psth, time_bins = analyzer.compute_psth(
    spikes=session_data['spikes'][0],  # First neuron
    time_window=(-0.5, 1.0),          # From 500ms before to 1s after
    bin_size=0.01                     # 10ms bins
)

# Identify fast-spiking neurons
is_fast_spiking = loader.get_fast_spiking_neurons(session_data)
```

### Advanced LFP Analysis
```python
# Compute coherence between specific frequency bands
def compute_band_coherence(lfp1, lfp2, band_range=(4, 8), fs=100):
    """Compute coherence in specific frequency band (e.g., theta: 4-8 Hz)"""
    from scipy import signal
    
    f, Cxy = signal.coherence(lfp1, lfp2, fs=fs)
    band_mask = (f >= band_range[0]) & (f <= band_range[1])
    return np.mean(Cxy[band_mask])

# Example usage
theta_coherence = compute_band_coherence(
    session_data['lfp'][:, 0],  # First channel
    session_data['lfp'][:, 1],  # Second channel
    band_range=(4, 8)           # Theta band
)
```

### Custom Population Analysis
```python
# Define custom dimensionality reduction
def custom_dimensionality_reduction(firing_rates, method='nmf'):
    """Alternative dimensionality reduction methods"""
    from sklearn.decomposition import NMF, FastICA
    
    if method == 'nmf':
        model = NMF(n_components=3, random_state=42)
    elif method == 'ica':
        model = FastICA(n_components=3, random_state=42)
    
    return model.fit_transform(firing_rates.T).T
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **Data Loading Issues**
   ```
   Error: Failed to download data
   ```
   - Check internet connection
   - Verify the data URLs are still valid
   - Try downloading manually and placing in `data/` directory

2. **Memory Errors**
   ```
   MemoryError: Unable to allocate array
   ```
   - Reduce batch size in analysis
   - Process fewer neurons/channels at once
   - Use memory-efficient operations:
   ```python
   # Instead of this
   all_data = np.array([compute_something(x) for x in large_dataset])
   
   # Do this
   results = []
   for x in large_dataset:
       result = compute_something(x)
       process_and_save(result)
   ```

3. **Visualization Issues**
   - Matplotlib backend problems:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # For systems without display
   ```
   - Interactive plot not showing:
   ```python
   # Enable notebook integration
   import plotly.io as pio
   pio.renderers.default = 'notebook'
   ```

4. **Pipeline Errors**
   - Process hanging:
   ```python
   # Add timeout to pipeline
   pipeline.batch_process(
       session_indices=[11, 12],
       timeout=3600  # 1 hour timeout
   )
   ```

## Extending the Pipeline

### Adding New Analysis Methods

1. **Create New Analysis Module**
```python
# src/custom_analysis.py
class CustomAnalyzer:
    def __init__(self):
        self.params = {}
    
    def new_analysis_method(self, data, **kwargs):
        """
        Implement new analysis method.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Analysis results
        """
        # Implementation
        pass
```

2. **Add to Pipeline**
```python
# In src/pipeline.py
from custom_analysis import CustomAnalyzer

class AnalysisPipeline:
    def __init__(self):
        self.custom_analyzer = CustomAnalyzer()
    
    def _run_custom_analysis(self, session_data, output_dir):
        results = self.custom_analyzer.new_analysis_method(
            session_data,
            param1=value1
        )
        return results
```

### Adding New Visualizations

1. **Create Visualization Function**
```python
# In src/visualization.py
def plot_custom_analysis(self, data, **kwargs):
    """Create custom visualization."""
    fig, ax = plt.subplots()
    # Plotting code
    return fig
```

2. **Add to Pipeline Output**
```python
# In pipeline
fig = self.visualizer.plot_custom_analysis(results)
fig.savefig(os.path.join(output_dir, 'custom_analysis.png'))
```

## Advanced Features

### Custom Data Transformations
```python
def apply_custom_filter(lfp_data, freq_range):
    """Apply custom filtering to LFP data."""
    from scipy.signal import butter, filtfilt
    
    nyq = 0.5 * fs
    b, a = butter(4, [f/nyq for f in freq_range], btype='band')
    return filtfilt(b, a, lfp_data)
```

### Parallel Processing Options
```python
def parallel_process(func, data_list, n_workers=4):
    """Custom parallel processing implementation."""
    from concurrent.futures import ProcessPoolExecutor
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(func, data_list))
    return results
```

### Advanced Configuration
```python
# config.py
class Config:
    ANALYSIS_PARAMS = {
        'basic': {
            'bin_size': 0.01,
            'window_size': (-0.5, 1.0)
        },
        'lfp': {
            'freq_ranges': {
                'theta': (4, 8),
                'beta': (13, 30),
                'gamma': (30, 80)
            }
        }
    }
```

### Custom Results Export
```python
def export_to_nwb(results, filename):
    """Export results in Neurodata Without Borders format."""
    from pynwb import NWBFile, NWBHDF5IO
    from datetime import datetime
    
    nwb = NWBFile(
        session_description='Steinmetz analysis results',
        identifier=str(uuid.uuid4()),
        session_start_time=datetime.now()
    )
    
    # Add results to file
    # ... implementation ...
    
    with NWBHDF5IO(filename, 'w') as io:
        io.write(nwb)
```

### Performance Monitoring
```python
def monitor_memory_usage(func):
    """Decorator to monitor memory usage."""
    import psutil
    
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        mem_before = process.memory_info().rss
        result = func(*args, **kwargs)
        mem_after = process.memory_info().rss
        print(f"Memory usage: {(mem_after - mem_before) / 1024**2:.2f} MB")
        return result
    return wrapper
``` 