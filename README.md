# Steinmetz Neural Data Analysis Project

This repository contains a comprehensive analysis pipeline for the Steinmetz et al. (2019) dataset, which includes neural recordings from multiple brain regions during a visual decision-making task.

## Project Structure
```
.
├── src/                    # Source code
│   ├── data_loader.py     # Data loading utilities
│   ├── neural_analysis.py # Core analysis functions
│   ├── behavior_analysis.py # Behavioral analysis
│   ├── visualization.py   # Visualization tools
│   └── pipeline.py       # Automated analysis pipeline
├── notebooks/             # Analysis notebooks
│   ├── 1_basic_neural_analysis.ipynb
│   ├── 2_lfp_analysis.ipynb
│   ├── 3_population_dynamics.ipynb
│   ├── 4_behavior_analysis.ipynb
│   ├── 5_cross_regional_analysis.ipynb
│   └── 6_automated_analysis.ipynb
├── data/                  # Data storage
├── results/               # Analysis results
├── figures/              # Generated figures
└── requirements.txt      # Project dependencies
```

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd Neuromatch
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Analysis Notebooks

### 1. Basic Neural Analysis (`1_basic_neural_analysis.ipynb`)
- PSTH analysis aligned to stimulus and choice
- Comparison of neural responses across brain regions
- Analysis of fast-spiking vs regular-spiking neurons
- Trial-to-trial variability analysis

### 2. LFP Analysis (`2_lfp_analysis.ipynb`)
- Power spectral analysis across brain regions
- Phase relationships between regions
- LFP patterns during decision-making
- Spike-LFP relationships
- LFP band analysis (delta, theta, alpha, beta, gamma)

### 3. Population Dynamics (`3_population_dynamics.ipynb`)
- Dimensionality reduction using PCA
- Neural trajectories during decision-making
- Population coding of task variables
- State space analysis

### 4. Behavior Analysis (`4_behavior_analysis.ipynb`)
- Choice probability analysis
- Reaction time analysis
- Sequential effects in behavior
- Neural prediction of behavior

### 5. Cross-Regional Analysis (`5_cross_regional_analysis.ipynb`)
- Inter-regional spike correlations
- LFP coherence analysis
- Information flow analysis
- Region-specific population dynamics
- Task-dependent connectivity

### 6. Automated Analysis (`6_automated_analysis.ipynb`)
- Demonstration of visualization tools
- Automated batch processing
- Interactive visualizations
- Results compilation and export

## Running the Analyses

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `notebooks/` directory and open notebooks in sequence.

3. For automated batch processing:
   ```python
   from pipeline import AnalysisPipeline
   
   # Initialize pipeline
   pipeline = AnalysisPipeline()
   
   # Process single session
   results = pipeline.process_session(
       session_idx=11,
       analyses=['basic', 'lfp', 'population']
   )
   
   # Batch process multiple sessions
   batch_results = pipeline.batch_process(
       session_indices=[11, 12, 13],
       analyses=['basic', 'lfp'],
       n_workers=3
   )
   ```

## Visualization Tools

The project includes a comprehensive visualization module (`src/visualization.py`) with tools for:
- Raster plots
- 3D brain region visualizations
- Population activity heatmaps
- Connectivity matrices
- Interactive time series plots
- Summary dashboards

Example usage:
```python
from visualization import NeuralViz

viz = NeuralViz()

# Create raster plot
viz.plot_raster(spike_times, trials, (-0.5, 0.5), title='Example Neuron')

# Create interactive visualization
viz.create_interactive_timeseries(time, data, labels)
```

## Results

Analysis results are automatically saved in the following structure:
```
results/
├── session_X/           # Results for each session
│   ├── figures/        # Generated plots
│   ├── data/          # Numerical results
│   └── results.json   # Summary statistics
└── batch_summary.json  # Batch processing summary
```

## Data Source
Data from Steinmetz et al. (2019): https://github.com/nsteinme/steinmetz-et-al-2019

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License 