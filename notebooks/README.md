# Steinmetz Dataset Analysis Notebooks

This directory contains a series of Jupyter notebooks for analyzing the Steinmetz et al. (2019) neural recording dataset. The notebooks are designed to be run in sequence, though each can also be used independently.

## Quick Start

For a quick introduction to the analysis tools and basic usage examples, start with:
- [`analysis_example.ipynb`](analysis_example.ipynb): Demonstrates basic usage of all analysis tools and visualization functions

## Main Analysis Notebooks

1. [`1_basic_neural_analysis.ipynb`](1_basic_neural_analysis.ipynb)
   - PSTH analysis aligned to stimulus and choice
   - Comparison of neural responses across brain regions
   - Analysis of fast-spiking vs regular-spiking neurons
   - Trial-to-trial variability analysis
   
2. [`2_lfp_analysis.ipynb`](2_lfp_analysis.ipynb)
   - Power spectral analysis across brain regions
   - Phase relationships between regions
   - LFP patterns during decision-making
   - Spike-LFP relationships
   - LFP band analysis (delta, theta, alpha, beta, gamma)

3. [`3_population_dynamics.ipynb`](3_population_dynamics.ipynb)
   - Dimensionality reduction using PCA
   - Neural trajectories during decision-making
   - Population coding of task variables
   - State space analysis
   - Region-specific population dynamics

4. [`4_behavior_analysis.ipynb`](4_behavior_analysis.ipynb)
   - Choice probability analysis
   - Reaction time analysis
   - Sequential effects in behavior
   - Neural prediction of behavior
   - Neural-behavioral correlations

5. [`5_cross_regional_analysis.ipynb`](5_cross_regional_analysis.ipynb)
   - Inter-regional spike correlations
   - LFP coherence between regions
   - Information flow analysis
   - Region-specific population dynamics
   - Task-dependent connectivity

6. [`6_automated_analysis.ipynb`](6_automated_analysis.ipynb)
   - Automated batch processing
   - Standardized visualizations
   - Results export
   - Performance monitoring
   - Custom analysis configurations

## Running Order

While each notebook can be run independently, the recommended order is:

1. Start with `analysis_example.ipynb` to understand the basic tools
2. Run notebooks 1-5 in sequence for detailed analysis
3. Use notebook 6 for batch processing and automation

## Dependencies

All notebooks require the following Python packages:
```
numpy>=1.21.0
matplotlib>=3.4.0
requests>=2.26.0
scipy>=1.7.0
pandas>=1.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```

## Data Structure

The notebooks expect data in the following format:
- Spike times: List of spike times for each neuron
- LFP data: Array of LFP signals (time x channels)
- Brain areas: List of brain area labels
- Behavioral data: Trial outcomes and timing information

## Output Structure

Analysis results are saved in the following locations:
- `../results/`: Numerical results and data
- `../figures/`: Generated plots and visualizations
- `../results/batch_analysis_*/`: Results from batch processing

## Common Issues and Solutions

1. Memory Issues
   - Reduce batch size in analysis
   - Process fewer neurons/channels at once
   - Use memory-efficient operations

2. Visualization Problems
   - Use `plt.close()` after generating plots
   - Set appropriate backend with `matplotlib.use()`
   - Enable notebook integration for interactive plots

3. Data Loading
   - Ensure data files are in the correct location
   - Check internet connection for downloads
   - Verify file permissions

## References

- Steinmetz, N.A., et al. (2019). Distributed coding of choice, action and engagement across the mouse brain. Nature, 576(7786), 266-273.
- Original data repository: https://github.com/nsteinme/steinmetz-et-al-2019

## Contributing

Feel free to contribute by:
1. Adding new analysis methods
2. Improving visualizations
3. Optimizing performance
4. Adding documentation
5. Reporting issues

## Contact

For questions or issues, please open an issue in the repository. 