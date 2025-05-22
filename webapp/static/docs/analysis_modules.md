# Analysis Modules and Controller

This document describes the core analysis capabilities of the dashboard, managed primarily by the `AnalysisController`.

## `AnalysisController` Overview

The `webapp.controllers.analysis_controller.AnalysisController` is the central class responsible for:
- Managing the list of available analysis types.
- Handling default and user-provided configurations for each analysis.
- Loading necessary session data via the `DataController`.
- Executing the selected analysis functions.
- Returning structured results suitable for visualization or reporting.

[PYTHON_FUNC:webapp/controllers/analysis_controller.py:AnalysisController:class]

Key methods in `AnalysisController` include:
- `get_available_analyses()`: Returns a list of analysis keys.
- `run_analyses(session_indices, analyses, config)`: Runs multiple analyses on multiple sessions.
- `process_session(session_idx, analyses, config)`: Processes a single session.

[PYTHON_FUNC:webapp/controllers/analysis_controller.py:AnalysisController.run_analyses:method]

## Available Analysis Types

Below are details for some of the available analysis modules.

### 1. Basic Analysis (`basic`)

-   **Description**: Performs fundamental spike train analysis, primarily focusing on generating a Peri-Stimulus Time Histogram (PSTH) for a neuron.
-   **Data Used**: 
    -   `session_data['spikes']`: Binned spike counts (neurons x trials x timebins).
    -   Configuration for `time_window` and `bin_size` (though `bin_size` from data might be used implicitly if spikes are pre-binned).
-   **Configuration Parameters (from `default_configs['basic']`):
    -   `time_window`: Tuple `(start, end)` in seconds, relative to stimulus onset, for PSTH calculation.
    -   `bin_size`: Bin size in seconds for PSTH calculation (note: `NeuralAnalyzer.compute_psth` might re-bin or use existing bins).
-   **Outputs**: A dictionary including:
    -   `psth`: List of PSTH values.
    -   `time_bins`: List of time bin centers/edges for the PSTH.
    -   `neuron_id`: Index of the neuron analyzed (currently defaults to the first neuron for demo).
    -   `time_window`, `bin_size`: The parameters used.
-   **Relevant Code**:
    [PYTHON_FUNC:webapp/controllers/analysis_controller.py:AnalysisController._run_basic_analysis:method]

### 2. LFP Analysis (`lfp`)

-   **Description**: Analyzes Local Field Potential (LFP) data, calculating the power spectrum and power in specific frequency bands.
-   **Data Used**:
    -   `session_data['lfp']`: LFP data (channels x samples or channels x trials x samples).
    -   `session_data['brain_area_lfp']` (optional, for context).
-   **Configuration Parameters**:
    -   `freq_range`: Tuple `(min_freq, max_freq)` for power spectrum calculation.
    -   `freq_bands`: Dictionary defining named frequency bands, e.g., `{'theta': (4, 8)}`.
-   **Outputs**: A dictionary including:
    -   `freqs`: List of frequencies.
    -   `power`: List of power values for the spectrum.
    -   `band_powers`: Dictionary of power per defined band.
    -   `channel_id`: Index of the LFP channel analyzed (currently defaults to the first channel for demo).
-   **Relevant Code**:
    [PYTHON_FUNC:webapp/controllers/analysis_controller.py:AnalysisController._run_lfp_analysis:method]

### 3. Population Analysis (`population`)

-   **Description**: Performs population-level analysis, currently focused on Principal Component Analysis (PCA) of neural activity.
-   **Data Used**:
    -   `session_data['spikes']`: Binned spike counts.
-   **Configuration Parameters**:
    -   `n_components`: Number of principal components to compute.
    -   `scale_data`: Boolean, whether to scale data before PCA.
-   **Outputs**: A dictionary including:
    -   `explained_variance`: Explained variance ratio per component.
    -   `cumulative_variance`: Cumulative explained variance.
    -   `n_components`, `scale_data`: Parameters used.
-   **Relevant Code**:
    [PYTHON_FUNC:webapp/controllers/analysis_controller.py:AnalysisController._run_population_analysis:method]

### 4. Choice-Aligned Analysis (`choice_aligned`)

-   **Description**: Analyzes neural activity aligned to the animal's choice (left vs. right).
    It separates trials by choice direction and calculates average activity profiles.
-   **Data Used**:
    -   `session_data['spikes']`: Binned spike counts (neurons x trials x timebins).
    -   `session_data['response']`: Animal's choice on each trial (e.g., 1 for left, -1 for right).
    -   `session_data['bin_size']`, `session_data['stim_onset']`: For generating the time axis.
-   **Configuration Parameters**: Currently uses a placeholder `example_param`.
-   **Outputs**: A dictionary including:
    -   `avg_activity_left`: Average activity for left-choice trials (neurons x timebins).
    -   `avg_activity_right`: Average activity for right-choice trials (neurons x timebins).
    -   `activity_difference`: Difference (left - right) in average activity.
    -   `time_axis`: Time points relative to stimulus onset.
    -   `n_left_trials`, `n_right_trials`, `n_neurons`.
-   **Relevant Code**:
    [PYTHON_FUNC:webapp/controllers/analysis_controller.py:AnalysisController._run_choice_aligned_analysis:method]


<!-- Add sections for other analyses like behavior, cross_regional etc. --> 