# Data Loading and Structure

This document provides an overview of how data is loaded into the application and the general structure of the session data.

## Data Source

The application uses the Steinmetz dataset, which features recordings from multiple brain regions in mice performing a visual discrimination task. The primary data files are expected to be in `*.npz` format, typically split into parts (e.g., `steinmetz_part0.npz`, `steinmetz_part1.npz`, etc.) located in the `webapp/static/docs/` directory (as configured in the `DataController`).

## Data Loading Mechanism

Data loading is primarily handled by two main controllers:

1.  **`webapp.controllers.data_controller.DataController`**: This is the main interface for the web application to access session data. 
    - It initializes the `SteinmetzDataLoader`.
    - It provides methods like `load_session(session_idx)` to fetch data for a specific session and `get_available_sessions()` to list available sessions.
    - It also handles caching of loaded sessions to improve performance.
    [PYTHON_FUNC:webapp/controllers/data_controller.py:DataController.load_session:method]

2.  **`src.data_loader.SteinmetzDataLoader`**: This class is responsible for the low-level loading of data from the `*.npz` files.
    - It determines which `part*.npz` file to use based on the `session_idx`.
    - It loads the data for the specific session from the appropriate file part.
    - It also handles merging LFP (Local Field Potential) data, which might be stored in a separate `steinmetz_lfp.npz` file.
    - Key method: `load_session(session_idx)`.
    [PYTHON_FUNC:src/data_loader.py:SteinmetzDataLoader.load_session:method]

## Session Data Structure

When a session is loaded (e.g., via `DataController.load_session()`), the result is a Python dictionary typically referred to as `session_data`. This dictionary contains various keys, each mapping to different aspects of the recorded data for that session. Common keys include:

-   `'spks'` (aliased to `'spikes'`): Spike times or binned spike counts. Usually a 3D NumPy array (neurons x trials x time_bins).
-   `'response'`: The animal's response on each trial (e.g., -1 for right choice, 1 for left choice, 0 for no-go). A 1D NumPy array.
-   `'response_time'`: Timing of the animal's response on each trial. A 1D NumPy array.
-   `'contrast_left'`, `'contrast_right'`: Contrast levels of visual stimuli presented on each trial.
-   `'brain_area'`: A list or array indicating the brain region for each recorded neuron.
-   `'bin_size'`: The time bin size used for binning spike data (e.g., 0.01 seconds).
-   `'stim_onset'`: The time point considered as stimulus onset within each trial's time window (e.g., 0.5 seconds).
-   `'lfp'`: LFP data, if available and merged.
-   Many other keys related to behavior, trial structure, and neural recordings.

### Exploring Session Data

You can use the **Data Explorer** tab on the main application page to view the exact keys, data types, shapes, and previews for any loaded session. This is the best way to understand the specific structure of the data you are working with for a given session.

```python
# Example of how data might be accessed (conceptual)
# data_controller = DataController()
# session_2_data = data_controller.load_session(session_idx=1) # Assuming 0-indexed for session 2
# spikes_session_2 = session_2_data['spikes']
# responses_session_2 = session_2_data['response']
```

This structure is critical for understanding how to select data for various analyses available in the dashboard. 