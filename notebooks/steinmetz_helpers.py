import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, pearsonr
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind

def ensure_packages_installed():
    """
    Check if required packages are installed and install them if needed.
    """
    try:
        import importlib
        import subprocess
        import sys
        
        # List of required packages with their specific versions
        # Using TensorFlow 2.8.0 and SHAP 0.41.0 which are known to work well together
        required_packages = {
            'tensorflow': '2.8.0',
            'shap': '0.41.0'
        }
        
        # Check if we're in a Jupyter/Colab environment
        in_jupyter = 'ipykernel' in sys.modules
        packages_installed = False
        
        if in_jupyter:
            # In Jupyter/Colab, we can use !pip install
            for package, version in required_packages.items():
                try:
                    # Try to import the package
                    module = importlib.import_module(package)
                    
                    # Get the version
                    current_version = getattr(module, '__version__', '0.0.0')
                    
                    if current_version != version:
                        print(f"Installing {package}=={version} (current version: {current_version})...")
                        from IPython.display import display, HTML
                        display(HTML(f"<p>Installing {package}=={version}...</p>"))
                        
                        # Use IPython magic command for pip
                        import IPython
                        IPython.get_ipython().system(f'pip install {package}=={version} --quiet')
                        packages_installed = True
                        
                        # Force reload the module
                        if package in sys.modules:
                            import importlib
                            importlib.reload(sys.modules[package])
                        
                        print(f"{package}=={version} has been installed.")
                    else:
                        print(f"{package}=={version} is already installed.")
                        
                except ImportError:
                    print(f"Installing {package}=={version}...")
                    from IPython.display import display, HTML
                    display(HTML(f"<p>Installing {package}=={version}...</p>"))
                    
                    # Use IPython magic command for pip
                    import IPython
                    IPython.get_ipython().system(f'pip install {package}=={version} --quiet')
                    packages_installed = True
                    
                    # Import the newly installed package
                    importlib.import_module(package)
                    print(f"{package}=={version} has been installed.")
        else:
            # In a regular Python environment, use subprocess
            for package, version in required_packages.items():
                try:
                    # Try to import the package
                    module = importlib.import_module(package)
                    
                    # Get the version
                    current_version = getattr(module, '__version__', '0.0.0')
                    
                    if current_version != version:
                        print(f"Installing {package}=={version} (current version: {current_version})...")
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{package}=={version}', '--quiet'])
                        packages_installed = True
                        
                        # Force reload the module
                        if package in sys.modules:
                            import importlib
                            importlib.reload(sys.modules[package])
                        
                        print(f"{package}=={version} has been installed.")
                    else:
                        print(f"{package}=={version} is already installed.")
                        
                except ImportError:
                    print(f"Installing {package}=={version}...")
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{package}=={version}', '--quiet'])
                    packages_installed = True
                    
                    # Import the newly installed package
                    importlib.import_module(package)
                    print(f"{package}=={version} has been installed.")
        
        # If packages were installed, suggest restarting the kernel
        if packages_installed and in_jupyter:
            from IPython.display import display, HTML
            display(HTML(
                "<div style='background-color: #FFFFCC; padding: 10px; border: 1px solid #FFCC00; border-radius: 5px;'>"
                "<p><strong>Packages were installed or updated.</strong></p>"
                "<p>It's recommended to restart the kernel to ensure all changes take effect.</p>"
                "<p>Use the <code>restart_kernel()</code> function or manually restart the kernel.</p>"
                "</div>"
            ))
        
        # Ensure TensorFlow is properly configured
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Eager execution enabled: {tf.executing_eagerly()}")
        
        # If eager execution is not enabled, enable it
        if not tf.executing_eagerly():
            print("Enabling eager execution...")
            tf.compat.v1.enable_eager_execution()
            print(f"Eager execution now enabled: {tf.executing_eagerly()}")
            
    except Exception as e:
        print(f"Error ensuring packages: {e}")
        print("You may need to manually install required packages.")
        print("Try running: !pip install tensorflow==2.8.0 shap==0.41.0")

def restart_kernel():
    """
    Restart the Jupyter kernel.
    This function should be called after installing new packages.
    """
    try:
        from IPython.display import display, HTML
        display(HTML("<p>Restarting kernel...</p>"))
        import IPython
        IPython.get_ipython().kernel.do_shutdown(True)
        print("Kernel restarted.")
    except Exception as e:
        print(f"Error restarting kernel: {e}")
        print("Please restart the kernel manually.")

# Data loading and preprocessing functions
def load_steinmetz_data(fname_parts=['steinmetz_part0.npz', 'steinmetz_part1.npz', 'steinmetz_part2.npz']):
    """
    Load the Steinmetz dataset from the specified files.
    
    Parameters:
    -----------
    fname_parts : list
        List of filenames to load
        
    Returns:
    --------
    alldat : list
        List of dictionaries containing the data for each session
    """
    alldat = []
    for fname in fname_parts:
        try:
            data = np.load(fname, allow_pickle=True)
            loaded_data = data['dat'][()]
            
            # Check if loaded_data is a list or array
            if isinstance(loaded_data, (list, np.ndarray)):
                # If it's a list or array, extend alldat with its items
                for item in loaded_data:
                    if isinstance(item, dict):
                        alldat.append(item)
                    else:
                        print(f"Warning: Item in {fname} is not a dictionary. Skipping.")
            elif isinstance(loaded_data, dict):
                # If it's a single dictionary, append it
                alldat.append(loaded_data)
            else:
                print(f"Warning: Data in {fname} is not in expected format. Skipping.")
                
        except FileNotFoundError:
            print(f"File {fname} not found. Please download it first.")
    
    print(f"Loaded {len(alldat)} sessions from {len(fname_parts)} files.")
    return alldat

def load_steinmetz_lfp(fname='steinmetz_lfp.npz'):
    """
    Load the Steinmetz LFP dataset.
    
    Parameters:
    -----------
    fname : str
        Filename to load
        
    Returns:
    --------
    dat_LFP : list
        List of dictionaries containing the LFP data for each session
    """
    try:
        data = np.load(fname, allow_pickle=True)
        loaded_data = data['dat'][()]
        
        # Initialize an empty list for the LFP data
        dat_LFP = []
        
        # Check if loaded_data is a list or array
        if isinstance(loaded_data, (list, np.ndarray)):
            # If it's a list or array, extend dat_LFP with its items
            for item in loaded_data:
                if isinstance(item, dict):
                    dat_LFP.append(item)
                else:
                    print(f"Warning: Item in {fname} is not a dictionary. Skipping.")
        elif isinstance(loaded_data, dict):
            # If it's a single dictionary, append it
            dat_LFP.append(loaded_data)
        else:
            print(f"Warning: Data in {fname} is not in expected format. Skipping.")
        
        print(f"Loaded {len(dat_LFP)} LFP sessions from {fname}.")
        return dat_LFP
    
    except FileNotFoundError:
        print(f"File {fname} not found. Please download it first.")
        return []

def combine_data(alldat, dat_LFP):
    """
    Combine the spike and LFP data into a single dataset.
    
    Parameters:
    -----------
    alldat : list
        List of dictionaries containing the spike data
    dat_LFP : list or ndarray
        List or array of dictionaries containing the LFP data
        
    Returns:
    --------
    combined_data : list
        List of dictionaries containing both spike and LFP data
    """
    combined_data = []
    
    # Convert dat_LFP to list if it's a numpy array
    if isinstance(dat_LFP, np.ndarray):
        dat_LFP = dat_LFP.tolist()
    
    # Check if both lists are empty
    if not alldat and not dat_LFP:
        print("Warning: Both alldat and dat_LFP are empty.")
        return np.array(combined_data)
    
    # If one list is empty, return the other
    if not alldat:
        print("Warning: alldat is empty. Returning dat_LFP.")
        return np.array(dat_LFP)
    
    if not dat_LFP:
        print("Warning: dat_LFP is empty. Returning alldat.")
        return np.array(alldat)
    
    # Match sessions by mouse_name and date_exp if available
    matched_pairs = []
    
    # First, try to match by mouse_name and date_exp
    for i, spike_session in enumerate(alldat):
        if 'mouse_name' in spike_session and 'date_exp' in spike_session:
            mouse_name = spike_session['mouse_name']
            date_exp = spike_session['date_exp']
            
            for j, lfp_session in enumerate(dat_LFP):
                if ('mouse_name' in lfp_session and 'date_exp' in lfp_session and 
                    lfp_session['mouse_name'] == mouse_name and 
                    lfp_session['date_exp'] == date_exp):
                    matched_pairs.append((i, j))
                    break
    
    # If no matches found, try to match by index if lengths are the same
    if not matched_pairs and len(alldat) == len(dat_LFP):
        print("Warning: No matches found by mouse_name and date_exp. Matching by index.")
        matched_pairs = [(i, i) for i in range(len(alldat))]
    
    # If still no matches, use the shorter list's length
    if not matched_pairs:
        print("Warning: Could not match sessions. Combining by index up to the shorter list's length.")
        min_len = min(len(alldat), len(dat_LFP))
        matched_pairs = [(i, i) for i in range(min_len)]
    
    # Combine the matched pairs
    for spike_idx, lfp_idx in matched_pairs:
        spike_session = alldat[spike_idx]
        lfp_session = dat_LFP[lfp_idx]
        
        # Make sure both are dictionaries before combining
        if isinstance(lfp_session, dict) and isinstance(spike_session, dict):
            # Create a new dictionary to avoid modifying the originals
            combined_session = {}
            
            # Add all keys from spike_session
            for key, value in spike_session.items():
                combined_session[key] = value
            
            # Add all keys from lfp_session, potentially overwriting spike_session keys
            for key, value in lfp_session.items():
                # Skip if the key already exists and the values are different
                if key in combined_session and np.any(combined_session[key] != value):
                    print(f"Warning: Key '{key}' exists in both sessions with different values. Using spike session value.")
                else:
                    combined_session[key] = value
            
            combined_data.append(combined_session)
        else:
            print(f"Warning: Data at indices ({spike_idx}, {lfp_idx}) are not dictionaries. Skipping.")
    
    print(f"Combined {len(combined_data)} sessions.")
    return np.array(combined_data)

def filter_by_mouse(combined_data, mouse_name):
    """
    Filter the combined data by mouse name.
    
    Parameters:
    -----------
    combined_data : list
        List of dictionaries containing the combined data
    mouse_name : str
        Name of the mouse to filter by
        
    Returns:
    --------
    filtered_data : list
        List of dictionaries containing the filtered data
    """
    filtered_data = []
    for i in range(len(combined_data)):
        if combined_data[i]['mouse_name'] == mouse_name:
            filtered_data.append(combined_data[i])
    return np.array(filtered_data)

# Analysis functions for neural dynamics
def analyze_brain_regions(sessions, brain_groups=None, region_colors=None):
    """
    Analyze brain regions across sessions.
    
    Parameters:
    -----------
    sessions : list
        List of session dictionaries
    brain_groups : list, optional
        List of lists containing brain regions grouped by function
    region_colors : list, optional
        List of colors for each brain region group
    
    Returns:
    --------
    tuple
        (areas_by_dataset, connectivity_matrix, area_to_index, brain_groups, region_colors)
    """
    if brain_groups is None:
        # Define brain regions by functional group
        brain_groups = [
            ["VISp", "VISl", "VISrl", "VISam", "VISpm", "VIS", "VISa", "VISal"],  # Visual cortex
            ["CL", "LD", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SGN", "TH", "VAL", "VPL", "VPM"],  # Thalamus
            ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST", "PRE", "ProS", "HPF"],  # Hippocampal formation
            ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBl", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP", "TT", "TTd", "TTv"],  # Non-visual cortex
            ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "NDB", "SI", "SRN", "TRS"],  # Striatum and pallidum
            ["MB", "SC", "SCm", "SCsg", "ZI"]  # Midbrain
        ]
    
    if region_colors is None:
        # Define colors for each group
        region_colors = ["blue", "red", "green", "purple", "orange", "brown"]
    
    # Collect all unique brain areas from the sessions
    all_areas = set(["root"])  # Start with root
    for session in sessions:
        if 'brain_area' in session:
            all_areas.update(session['brain_area'])
    
    # Create a mapping of brain areas to indices
    area_to_index = {"root": 0}
    idx = 1
    
    # First add areas from brain_groups to maintain order
    for group in brain_groups:
        for area in group:
            if area not in area_to_index and area in all_areas:
                area_to_index[area] = idx
                idx += 1
    
    # Then add any remaining areas from sessions
    for area in all_areas:
        if area not in area_to_index:
            area_to_index[area] = idx
            idx += 1
    
    # Create a matrix to track which areas are present in each dataset
    num_areas = len(area_to_index)
    num_sessions = len(sessions)
    areas_by_dataset = np.zeros((num_areas, num_sessions))
    
    # Populate the matrix
    for s_idx, session in enumerate(sessions):
        if 'brain_area' in session:
            areas = session['brain_area']
            for area in areas:
                if area in area_to_index:
                    areas_by_dataset[area_to_index[area], s_idx] = 1
    
    # Create a connectivity matrix
    connectivity_matrix = np.zeros((num_areas, num_areas))
    
    # Populate the connectivity matrix based on co-occurrence
    for s_idx in range(num_sessions):
        present_areas = np.where(areas_by_dataset[:, s_idx] == 1)[0]
        for i in present_areas:
            for j in present_areas:
                if i != j:
                    connectivity_matrix[i, j] += 1
    
    # Normalize by the number of sessions
    connectivity_matrix = connectivity_matrix / num_sessions
    
    return areas_by_dataset, connectivity_matrix, area_to_index, brain_groups, region_colors

def plot_brain_regions_presence(areas_by_dataset, area_to_index, brain_groups, region_colors):
    """
    Plot the presence of brain regions across sessions.
    
    Parameters:
    -----------
    areas_by_dataset : numpy.ndarray
        Binary matrix indicating which areas are present in each dataset
    area_to_index : dict
        Dictionary mapping brain areas to indices
    brain_groups : list
        List of lists containing brain regions grouped by function
    region_colors : list
        List of colors for each brain region group
    """
    plt.figure(figsize=(12, 10))
    plt.imshow(areas_by_dataset, cmap="Greys", aspect="auto", interpolation="none")
    
    # Label the axes
    plt.xlabel("Dataset")
    plt.ylabel("Brain Area")
    
    # Create a reverse mapping from index to area
    index_to_area = {v: k for k, v in area_to_index.items()}
    
    # Add tick labels
    yticklabels = [index_to_area.get(i, f"Unknown_{i}") for i in range(len(area_to_index))]
    plt.yticks(np.arange(len(area_to_index)), yticklabels, fontsize=8)
    plt.xticks(np.arange(areas_by_dataset.shape[1]), fontsize=9)
    
    # Color the tick labels by region
    ytickobjs = plt.gca().get_yticklabels()
    
    # Set default color for all labels
    for label in ytickobjs:
        label.set_color('gray')  # Default color for unknown areas
    
    # Set specific colors for known areas
    ytickobjs[0].set_color("black")  # root
    
    # Create a flat list of all areas in brain_groups
    all_group_areas = []
    for group in brain_groups:
        all_group_areas.extend(group)
    
    # Color the labels for areas in brain_groups
    for i, area_name in enumerate(yticklabels):
        for group_idx, group in enumerate(brain_groups):
            if area_name in group:
                ytickobjs[i].set_color(region_colors[group_idx])
                break
    
    plt.title("Brain areas present in each session")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def prepare_lstm_data(alldat, brain_groups):
    """
    Prepare data for LSTM model to analyze neural dynamics.
    
    Parameters:
    -----------
    alldat : list
        List of dictionaries containing the data for each session
    brain_groups : list
        List of lists containing brain regions grouped by function
        
    Returns:
    --------
    X_lstm_padded : numpy.ndarray
        Padded input data for LSTM model
    y_lstm : numpy.ndarray
        Labels for LSTM model
    """
    # Include 'other' brain areas not listed in brain_groups
    all_known_areas = [area for group in brain_groups for area in group]
    nareas = len(brain_groups) + 1  # Adding 1 for 'other' brain areas
    
    # Initialize lists to collect features and labels from all sessions
    X_lstm_list = []
    y_lstm_list = []
    
    # Loop over each session in alldat
    for session_idx, dat in enumerate(alldat):
        print(f"Processing session {session_idx + 1}/{len(alldat)}")
        
        # Extract session data
        spks = dat['spks']  # Shape: neurons x trials x time
        brain_areas = dat['brain_area']  # List of brain areas per neuron
        response = dat['response']  # Responses per trial (-1: left, 1: right, 0: none)
        dt = dat.get('bin_size', 0.01)  # Time bin size in seconds (use 'bin_size' if available)
        
        NN = len(brain_areas)  # Number of neurons
        barea = (nareas - 1) * np.ones(NN,)  # Initialize brain area indices to 'other' (last index)
        
        # Assign brain area indices to neurons
        for j in range(len(brain_groups)):
            is_in_group = np.isin(brain_areas, brain_groups[j])
            barea[is_in_group] = j  # Assign index j to neurons in brain_groups[j]
        
        num_trials = spks.shape[1]
        time_steps = spks.shape[2]
        
        # Process each trial
        for trial in range(num_trials):
            # Initialize a 2D array for this trial: time_steps x nareas
            trial_data = np.zeros((time_steps, nareas))
            
            # Process each brain region (including 'other')
            for j in range(nareas):
                neurons_in_region = np.where(barea == j)[0]
                if len(neurons_in_region) == 0:
                    # If no neurons in this brain region, fill with zeros
                    trial_data[:, j] = 0.0
                else:
                    # Extract spike data for neurons in this region and trial
                    spikes = spks[neurons_in_region, trial, :]  # Shape: neurons x time_steps
                    # Compute average firing rate over neurons at each time step
                    avg_spikes = np.mean(spikes, axis=0) / dt  # Firing rate per time step
                    trial_data[:, j] = avg_spikes
            
            # Append the trial data and label
            X_lstm_list.append(trial_data)  # Shape: time_steps x nareas
            y_lstm_list.append(response[trial])
    
    # Convert labels to array
    y_lstm = np.array(y_lstm_list)
    
    # Map responses to labels
    label_mapping = {-1: 0, 1: 1, 0: 2}  # 0: Left, 1: Right, 2: None
    y_lstm = np.array([label_mapping[label] for label in y_lstm])
    
    # Determine the maximum sequence length
    max_time_steps = max(trial.shape[0] for trial in X_lstm_list)
    
    # Pad sequences to the maximum length
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        X_lstm_padded = pad_sequences(
            X_lstm_list,
            maxlen=max_time_steps,
            dtype='float32',
            padding='post',
            truncating='post',
            value=0.0  # Value to pad with
        )
    except ImportError:
        print("TensorFlow not available. Using a simple padding implementation.")
        # Simple padding implementation
        X_lstm_padded = np.zeros((len(X_lstm_list), max_time_steps, nareas), dtype=np.float32)
        for i, x in enumerate(X_lstm_list):
            X_lstm_padded[i, :x.shape[0], :] = x
    
    return X_lstm_padded, y_lstm

# Visualization functions for LFP analysis
def plot_lfp_feedback(session_data, feedback_value, title_suffix):
    """
    Plot LFP activity for a given feedback value.
    
    Parameters:
    -----------
    session_data : dict
        Dictionary containing the session data
    feedback_value : int
        Feedback value to plot (-1 for negative, 1 for positive)
    title_suffix : str
        Suffix for the plot title
    """
    lfp_data = session_data['lfp']
    feedback = session_data['feedback_type']
    response_times = session_data['response_time']
    feedback_times = session_data['feedback_time']
    brain_areas = session_data['brain_area_lfp']
    
    # Time vector for plotting
    time = np.linspace(0, 2500, 250)  # Assuming 2500 ms total
    
    feedback_indices = np.where(feedback == feedback_value)[0]
    lfp_filtered = lfp_data[:, feedback_indices, :]
    average_lfp = np.mean(lfp_filtered, axis=1)
    mean_response_time = np.mean(response_times[feedback_indices]) * 1000  # converting to ms
    mean_feedback_time = np.mean(feedback_times[feedback_indices]) * 1000  # converting to ms
    
    fig = go.Figure()
    
    for i, area in enumerate(brain_areas):
        fig.add_trace(go.Scatter(
            x=time,
            y=average_lfp[i],
            mode='lines',
            name=area
        ))
    
    # Adding mean time vertical lines
    fig.add_vline(x=mean_response_time, line_width=2, line_dash="dash", line_color="red", name="Mean Response Time")
    fig.add_vline(x=mean_feedback_time, line_width=2, line_dash="dot", line_color="green", name="Mean Feedback Time")
    fig.add_vline(x=500, line_width=2, line_dash="dash", line_color="blue", name="Stim Onset")
    fig.add_vline(x=1000, line_width=2, line_dash="dash", line_color="purple", name="Go Cue Start")
    fig.add_vline(x=1200, line_width=2, line_dash="dash", line_color="purple", name="Go Cue End")
    
    fig.update_layout(
        title=f"{title_suffix} Feedback - Average LFP Activity Across Brain Areas",
        xaxis_title="Time (ms)",
        yaxis_title="LFP (µV)",
        legend_title="Brain Areas",
        height=600, width=800
    )
    
    return fig

def analyze_cross_correlation(sessions):
    """
    Analyze cross-correlation between brain regions across sessions.
    
    Parameters:
    -----------
    sessions : list
        List of dictionaries containing the session data
        
    Returns:
    --------
    correlations : dict
        Dictionary containing correlation values for each brain region between sessions
    """
    correlations = {}
    
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            common_regions = set(sessions[i]['brain_area_lfp']).intersection(set(sessions[j]['brain_area_lfp']))
            print(f"Overlapping areas between Session {i+1} and Session {j+1}: {list(common_regions)}")
            
            for region in common_regions:
                indices_i = [idx for idx, area in enumerate(sessions[i]['brain_area_lfp']) if area == region]
                indices_j = [idx for idx, area in enumerate(sessions[j]['brain_area_lfp']) if area == region]
                
                if len(indices_i) > 0 and len(indices_j) > 0:
                    data_i = np.mean(sessions[i]['lfp'][indices_i], axis=1)
                    data_j = np.mean(sessions[j]['lfp'][indices_j], axis=1)
                    
                    if data_i.size > 1 and data_j.size > 1:
                        correlation = np.corrcoef(data_i.flatten(), data_j.flatten())[0, 1]
                        correlations[(region, f'Session {i+1}', f'Session {j+1}')] = correlation
                        print(f"Correlation for {region} between Session {i+1} and Session {j+1}: {correlation}")
    
    return correlations

def plot_correlation_heatmap(correlations):
    """
    Plot a heatmap of correlations between brain regions across sessions.
    
    Parameters:
    -----------
    correlations : dict
        Dictionary containing correlation values for each brain region between sessions
    """
    # Creating a DataFrame for plotting
    data = []
    for (region, sess1, sess2), corr in correlations.items():
        data.append([region, f"{sess1} vs {sess2}", corr])
    
    df = pd.DataFrame(data, columns=['Region', 'Session Comparison', 'Correlation'])
    
    # Pivot for heatmap
    pivot_table = df.pivot(index='Region', columns='Session Comparison', values='Correlation')
    
    # Adjust figure size
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Brain Regions Across Sessions')
    plt.show()
    
    return df

def analyze_visual_stimuli_response(session_data):
    """
    Analyze LFP response to visual stimuli.
    
    Parameters:
    -----------
    session_data : dict
        Dictionary containing the session data
    """
    lfp_data = session_data['lfp']  # shape: (n_channels, n_trials, n_timepoints)
    vis_left = session_data['contrast_left']  # length: n_trials
    vis_right = session_data['contrast_right']  # length: n_trials
    regions = session_data['brain_area_lfp']  # length: n_channels
    time = np.linspace(0, 2500, 250)  # Assuming 250 time points
    
    right_only_mask = np.logical_and(vis_left == 0, vis_right > 0)
    left_only_mask = np.logical_and(vis_left > 0, vis_right == 0)
    both_mask = np.logical_and(vis_left > 0, vis_right > 0)
    
    # Re-define unique_regions
    unique_regions = np.unique([r.lower() for r in regions])
    
    fig, axes = plt.subplots(1, len(unique_regions), figsize=(len(unique_regions) * 5, 5), sharey=True)
    
    for j, region in enumerate(unique_regions):
        ax = axes[j]
        ax.set_title(region)
        
        region_indices = np.where([r.lower() == region for r in regions])[0]
        
        for mask, label in zip([right_only_mask, left_only_mask, both_mask], ['Right Only', 'Left Only', 'Both']):
            if np.any(mask):
                if region_indices.size > 0:
                    avg_lfp = np.mean(lfp_data[region_indices][:, mask, :], axis=1)  # mean across trials
                    ax.plot(time, np.mean(avg_lfp, axis=0), label=label)  # Further average across channels if needed
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Average LFP Activity (µV)')
        if len(ax.get_legend_handles_labels()[0]) > 0:  # Check if there are any legend entries to add
            ax.legend()
    
    plt.tight_layout()
    plt.show()

# Age-related analysis functions
def analyze_age_differences(young_sessions, old_sessions, brain_groups):
    """
    Analyze age-related differences in neural dynamics.
    
    Parameters:
    -----------
    young_sessions : list
        List of dictionaries containing data for young mice
    old_sessions : list
        List of dictionaries containing data for old mice
    brain_groups : list
        List of lists containing brain regions grouped by function
        
    Returns:
    --------
    age_diff_metrics : dict
        Dictionary containing metrics of age-related differences
    """
    age_diff_metrics = {}
    
    # Analyze response times
    young_response_times = np.concatenate([session['response_time'].flatten() for session in young_sessions]) * 1000
    old_response_times = np.concatenate([session['response_time'].flatten() for session in old_sessions]) * 1000
    
    age_diff_metrics['mean_response_time_young'] = np.mean(young_response_times)
    age_diff_metrics['mean_response_time_old'] = np.mean(old_response_times)
    age_diff_metrics['median_response_time_young'] = np.median(young_response_times)
    age_diff_metrics['median_response_time_old'] = np.median(old_response_times)
    
    # Analyze feedback types (success rates)
    young_feedback = np.concatenate([session['feedback_type'].flatten() for session in young_sessions])
    old_feedback = np.concatenate([session['feedback_type'].flatten() for session in old_sessions])
    
    age_diff_metrics['success_rate_young'] = np.mean(young_feedback == 1)
    age_diff_metrics['success_rate_old'] = np.mean(old_feedback == 1)
    
    return age_diff_metrics

def plot_age_differences(age_diff_metrics):
    """
    Plot age-related differences in neural dynamics.
    
    Parameters:
    -----------
    age_diff_metrics : dict
        Dictionary containing metrics of age-related differences
    """
    # Plot response times
    plt.figure(figsize=(10, 5))
    
    # Response time comparison
    plt.subplot(1, 2, 1)
    response_times = [age_diff_metrics['mean_response_time_young'], age_diff_metrics['mean_response_time_old']]
    plt.bar(['Young', 'Old'], response_times, color=['blue', 'orange'])
    plt.title('Mean Response Time by Age')
    plt.ylabel('Response Time (ms)')
    
    # Success rate comparison
    plt.subplot(1, 2, 2)
    success_rates = [age_diff_metrics['success_rate_young'], age_diff_metrics['success_rate_old']]
    plt.bar(['Young', 'Old'], success_rates, color=['blue', 'orange'])
    plt.title('Success Rate by Age')
    plt.ylabel('Success Rate')
    
    plt.tight_layout()
    plt.show()

# LSTM model for decoding temporal neural dynamics
def build_lstm_model(X_train, y_train, X_test, y_test, num_classes=3):
    """
    Build and train an LSTM model to decode temporal neural dynamics.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training input data
    y_train : numpy.ndarray
        Training labels
    X_test : numpy.ndarray
        Testing input data
    y_test : numpy.ndarray
        Testing labels
    num_classes : int
        Number of classes
        
    Returns:
    --------
    model : tensorflow.keras.Model or None
        Trained LSTM model or None if TensorFlow is not available
    history : tensorflow.keras.callbacks.History or None
        Training history or None if TensorFlow is not available
    """
    # Ensure required packages are installed
    ensure_packages_installed()
    
    try:
        import tensorflow as tf
        
        # Make sure eager execution is enabled
        if not tf.executing_eagerly():
            print("Enabling eager execution...")
            tf.compat.v1.enable_eager_execution()
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, Masking, TimeDistributed, BatchNormalization, LSTM, Dense
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.callbacks import EarlyStopping
        
        # One-hot encode the labels for categorical cross-entropy loss
        y_train_categorical = to_categorical(y_train, num_classes=num_classes)
        y_test_categorical = to_categorical(y_test, num_classes=num_classes)
        
        # Define EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        
        # Build the LSTM model
        model = Sequential()
        
        # Add the Input layer
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        
        # Add the Masking layer to handle padded values
        model.add(Masking(mask_value=0.0))
        
        # Add the LSTM layer
        model.add(LSTM(64, return_sequences=False))
        
        # Add BatchNormalization
        model.add(BatchNormalization())
        
        # Add the Dense layers
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Display the model summary
        model.summary()
        
        # Train the model with EarlyStopping
        try:
            history = model.fit(
                X_train, y_train_categorical,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test_categorical),
                callbacks=[early_stopping]
            )
        except (LookupError, RuntimeError) as e:
            error_msg = str(e)
            if 'shap_DivNoNan' in error_msg:
                print("Encountered SHAP-related error. Trying alternative approach...")
                # Try a different approach without disabling eager execution
                
                # Create a new model with the same architecture
                new_model = Sequential()
                new_model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
                new_model.add(Masking(mask_value=0.0))
                new_model.add(LSTM(64, return_sequences=False))
                new_model.add(BatchNormalization())
                new_model.add(Dense(32, activation='relu'))
                new_model.add(Dense(num_classes, activation='softmax'))
                
                # Use a different optimizer that might be more compatible
                new_model.compile(loss='categorical_crossentropy', 
                                 optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), 
                                 metrics=['accuracy'])
                
                # Train with the new model
                history = new_model.fit(
                    X_train, y_train_categorical,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test_categorical),
                    callbacks=[early_stopping]
                )
                
                # Replace the original model with the new one
                model = new_model
            elif 'eager mode' in error_msg:
                print("Encountered eager execution error. Trying alternative approach...")
                
                # Make sure eager execution is enabled
                if hasattr(tf, 'enable_eager_execution'):
                    tf.enable_eager_execution()
                
                # Create a new model with the same architecture
                new_model = Sequential()
                new_model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
                new_model.add(Masking(mask_value=0.0))
                new_model.add(LSTM(64, return_sequences=False))
                new_model.add(BatchNormalization())
                new_model.add(Dense(32, activation='relu'))
                new_model.add(Dense(num_classes, activation='softmax'))
                
                # Use a different optimizer that might be more compatible
                new_model.compile(loss='categorical_crossentropy', 
                                 optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), 
                                 metrics=['accuracy'])
                
                # Convert data to tensors explicitly
                X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
                y_train_tensor = tf.convert_to_tensor(y_train_categorical, dtype=tf.float32)
                X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
                y_test_tensor = tf.convert_to_tensor(y_test_categorical, dtype=tf.float32)
                
                # Train with the new model and tensor data
                history = new_model.fit(
                    X_train_tensor, y_train_tensor,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test_tensor, y_test_tensor),
                    callbacks=[early_stopping]
                )
                
                # Replace the original model with the new one
                model = new_model
            else:
                raise e
        
        return model, history
    
    except ImportError as e:
        print(f"TensorFlow not available: {e}")
        return None, None
    except Exception as e:
        print(f"Error building or training LSTM model: {e}")
        return None, None

def evaluate_lstm_model(model, X_test, y_test, num_classes=3):
    """
    Evaluate the LSTM model on the test set.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model or None
        Trained LSTM model or None if TensorFlow is not available
    X_test : numpy.ndarray
        Testing input data
    y_test : numpy.ndarray
        Testing labels
    num_classes : int
        Number of classes
        
    Returns:
    --------
    metrics_df : pandas.DataFrame
        DataFrame containing evaluation metrics
    """
    # If model is None, return empty DataFrame
    if model is None:
        print("No model to evaluate. Returning empty DataFrame.")
        return pd.DataFrame({'Accuracy': [0], 'Precision': [0], 'Recall': [0], 'F1-score': [0]}, index=['LSTM'])
    
    try:
        from tensorflow.keras.utils import to_categorical
        
        # One-hot encode the labels for categorical cross-entropy loss
        y_test_categorical = to_categorical(y_test, num_classes=num_classes)
        
        # Evaluate the model on the test set
        loss, accuracy = model.evaluate(X_test, y_test_categorical)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        
        # Predict on the test data
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_test_decoded = np.argmax(y_test_categorical, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test_decoded, y_pred)
        
        # Get classification report
        report = classification_report(y_test_decoded, y_pred, target_names=['Left', 'Right', 'None'], output_dict=True)
        
        # Store metrics
        metrics = {
            'LSTM': {
                'Accuracy': accuracy,
                'Precision': report['weighted avg']['precision'],
                'Recall': report['weighted avg']['recall'],
                'F1-score': report['weighted avg']['f1-score']
            }
        }
        
        # Convert the metrics dictionary to a DataFrame
        metrics_df = pd.DataFrame(metrics).T  # Transpose to get models as rows
        
        return metrics_df
    
    except ImportError:
        print("TensorFlow not available. Model evaluation could not be performed.")
        return pd.DataFrame({'Accuracy': [0], 'Precision': [0], 'Recall': [0], 'F1-score': [0]}, index=['LSTM'])

def check_feature_importance_inputs(model, X_sample, regions, nareas):
    """
    Check and fix inputs for feature importance analysis.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model or None
        Trained LSTM model or None if TensorFlow is not available
    X_sample : numpy.ndarray
        Sample input data
    regions : list
        List of brain region names
    nareas : int
        Number of brain areas
        
    Returns:
    --------
    tuple
        (fixed_model, fixed_X_sample, fixed_regions, fixed_nareas)
    """
    # Check model
    if model is None:
        print("Warning: model is None. Will return zeros for feature importance.")
        fixed_model = None
    else:
        fixed_model = model
    
    # Check X_sample
    if X_sample is None:
        print("Warning: X_sample is None. Using random data for demonstration.")
        # Create random data with the right shape
        if fixed_model is not None and hasattr(fixed_model, 'input_shape'):
            input_shape = fixed_model.input_shape
            if input_shape and len(input_shape) > 1:
                # Create sample data with the right shape
                fixed_X_sample = np.random.random((10, input_shape[1], input_shape[2]))
            else:
                # Default shape if we can't determine it
                fixed_X_sample = np.random.random((10, 100, 7))
        else:
            # Default shape if we can't determine it
            fixed_X_sample = np.random.random((10, 100, 7))
    else:
        fixed_X_sample = X_sample
    
    # Check regions
    if regions is None or not isinstance(regions, (list, tuple, np.ndarray)):
        print("Warning: regions is None or not a list. Using generic region names.")
        # Determine number of features from X_sample
        if fixed_X_sample is not None and len(fixed_X_sample.shape) > 2:
            num_features = fixed_X_sample.shape[2]
        else:
            num_features = 7  # Default
        fixed_regions = [f"Region_{i+1}" for i in range(num_features)]
    else:
        fixed_regions = list(regions)  # Convert to list to ensure it's mutable
    
    # Check nareas
    if nareas is None or not isinstance(nareas, (int, np.integer)):
        print("Warning: nareas is None or not an integer. Using number of features from X_sample.")
        # Determine number of features from X_sample
        if fixed_X_sample is not None and len(fixed_X_sample.shape) > 2:
            fixed_nareas = fixed_X_sample.shape[2]
        else:
            fixed_nareas = len(fixed_regions)
    else:
        fixed_nareas = int(nareas)
    
    # Ensure nareas is consistent with X_sample
    if fixed_X_sample is not None and len(fixed_X_sample.shape) > 2:
        if fixed_nareas != fixed_X_sample.shape[2]:
            print(f"Warning: nareas ({fixed_nareas}) doesn't match X_sample.shape[2] ({fixed_X_sample.shape[2]}). Using X_sample.shape[2].")
            fixed_nareas = fixed_X_sample.shape[2]
    
    # Ensure regions list has enough elements
    if len(fixed_regions) < fixed_nareas:
        print(f"Warning: regions list has {len(fixed_regions)} elements, but nareas={fixed_nareas}")
        # Extend the regions list with generic names
        fixed_regions.extend([f"Region_{i+1+len(fixed_regions)}" for i in range(fixed_nareas - len(fixed_regions))])
    
    return fixed_model, fixed_X_sample, fixed_regions, fixed_nareas

def analyze_feature_importance(model, X_sample, regions, nareas):
    """
    Analyze feature importance in the LSTM model using a permutation-based approach.
    This method doesn't rely on SHAP and works with any TensorFlow version.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model or None
        Trained LSTM model or None if TensorFlow is not available
    X_sample : numpy.ndarray
        Sample input data
    regions : list
        List of brain region names
    nareas : int
        Number of brain areas
        
    Returns:
    --------
    importance_df : pandas.DataFrame
        DataFrame containing feature importance values
    """
    # Check and fix inputs
    model, X_sample, regions, nareas = check_feature_importance_inputs(model, X_sample, regions, nareas)
    
    # If model is None, return DataFrame with zeros
    if model is None:
        print("No model to analyze. Returning DataFrame with zeros.")
        return pd.DataFrame({
            'Brain Region': [f"Region_{i+1}" for i in range(nareas)],
            'Importance': np.zeros(nareas)
        }).sort_values(by='Importance', ascending=False)
    
    # Ensure regions list has enough elements
    if len(regions) < nareas:
        print(f"Warning: regions list has {len(regions)} elements, but nareas={nareas}")
        # Extend the regions list with generic names
        extended_regions = list(regions) + [f"Region_{i+1+len(regions)}" for i in range(nareas - len(regions))]
    else:
        extended_regions = regions
    
    try:
        # Create a smaller sample if X_sample is large
        sample_size = min(100, X_sample.shape[0])
        X_small = X_sample[:sample_size]
        
        print(f"Calculating permutation importance using {sample_size} samples...")
        
        # Get the baseline predictions
        baseline_preds = model.predict(X_small, verbose=0)
        baseline_preds_class = np.argmax(baseline_preds, axis=1)
        
        # Initialize importance scores
        importance_scores = np.zeros(nareas)
        
        # For each feature (brain region), permute its values and measure the drop in performance
        for i in range(nareas):
            # Create a copy of the data
            X_permuted = X_small.copy()
            
            # Permute the values for this feature across all samples and time steps
            permuted_values = X_permuted[:, :, i].copy()
            np.random.shuffle(permuted_values)
            X_permuted[:, :, i] = permuted_values
            
            # Get predictions with the permuted feature
            permuted_preds = model.predict(X_permuted, verbose=0)
            permuted_preds_class = np.argmax(permuted_preds, axis=1)
            
            # Calculate the drop in performance (importance)
            # Higher drop means more important feature
            baseline_acc = accuracy_score(baseline_preds_class, baseline_preds_class)  # Always 1.0
            permuted_acc = accuracy_score(baseline_preds_class, permuted_preds_class)
            importance = baseline_acc - permuted_acc
            
            # Store the importance score
            importance_scores[i] = importance
            
            # Use the extended_regions list to avoid index errors
            region_name = extended_regions[i] if i < len(extended_regions) else f"Region_{i+1}"
            print(f"Region {i+1}/{nareas} ({region_name}): Importance = {importance:.4f}")
        
        # Create a DataFrame
        importance_df = pd.DataFrame({
            'Brain Region': extended_regions[:nareas],
            'Importance': importance_scores
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        return importance_df
        
    except Exception as e:
        print(f"Error with permutation importance: {e}")
        print("Falling back to model weights analysis...")
        
        try:
            # Find the first dense layer
            dense_layer = None
            for layer in model.layers:
                if 'dense' in layer.name.lower() and layer.name != model.layers[-1].name:
                    dense_layer = layer
                    break
            
            if dense_layer is None:
                # Try to find LSTM layer if dense layer not found
                for layer in model.layers:
                    if 'lstm' in layer.name.lower():
                        dense_layer = layer
                        break
            
            if dense_layer is not None:
                # Get the weights
                weights = dense_layer.get_weights()[0]
                
                # For LSTM layers, reshape if needed
                if 'lstm' in dense_layer.name.lower() and len(weights.shape) > 2:
                    weights = weights.reshape(-1, weights.shape[-1])
                
                # Calculate importance as the mean absolute weight value
                importance = np.mean(np.abs(weights), axis=1)
                
                # Make sure we have the right number of importance values
                if len(importance) >= nareas:
                    importance = importance[:nareas]
                else:
                    # If we don't have enough values, pad with zeros
                    padded_importance = np.zeros(nareas)
                    padded_importance[:len(importance)] = importance
                    importance = padded_importance
                
                # Create a DataFrame using the extended regions list
                importance_df = pd.DataFrame({
                    'Brain Region': extended_regions[:nareas],
                    'Importance': importance
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values(by='Importance', ascending=False)
                
                return importance_df
            else:
                raise ValueError("Could not find appropriate layer for feature importance")
                
        except Exception as nested_e:
            print(f"Error with fallback approach: {nested_e}")
            print("Returning DataFrame with generic region names and zeros.")
            
            # Create a DataFrame with generic region names
            return pd.DataFrame({
                'Brain Region': [f"Region_{i+1}" for i in range(nareas)],
                'Importance': np.zeros(nareas)
            }).sort_values(by='Importance', ascending=False)

def plot_feature_importance(importance_df):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        DataFrame containing feature importance values
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Check if the DataFrame is valid
    if importance_df is None or not isinstance(importance_df, pd.DataFrame):
        print("Invalid DataFrame provided.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Invalid feature importance data", 
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Check if the DataFrame has the required columns
    required_columns = ['Brain Region', 'Importance']
    if not all(col in importance_df.columns for col in required_columns):
        print(f"DataFrame is missing required columns: {required_columns}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "DataFrame is missing required columns", 
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Check if the DataFrame is empty or contains only zeros
    if importance_df.empty or (importance_df['Importance'] == 0).all():
        print("No meaningful feature importance to plot.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No meaningful feature importance data available", 
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Create a horizontal bar plot
    fig, ax = plt.subplots(figsize=(12, max(6, len(importance_df) * 0.4)))
    
    # Sort by importance
    df_sorted = importance_df.sort_values(by='Importance', ascending=True)
    
    # Get the brain regions and importance values
    regions = df_sorted['Brain Region'].values
    importance = df_sorted['Importance'].values
    
    # Create a colormap based on importance values
    max_importance = importance.max() if importance.max() > 0 else 1.0
    colors = plt.cm.viridis(importance / max_importance)
    
    # Create the horizontal bar plot
    bars = ax.barh(regions, importance, color=colors)
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, max_importance))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Normalized Importance')
    
    # Add labels and title
    ax.set_xlabel('Feature Importance')
    ax.set_title('Brain Region Importance')
    
    # Add values to the end of each bar
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_brain_regions_connectivity(connectivity_matrix, area_to_index, brain_groups, region_colors, title="Brain Regions Connectivity"):
    """
    Plot the connectivity matrix between brain regions.
    
    Parameters:
    -----------
    connectivity_matrix : numpy.ndarray
        Matrix of connectivity values between brain regions
    area_to_index : dict
        Dictionary mapping brain areas to indices
    brain_groups : list
        List of lists containing brain regions grouped by function
    region_colors : list
        List of colors for each brain region group
    title : str, optional
        Title for the plot
    """
    plt.figure(figsize=(12, 10))
    plt.imshow(connectivity_matrix, cmap="viridis", aspect="auto")
    
    # Create a reverse mapping from index to area
    index_to_area = {v: k for k, v in area_to_index.items()}
    
    # Add tick labels
    labels = [index_to_area.get(i, f"Unknown_{i}") for i in range(len(area_to_index))]
    plt.xticks(np.arange(len(area_to_index)), labels, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(area_to_index)), labels, fontsize=8)
    
    # Color the tick labels by region
    xtickobjs = plt.gca().get_xticklabels()
    ytickobjs = plt.gca().get_yticklabels()
    
    # Set default color for all labels
    for xlabel, ylabel in zip(xtickobjs, ytickobjs):
        xlabel.set_color('gray')  # Default color for unknown areas
        ylabel.set_color('gray')  # Default color for unknown areas
    
    # Set specific colors for known areas
    for i, area_name in enumerate(labels):
        for group_idx, group in enumerate(brain_groups):
            if area_name in group:
                xtickobjs[i].set_color(region_colors[group_idx])
                ytickobjs[i].set_color(region_colors[group_idx])
                break
    
    plt.title(title)
    plt.colorbar(label="Connectivity Strength")
    plt.tight_layout()
    plt.show() 