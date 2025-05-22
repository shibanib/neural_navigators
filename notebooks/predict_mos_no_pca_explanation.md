# Neural Navigator: MOs LFP Prediction (No PCA) - Notebook Analysis

This document details the design choices, logic, and data flow within the `predict_mos_no_pca.ipynb` Jupyter notebook. The primary goal of this notebook is to predict a motor output area (MOs) LFP signal (averaged across channels and time) based on LFP signals from the visual primary cortex (VISp) using an LSTM neural network. **This version of the notebook explicitly disables Principal Component Analysis (PCA) for both input (X) and target (Y) features.**

## 1. Overall Workflow

The notebook follows these general steps:
1.  **Configuration:** Set up parameters for data processing, model architecture, and training. PCA-related parameters are disabled.
2.  **Data Loading:** Load pre-existing processed Steinmetz dataset files.
3.  **Data Preprocessing (Core Logic in Cell 5):**
    *   Iterate through experimental sessions.
    *   Extract LFP data for VISp (input features, X) and MOs (target, Y).
    *   Align data to specific behavioral events (e.g., stimulus onset, gocue, response time).
    *   Define time windows for X and Y.
    *   **No PCA is applied.** Input X features are the raw LFP channels from VISp. The target Y is the MOs LFP averaged across its channels and then averaged over its defined time window.
    *   Handle trial mismatches and missing data.
    *   Aggregate data from all valid trials and sessions.
    *   Scale X features (raw VISp LFP channels).
4.  **Model Definition:** Define an LSTM-based neural network.
5.  **Model Training and Cross-Validation:** Train the model using GroupKFold cross-validation.
6.  **Evaluation:** Assess model performance using MSE, MAE, and R-squared.
7.  **Results Aggregation & Visualization:** Summarize results and plot predictions.

## 2. Cell-by-Cell Breakdown

### Cell 1: Imports
*   **Purpose:** Imports necessary Python libraries (NumPy, scikit-learn, TensorFlow, Matplotlib).
*   Sets random seeds for reproducibility.

### Cell 2: Configuration
*   **Purpose:** Defines global parameters and hyperparameters.
*   **Key Parameters & Design Choices (No PCA):**
    *   `TARGET_BRAIN_REGION_X` ('VISp'), `TARGET_BRAIN_REGION_Y` ('MOs').
    *   Time windows for X (relative to `stim_onset` or `gocue`) and Y (relative to `response_time`).
    *   LSTM model hyperparameters.
    *   `APPLY_PCA_X = False`, `APPLY_PCA_Y = False`: PCA is explicitly turned off. Parameters like `N_PCA_COMPONENTS_X` and `N_PCA_COMPONENTS_Y` are not used.

### Cell 3: Data Loading Functions
*   **Purpose:** Loads Steinmetz dataset files (`alldat.npy`, `dat_LFP.npy`).

### Cell 4: Helper Function - Mouse Details
*   **Purpose:** `get_mouse_details_by_session` extracts mouse metadata and categorizes age.

### Cell 5: Data Preprocessing for MOs LFP Prediction (Crucial Cell - No PCA)
*   **Purpose:** Prepares `X` (input VISp LFP) and `Y` (target MOs LFP) data.
*   **Core Logic (No PCA):**
    1.  **Session Iteration & Metadata:** Standard session loop and metadata checks.
    2.  **X-Alignment Event:** Prioritizes per-trial `stim_onset_times`; falls back to `gocue_times` for X-feature alignment.
    3.  **Trial Synchronization:** Aligns trials across LFP, X-event times, and response times.
    4.  **Channel Selection:** Extracts LFP channels for VISp and MOs.
    5.  **Per-Trial Windowing:** Extracts `x_trial_lfp` (VISp) and `y_trial_lfp` (MOs) based on defined windows.
    6.  **Padding:** Pads LFP sequences for uniform length.
    7.  **X Data Processing (No PCA):**
        *   `X_session_processed` uses the padded `x_trial_lfp` directly. The features are the raw LFP values from all selected VISp channels. Shape per trial: `(n_timesteps_x, n_visp_channels)`.
    8.  **Y Target Definition (No PCA - Critical):**
        *   `Y_session_padded_raw` contains MOs LFP data for the Y-window: `(n_trials_session, max_len_y, n_mos_channels_raw)`.
        *   This is first averaged across MOs channels: `np.mean(Y_session_padded_raw, axis=2)`, resulting in `(n_trials_session, max_len_y)`.
        *   Then, it's averaged across the time steps of the Y-window: `np.mean(Y_avg_channels, axis=1).reshape(-1, 1)`.
        *   The final `Y_session_processed_target` is a single value per trial, representing the grand average of MOs LFP activity (across all its channels and the full Y-time window). Shape: `(1,)` per trial.
    9.  **Aggregation:** Collects processed `X` and `Y` data.
    10. **Global Scaling (X only):** Applies `StandardScaler` to the `X_combined` features (raw VISp LFP channels).

*   **Key Design Points (No PCA):**
    *   Simpler feature pipeline by omitting PCA.
    *   Input X features are multi-channel time-series LFP from VISp.
    *   Target Y is a single scalar value per trial: the overall average MOs LFP in the response-aligned window.

### Cell 6: Model Definition
*   **Purpose:** Defines the `create_lstm_model` function.
*   **Architecture:** Standard LSTM stack (Masking, LSTMs, Dropouts, Dense layers).
    *   Input shape will be `(n_timesteps_x, n_visp_channels)`.
    *   Output dimension of the final `Dense` layer will be `1` (predicting the single averaged MOs LFP value).
*   **Compilation:** Adam optimizer, MSE loss.

### Cell 7: Model Training, Cross-Validation, and Evaluation
*   **Purpose:** Trains and evaluates the LSTM model.
*   **Logic:**
    *   Uses `GroupKFold` (grouped by session ID).
    *   `EarlyStopping` callback.
    *   Model input/output shapes are determined dynamically from `X_final` and `Y_final`.
    *   Standard training loop, prediction, and metric calculation (MSE, MAE, R-squared).

### Cell 8: Results Aggregation and Age-Based Evaluation
*   **Purpose:** Summarizes and visualizes results.
*   **Logic:**
    *   Prints average CV metrics.
    *   Plots validation loss.
    *   Calculates overall R-squared.
    *   Provides performance breakdown by age category.
    *   **True vs. Predicted Plot:** Visualizes `Y_true_all_cv` (true average MOs LFP) against `Y_pred_all_cv` (predicted average MOs LFP). Plot labels reflect "Avg MOs LFP".

## 3. Summary of Key Design Choices & Implications (No PCA Version)

*   **Input (X):** Time-series LFP from all selected VISp channels, aligned to `stim_onset` or `gocue`.
*   **Target (Y):** A single scalar value per trial, representing the channel-averaged and then time-averaged LFP activity in MOs over a window aligned to `response_time`.
*   **Model:** A sequence-to-vector LSTM model.
*   **Evaluation:** Grouped cross-validation by session.
*   **Interpretation of "Flat Line" for True Y:** If the grand average MOs LFP value within the Y-window does not vary much across trials, the "True Avg MOs LFP" plot might still appear relatively flat. This reflects the nature of the defined target variable (an overall average). This version tests if predicting this simpler, aggregated signal is feasible without the intermediate PCA step.

This "No PCA" version simplifies the feature extraction process. The model now directly learns from the time-series of multiple VISp channels to predict a single, heavily averaged MOs LFP value. The "flat line" concern for the target `Y` remains relevant if the averaging process (both across channels and time) significantly reduces trial-to-trial variance in the target signal. 