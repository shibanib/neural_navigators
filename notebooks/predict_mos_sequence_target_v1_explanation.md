# Neural Navigator: MOs LFP Sequence Prediction (No PCA, Upsampling) - Notebook Analysis

This document details the design choices, logic, data flow, and results interpretation for the `predict_mos_sequence_target_v1.ipynb` Jupyter notebook. The primary goal of this notebook is to predict a sequence of motor output area (MOs) LFP signals (channel-averaged) based on LFP signals from the visual primary cortex (VISp) using an LSTM neural network. This version explicitly disables PCA, implements upsampling for age categories, and predicts a sequence for the target variable Y.

## 1. Overall Workflow

1.  **Configuration:** Set up parameters for data processing, model architecture, and training. PCA is disabled, upsampling is enabled.
2.  **Data Loading:** Load pre-existing processed Steinmetz dataset files (`steinmetz_partX.npz`, `steinmetz_lfp.npz`).
3.  **Data Preprocessing (Cell 5):**
    *   Iterate through experimental sessions.
    *   Extract LFP data for VISp (input features, X) and MOs (target, Y).
    *   Align data to behavioral events (`stim_onset` or `gocue` for X, `response_time` for Y).
    *   Define time windows for X and Y.
    *   **X Features:** Raw LFP time-series from selected VISp channels, padded to uniform length (`max_len_x`).
    *   **Y Target (Sequence):** MOs LFP time-series, first averaged across MOs channels, then padded to uniform length (`max_len_y`). The model predicts this sequence.
    *   Aggregate data from all valid trials.
    *   Scale X features.
4.  **Model Definition (Cell 6):** Define an LSTM-based neural network designed for sequence-to-sequence prediction (outputting a sequence of length `max_len_y`).
5.  **Model Training and Cross-Validation (Cell 7):**
    *   Train the model using GroupKFold cross-validation (grouped by session ID).
    *   **Upsampling:** Within each training fold, the minority age class (Younger/Older) is upsampled to match the sample count of the majority class.
6.  **Evaluation (Cell 8):** Assess model performance using MSE, MAE, and R-squared, considering the sequence nature of the target.
7.  **Results Aggregation & Visualization:** Summarize results and plot example predicted vs. true sequences.

## 2. Key Variables and Design Choices

### Cell 2: Configuration
*   `PROCESSED_DATA_DIR`: Path to the Steinmetz data files (e.g., `../steinmetz_data_downloads`).
*   `TARGET_BRAIN_REGION_X`: 'VISp' (input LFP features).
*   `TARGET_BRAIN_REGION_Y`: 'MOs' (target LFP sequence).
*   `X_TIME_WINDOW_START_POST_EVENT_MS`, `X_TIME_WINDOW_END_POST_EVENT_MS`: Defines the time window for X features relative to `stim_onset` or `gocue` (e.g., 0-500ms post-event).
*   `Y_TIME_WINDOW_BEFORE_RESPONSE_MS`, `Y_TIME_WINDOW_AFTER_RESPONSE_MS`: Defines the time window for the Y target sequence relative to `response_time` (e.g., 100ms before to 100ms after).
*   `APPLY_PCA_X`, `APPLY_PCA_Y`: Both set to `False`. No PCA is used.
*   `APPLY_UPSAMPLING`: Set to `True`. The minority age class in training folds is upsampled.
*   `LSTM_UNITS_1`, `LSTM_UNITS_2`, `DENSE_UNITS_REG`, `DROPOUT_RATE`, `LEARNING_RATE`: Hyperparameters for the LSTM model.

### Cell 5: Data Preprocessing
*   **X Feature Engineering:**
    *   LFP data from `TARGET_BRAIN_REGION_X` (VISp) is extracted for the defined X-window.
    *   If multiple VISp channels are found, they are all kept as separate features. The data for each trial is transposed to `(n_timesteps_x, n_visp_channels)`.
    *   Sequences are padded to `max_len_x` (the maximum number of time steps across trials in a session for the X-window).
    *   `X_final` shape: `(total_trials, max_len_x, n_visp_channels)` after scaling.
*   **Y Target Engineering (Sequence Prediction):**
    *   LFP data from `TARGET_BRAIN_REGION_Y` (MOs) is extracted for the defined Y-window.
    *   This LFP data is first **averaged across all available MOs channels** for each time step.
    *   The resulting channel-averaged sequence is padded to `max_len_y` (the maximum number of time steps across trials in a session for the Y-window).
    *   `Y_final` shape: `(total_trials, max_len_y, 1)` (1 feature: the channel-averaged LFP value at each time step).
*   **Alignment:** X features are aligned to `stim_onset` or `gocue` (if `stim_onset` is unsuitable, e.g., scalar). Y target sequences are aligned to `response_time`.

### Cell 6: LSTM Architecture (Sequence-to-Sequence)
*   **Input Layer:** `Input(shape=(max_len_x, n_visp_channels))` explicitly defines the input shape.
*   **Masking Layer:** Ignores padded values (0.0) in the input sequences.
*   **LSTM Layers:** Two LSTM layers (`LSTM_UNITS_1`, `LSTM_UNITS_2`).
    *   The first LSTM (`lstm_1`) has `return_sequences=True`.
    *   The second LSTM (`lstm_2`) also has `return_sequences=True`. This is crucial because its full sequence output is needed by the subsequent `TimeDistributed` layer to make predictions for each step in the output sequence.
*   **Dropout Layers:** Applied after each LSTM layer for regularization.
*   **Output Layer:** `TimeDistributed(Dense(1, activation='linear'))`.
    *   `TimeDistributed` applies the `Dense` layer independently to each time step of the output from `lstm_2`.
    *   The `Dense` layer has `1` unit (for the single channel-averaged MOs LFP feature) and a `linear` activation for regression.
    *   This structure allows the model to predict an output sequence of length `max_len_x` (the length of the sequence coming out of `lstm_2`).
*   **Model Output vs. Target Mismatch & Resolution (Previous Iteration):**
    *   A previous iteration of this notebook design faced a `ValueError` because the LSTM stack (processing `max_len_x` input timesteps) was naturally outputting sequences of length `max_len_x`, while the target `Y_final` had `max_len_y` timesteps.
    *   The architecture was then adjusted: the final LSTM layer was set to `return_sequences=False`, its output was passed through an intermediate `Dense` layer, then a `Dense` layer outputting `output_timesteps_y * output_features_y` flat features, followed by a `Reshape((output_timesteps_y, output_features_y))` layer. This forced the output sequence length to match `max_len_y`.
    *   **Current Implication (based on results):** The results provided suggest this reshaping strategy is likely in place for the current run, as the model is training without shape mismatch errors on the loss function.

### Cell 7: Model Training
*   **Cross-Validation:** `GroupKFold` by `session_idx` ensures data from the same session does not appear in both training and validation sets of a fold.
*   **Upsampling:** If `APPLY_UPSAMPLING` is `True`, within each training fold:
    *   Identifies the majority and minority age classes.
    *   Uses `sklearn.utils.resample` to upsample (with replacement) the data (X and Y) of the minority class to match the number of samples in the majority class.
    *   The combined (original majority + upsampled minority) training data is then shuffled.
    *   This helps prevent the model from being biased towards the majority class due to imbalanced data.
*   **Early Stopping:** Used to prevent overfitting by monitoring validation loss.

## 3. Interpretation of Provided Results

**Cross-Validation Summary:**
*   **Avg Val MSE: 169.2554 (std: 74.2262)**
    *   The Mean Squared Error is quite high, and the large standard deviation indicates significant performance differences across the 5 folds. This suggests the model's ability to predict the MOs LFP sequence is not stable or consistently good.
*   **Avg Val MAE: 9.0008 (std: 1.8816)**
    *   On average, the model's predictions for the LFP sequence are off by approximately 9 LFP units at each time step. Again, the standard deviation points to variability.
*   **Avg Val R-squared: 0.0158 (std: 0.0447)**
    *   This R-squared value is extremely close to zero. It means that, on average, the model explains only about 0.158% of the variance in the target MOs LFP sequence. This is a very poor fit, indicating the model has learned very little about the relationship between VISp and MOs LFP sequences.
*   **Overall R2 (variance_weighted) on CV data: 0.0267**
    *   When all cross-validation predictions are concatenated and R-squared is calculated (weighted by variance if multiple outputs per step, though here it's 1 feature per step), the value is 0.0267. This is still very low, confirming the model's weak predictive power on unseen data.

**Performance by Age (CV samples):**
*   **Younger (N=1489): MSE=196.2456, MAE=9.3867, R2=0.0403**
    *   For the younger mice (which formed the majority class before potential upsampling), the model shows a slightly positive R-squared (4.03% variance explained), but this is still very low. MSE and MAE are high.
*   **Older (N=659): MSE=120.4255, MAE=8.4727, R2=-0.0302**
    *   For the older mice, the R-squared is negative (-3.02%). This means the model performs *worse* than simply predicting the average MOs LFP sequence for this group. Interestingly, the MSE and MAE are slightly lower for the older group compared to the younger, despite the worse R-squared. This can happen if the variance of the true Y values for the older group is smaller, making it easier to get a lower absolute error even if the model doesn't capture the (potentially small) variance well.

**Overall Implications from Results:**

1.  **Poor Predictive Performance:** The primary takeaway is that the current model configuration (LSTM predicting MOs LFP sequence from VISp LFP sequence, without PCA, with upsampling) is not effective. The R-squared values near zero (or negative) indicate that the model is not learning a meaningful relationship between the input and target sequences.
2.  **Upsampling Impact:** The question of whether upsampling was active needs to be confirmed by looking at the notebook's execution logs (Cell 7 would print messages about upsampling). If it *was* active and these are the results, it suggests that simply balancing the age classes did not resolve the fundamental prediction problem.
3.  **Target Complexity:** Predicting an entire LFP sequence is a much harder task than predicting a single averaged value. The increased MSE/MAE compared to previous attempts (where Y was a single value) reflects this increased difficulty and the model's struggle to capture the temporal dynamics accurately across the sequence.
4.  **Feature-Target Relationship:** There might be a weak or highly non-linear relationship between the raw VISp LFP time-series and the channel-averaged MOs LFP time-series that this specific LSTM architecture cannot easily learn. The information in VISp LFP might not be sufficiently direct or strong to predict the MOs LFP sequence with high fidelity using this approach.
5.  **X Feature Representation:** The effectiveness still hinges on `X_final` having a meaningful number of features (VISp channels). If it was inadvertently averaged to 1 channel, the input information is severely limited.

**Further Considerations/Next Steps (similar to previous, but now in context of sequence prediction):**

*   **Verify `X_final` Shape:** Confirm the number of features (channels) in `X_final` from the output of Cell 5.
*   **Inspect Target Sequences:** Plot some examples of `Y_final` sequences. Are there clear, consistent patterns, or are they highly variable/noisy? This affects predictability.
*   **Feature Engineering for X:** Consider more advanced features from VISp LFP beyond the raw time-series, such as power in different frequency bands *over time* (e.g., spectrograms or time-frequency representations), or measures of inter-channel coherence if multiple VISp channels are used.
*   **Model Architecture:**
    *   The current sequence-to-sequence model (LSTM outputting final state -> Dense -> Reshape) is one approach. An alternative would be a more canonical encoder-decoder LSTM architecture if input and output sequences have different preferred lengths or more complex transformations are needed.
    *   Experiment with model capacity (number of LSTM units, layers).
*   **Simpler Sequence Targets:** Before predicting the full, raw channel-averaged LFP sequence, consider if predicting, for example, the sequence of a few principal components of the MOs LFP (if `APPLY_PCA_Y` were true and adapted for sequences) might be a more tractable intermediate step.
*   **Domain Knowledge:** Are there specific frequency bands or event-related potentials within the MOs LFP sequence that are hypothesized to be related to VISp activity? Targeting these more specific phenomena might be more successful than predicting the entire raw (averaged) waveform.

The results indicate that predicting the MOs LFP *sequence* is challenging with the current setup. Significant improvements would likely require revisiting feature engineering, target definition, or exploring more sophisticated model architectures tailored for complex time-series relationships. 