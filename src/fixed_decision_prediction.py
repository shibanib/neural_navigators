def prepare_neural_decision_data(session_data, time_window=(-0.1, 0.3)):
    """Extract neural features and decision data for prediction models"""
    import numpy as np
    
    # For demonstration, we'll use synthetic decision data
    # In a real implementation, you would extract actual decision data from the session
    
    # 1. Extract neural features during the decision period
    # For simplicity, we'll use spike counts in our regions of interest
    
    # Get spikes data
    spikes = session_data.get('spikes', [])
    
    # Check if spikes is empty or not properly structured
    if not isinstance(spikes, (list, np.ndarray)) or len(spikes) == 0:
        print("No spike data available")
        return None, None
    
    # For simplicity, create a synthetic binary decision variable
    # In reality, you would extract this from the behavioral data
    n_trials = len(spikes[0]) if len(spikes) > 0 else 0
    if n_trials == 0:
        print("No trials found")
        return None, None
    
    # Create synthetic decisions (0 or 1 for each trial)
    np.random.seed(42)  # For reproducibility
    decisions = np.random.randint(0, 2, size=n_trials)
    
    # Feature matrix: trials x neurons
    # We'll use average firing rate in the time window as features
    n_neurons = len(spikes)
    features = np.zeros((n_trials, n_neurons))
    
    for i in range(n_neurons):
        for j in range(n_trials):
            # Get spikes for this neuron in this trial
            trial_spikes = spikes[i][j]
            # Count spikes in the time window
            window_spikes = [spike for spike in trial_spikes if time_window[0] <= spike <= time_window[1]]
            # Calculate firing rate
            window_duration = time_window[1] - time_window[0]
            firing_rate = len(window_spikes) / window_duration if window_duration > 0 else 0
            features[j, i] = firing_rate
    
    return features, decisions


def train_decision_model(features, decisions, test_size=0.3):
    """Train a model to predict decisions from neural activity"""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, decisions, test_size=test_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'SVM': SVC(kernel='linear', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        # Test model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
    
    return results 