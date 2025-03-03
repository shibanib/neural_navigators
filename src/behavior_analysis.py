import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class BehaviorAnalyzer:
    """Class for analyzing behavioral data and neural correlates of behavior."""
    
    def __init__(self):
        pass

    def compute_choice_probability(self, firing_rates: np.ndarray, 
                                 choices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute choice probability for each neuron.
        
        Args:
            firing_rates: Array of shape (n_neurons, n_trials)
            choices: Binary array of choices for each trial
            
        Returns:
            tuple: (choice_probs, p_values)
        """
        n_neurons = firing_rates.shape[0]
        choice_probs = np.zeros(n_neurons)
        p_values = np.zeros(n_neurons)
        
        for i in range(n_neurons):
            # Split data by choice
            rates_choice1 = firing_rates[i, choices == 1]
            rates_choice0 = firing_rates[i, choices == 0]
            
            # Compute ROC curve
            thresholds = np.sort(np.concatenate([rates_choice0, rates_choice1]))
            true_pos = np.zeros(len(thresholds))
            false_pos = np.zeros(len(thresholds))
            
            for j, threshold in enumerate(thresholds):
                true_pos[j] = np.mean(rates_choice1 > threshold)
                false_pos[j] = np.mean(rates_choice0 > threshold)
            
            # Compute area under ROC curve
            choice_probs[i] = np.trapz(true_pos, false_pos)
            
            # Compute p-value using permutation test
            n_perms = 1000
            perm_areas = np.zeros(n_perms)
            all_rates = np.concatenate([rates_choice0, rates_choice1])
            
            for k in range(n_perms):
                np.random.shuffle(all_rates)
                perm_choice0 = all_rates[:len(rates_choice0)]
                perm_choice1 = all_rates[len(rates_choice0):]
                
                true_pos_perm = np.zeros(len(thresholds))
                false_pos_perm = np.zeros(len(thresholds))
                
                for j, threshold in enumerate(thresholds):
                    true_pos_perm[j] = np.mean(perm_choice1 > threshold)
                    false_pos_perm[j] = np.mean(perm_choice0 > threshold)
                
                perm_areas[k] = np.trapz(true_pos_perm, false_pos_perm)
            
            p_values[i] = np.mean(perm_areas >= choice_probs[i])
        
        return choice_probs, p_values

    def predict_choice(self, neural_data: np.ndarray, 
                      choices: np.ndarray) -> Tuple[float, LogisticRegression]:
        """
        Train a logistic regression model to predict choices from neural data.
        
        Args:
            neural_data: Array of shape (n_trials, n_features)
            choices: Binary array of choices for each trial
            
        Returns:
            tuple: (accuracy, model)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            neural_data, choices, test_size=0.2, random_state=42
        )
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        accuracy = model.score(X_test, y_test)
        
        return accuracy, model

    def analyze_reaction_times(self, reaction_times: np.ndarray, 
                             conditions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze reaction times across different conditions.
        
        Args:
            reaction_times: Array of reaction times
            conditions: Dictionary of condition labels for each trial
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Basic statistics
        results['mean'] = np.mean(reaction_times)
        results['median'] = np.median(reaction_times)
        results['std'] = np.std(reaction_times)
        
        # Analyze by condition
        for condition, labels in conditions.items():
            unique_labels = np.unique(labels)
            condition_stats = {
                'means': [],
                'sems': [],
                'labels': unique_labels
            }
            
            for label in unique_labels:
                mask = labels == label
                condition_stats['means'].append(np.mean(reaction_times[mask]))
                condition_stats['sems'].append(
                    np.std(reaction_times[mask]) / np.sqrt(np.sum(mask))
                )
            
            results[condition] = condition_stats
        
        return results

    def compute_sequential_effects(self, choices: np.ndarray, 
                                 outcomes: np.ndarray) -> Dict[str, float]:
        """
        Analyze how previous trials affect current choice.
        
        Args:
            choices: Binary array of choices
            outcomes: Binary array of outcomes (correct/incorrect)
            
        Returns:
            Dictionary containing sequential effect measures
        """
        n_trials = len(choices)
        results = {
            'stay_after_reward': 0,
            'switch_after_error': 0,
            'total_reward_trials': 0,
            'total_error_trials': 0
        }
        
        for i in range(1, n_trials):
            if outcomes[i-1] == 1:  # Previous trial was correct
                results['total_reward_trials'] += 1
                if choices[i] == choices[i-1]:  # Stayed with same choice
                    results['stay_after_reward'] += 1
            else:  # Previous trial was error
                results['total_error_trials'] += 1
                if choices[i] != choices[i-1]:  # Switched choice
                    results['switch_after_error'] += 1
        
        # Convert to probabilities
        if results['total_reward_trials'] > 0:
            results['stay_after_reward'] /= results['total_reward_trials']
        if results['total_error_trials'] > 0:
            results['switch_after_error'] /= results['total_error_trials']
        
        return results 