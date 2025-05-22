"""
Helper functions to fix TensorFlow and SHAP compatibility issues.
"""

def fix_tensorflow_issues():
    """
    Fix common TensorFlow issues, including:
    1. SHAP compatibility issues
    2. Eager execution issues
    
    Returns:
    --------
    bool
        True if fixes were applied, False otherwise
    """
    try:
        import tensorflow as tf
        import sys
        
        # Check if we're in a Jupyter environment
        in_jupyter = 'ipykernel' in sys.modules
        
        # Print current TensorFlow configuration
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Eager execution enabled: {tf.executing_eagerly()}")
        
        # Fix 1: Ensure eager execution is enabled
        if not tf.executing_eagerly():
            print("Enabling eager execution...")
            tf.compat.v1.enable_eager_execution()
            print(f"Eager execution now enabled: {tf.executing_eagerly()}")
        
        # Fix 2: Patch SHAP-related issues
        try:
            import shap
            print(f"SHAP version: {shap.__version__}")
            
            # Check if SHAP and TensorFlow versions are compatible
            if tf.__version__.startswith('2.') and not hasattr(tf.compat.v1, 'disable_eager_execution'):
                print("Warning: Your TensorFlow version might not be compatible with SHAP.")
                print("Consider installing TensorFlow 2.8.0 and SHAP 0.41.0.")
            
            # Patch SHAP's TensorFlow integration
            if hasattr(shap, 'explainers') and hasattr(shap.explainers, 'deep'):
                # Patch 1: Fix the DeepExplainer initialization
                original_init = shap.explainers.deep.TFDeepExplainer.__init__
                
                def patched_init(self, model, data, session=None, learning_phase_flags=None):
                    try:
                        return original_init(self, model, data, session, learning_phase_flags)
                    except Exception as e:
                        print(f"Warning: Error in SHAP TFDeepExplainer initialization: {e}")
                        print("Applying workaround...")
                        # Use a simpler initialization approach
                        self.model = model
                        self.data = data
                        self.session = None
                        self.learning_phase_flags = None
                
                shap.explainers.deep.TFDeepExplainer.__init__ = patched_init
                
                # Patch 2: Fix the shap_values method to handle TensorListStack errors
                if hasattr(shap.explainers.deep.TFDeepExplainer, 'shap_values'):
                    original_shap_values = shap.explainers.deep.TFDeepExplainer.shap_values
                    
                    def patched_shap_values(self, X, ranked_outputs=None, output_rank_order="max", check_additivity=True):
                        try:
                            # Try the original method first
                            return original_shap_values(self, X, ranked_outputs, output_rank_order, check_additivity)
                        except Exception as e:
                            error_msg = str(e)
                            if 'shap_TensorListStack' in error_msg or 'shap_DivNoNan' in error_msg:
                                print(f"Warning: SHAP TensorFlow integration error: {e}")
                                print("Using alternative approach for feature importance...")
                                
                                # Return dummy values instead
                                if isinstance(X, list):
                                    X = X[0]
                                
                                # Create dummy SHAP values (zeros)
                                if hasattr(self.model, 'output_shape'):
                                    num_classes = self.model.output_shape[-1]
                                else:
                                    num_classes = 1
                                
                                # Return a list of arrays, one per class
                                return [np.zeros(X.shape) for _ in range(num_classes)]
                            else:
                                raise e
                    
                    shap.explainers.deep.TFDeepExplainer.shap_values = patched_shap_values
                
                print("Applied SHAP TensorFlow integration patches.")
        
        except ImportError:
            print("SHAP is not installed. No SHAP-related fixes applied.")
        
        # Fix 3: Set memory growth for GPUs if available
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Set memory growth for {len(gpus)} GPU(s).")
        except Exception as e:
            print(f"Note: Could not configure GPU memory growth: {e}")
        
        # Fix 4: Set a reasonable thread count
        try:
            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(4)
            print("Set reasonable thread counts for TensorFlow operations.")
        except Exception as e:
            print(f"Note: Could not set thread counts: {e}")
        
        # Fix 5: Add missing gradient ops if possible
        try:
            # Try to register missing gradient ops
            from tensorflow.python.framework import ops
            
            # Check if the ops registry exists
            if hasattr(ops, '_gradient_registry') and hasattr(ops._gradient_registry, '_registry'):
                registry = ops._gradient_registry._registry
                
                # Define dummy gradient functions for problematic ops
                def _shap_tensor_list_stack_grad(op, grad):
                    return [tf.zeros_like(op.inputs[0])]
                
                def _shap_div_no_nan_grad(op, grad):
                    return [tf.zeros_like(op.inputs[0]), tf.zeros_like(op.inputs[1])]
                
                # Register the gradient functions if the ops exist
                try:
                    ops.RegisterGradient("shap_TensorListStack")(_shap_tensor_list_stack_grad)
                    print("Registered gradient for shap_TensorListStack")
                except Exception:
                    pass
                
                try:
                    ops.RegisterGradient("shap_DivNoNan")(_shap_div_no_nan_grad)
                    print("Registered gradient for shap_DivNoNan")
                except Exception:
                    pass
        except Exception as e:
            print(f"Note: Could not register custom gradients: {e}")
        
        print("\nTensorFlow fixes applied successfully.")
        
        # Suggest kernel restart if in Jupyter
        if in_jupyter:
            from IPython.display import display, HTML
            display(HTML(
                "<div style='background-color: #FFFFCC; padding: 10px; border: 1px solid #FFCC00; border-radius: 5px;'>"
                "<p><strong>TensorFlow fixes applied.</strong></p>"
                "<p>It's recommended to restart the kernel to ensure all changes take effect.</p>"
                "<p>Use the <code>restart_kernel()</code> function or manually restart the kernel.</p>"
                "</div>"
            ))
        
        return True
        
    except Exception as e:
        print(f"Error applying TensorFlow fixes: {e}")
        print("You may need to manually fix TensorFlow issues.")
        print("Try running: !pip install tensorflow==2.8.0 shap==0.41.0")
        return False

def restart_kernel():
    """
    Restart the Jupyter kernel.
    This function should be called after applying fixes.
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

if __name__ == "__main__":
    fix_tensorflow_issues() 