import os
import sys
sys.path.append('../src')

from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import inspect
import ast

# Import analysis modules
from controllers.data_controller import DataController
from controllers.analysis_controller import AnalysisController
from controllers.visualization_controller import VisualizationController

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'steinmetz-dashboard-secret'

# Initialize controllers
data_controller = DataController()
analysis_controller = AnalysisController()
viz_controller = VisualizationController()

# Whitelist of directories from which code snippets can be served
# Paths are relative to the application root (neural_navigators-main)
ALLOWED_CODE_PATHS = ['webapp/controllers', 'src']

@app.route('/')
def index():
    """Render the main dashboard page"""
    # Get available sessions
    available_sessions = data_controller.get_available_sessions()
    # Get available analyses
    available_analyses = analysis_controller.get_available_analyses()
    
    return render_template('index.html', 
                          sessions=available_sessions,
                          analyses=available_analyses)

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    """Run selected analyses on selected sessions"""
    data = request.get_json()
    session_indices = data.get('sessions', [])
    selected_analyses = data.get('analyses', [])
    custom_config = data.get('config', {})
    
    # Run the analysis
    results = analysis_controller.run_analyses(
        session_indices=session_indices,
        analyses=selected_analyses,
        config=custom_config
    )
    
    # Return the results
    return jsonify(results)

@app.route('/visualize', methods=['POST'])
def visualize():
    """Generate visualizations based on analysis results"""
    data = request.get_json()
    viz_type = data.get('type', 'summary')
    results = data.get('results', {})
    
    # Generate visualization
    viz_data = viz_controller.generate_visualization(
        viz_type=viz_type,
        results=results
    )
    
    # Return the visualization data
    return jsonify(viz_data)

@app.route('/summary_report', methods=['POST'])
def summary_report():
    """Generate a summary report of multiple analyses"""
    data = request.get_json()
    results = data.get('results', {})
    
    # Generate summary report
    report = analysis_controller.generate_summary_report(results)
    
    # Return the report
    return jsonify(report)

@app.route('/documentation')
def documentation_page():
    """Render the documentation page."""
    # We can pass a list of available doc files/topics to the template later if needed
    return render_template('documentation.html')

@app.route('/get_code_snippet', methods=['POST'])
def get_code_snippet():
    data = request.get_json()
    file_path_rel = data.get('file_path') # e.g., webapp/controllers/data_controller.py
    entity_name = data.get('entity_name')   # e.g., DataController.load_session or load_session
    entity_type = data.get('entity_type', 'function') # function, method, class

    if not file_path_rel or not entity_name:
        return jsonify({'error': 'Missing file_path or entity_name'}), 400

    # Security check: Ensure the requested file is within an allowed directory
    app_root = os.path.dirname(os.path.abspath(__file__)).replace('/webapp','') # Get project root
    absolute_file_path = os.path.normpath(os.path.join(app_root, file_path_rel))
    
    is_allowed = False
    for allowed_dir_rel in ALLOWED_CODE_PATHS:
        allowed_abs_dir = os.path.normpath(os.path.join(app_root, allowed_dir_rel))
        if absolute_file_path.startswith(allowed_abs_dir):
            is_allowed = True
            break
    
    if not is_allowed or not os.path.isfile(absolute_file_path):
        app.logger.warning(f"Denied access or file not found for code snippet: {file_path_rel}")
        return jsonify({'error': 'Access denied or file not found.'}), 403

    try:
        with open(absolute_file_path, 'r') as f:
            source_code = f.read()
        
        module_ast = ast.parse(source_code, filename=absolute_file_path)
        target_node = None
        class_name = None
        func_name = entity_name

        if '.' in entity_name and (entity_type == 'method' or entity_type == 'class_method'):
            class_name, func_name = entity_name.split('.', 1)

        for node in module_ast.body:
            if class_name and isinstance(node, ast.ClassDef) and node.name == class_name:
                for child_node in node.body:
                    if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and child_node.name == func_name:
                        target_node = child_node
                        break
            elif not class_name and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == entity_name:
                target_node = node
                break
            if target_node: break

        if target_node:
            # Get the lines of source code for this node
            # ast.unparse requires Python 3.9+
            try:
                snippet = ast.unparse(target_node)
            except AttributeError: # Fallback for older Python if ast.unparse is not available
                 # This is a simplified fallback, inspect.getsource(object) would be better if we could import the module
                lines = source_code.splitlines()
                start_line = target_node.lineno -1
                end_line = target_node.end_lineno if hasattr(target_node, 'end_lineno') else start_line + 20 # Guess 20 lines
                snippet = '\n'.join(lines[start_line:end_line])

            return jsonify({'code_snippet': snippet, 'file_path': file_path_rel, 'entity_name': entity_name})
        else:
            return jsonify({'error': f'{entity_type.capitalize()} \'{entity_name}\' not found in {file_path_rel}'}), 404

    except Exception as e:
        app.logger.error(f"Error extracting code snippet for {entity_name} from {file_path_rel}: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Error extracting code snippet: {str(e)}'}), 500

@app.route('/session_summary', methods=['POST'])
def session_summary():
    """Get a summary of data for a specific session."""
    data = request.get_json()
    session_idx = data.get('session_idx')

    if session_idx is None:
        return jsonify({'error': 'session_idx not provided'}), 400
    
    try:
        # Session indices from UI are 1-based, convert to 0-based for DataController
        # Or ensure DataController consistently uses 0-based internally from load_session
        # Currently, data_controller.get_available_sessions() returns 1-19.
        # Let's assume session_idx from the client will be what DataController expects.
        # DataController.load_session expects 0-indexed if using part files.
        # DataController.get_available_sessions is currently list(range(1,20)), which is 1-based.
        # This needs to be consistent. For now, assume UI sends 0-based if it knows the total count (e.g. 0-38)
        # or we adjust here. Let's assume UI sends the exact index DataController needs.
        
        # If sessions are displayed 1-N, and controller expects 0-(N-1)
        # session_idx_0_based = int(session_idx) - 1 
        # For now, assuming session_idx received is already 0-based for part files.
        summary_data = data_controller.get_session_data_summary(int(session_idx))
        app.logger.info(f"Session summary data for session {session_idx} before jsonify: {str(summary_data)[:500]}...") # Log preview
        return jsonify(summary_data)
    except ValueError: # Handles if session_idx can't be int
        return jsonify({'error': 'Invalid session_idx format'}), 400
    except Exception as e:
        # Log the exception for server-side debugging
        app.logger.error(f"Error generating session summary for {session_idx}: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Error generating session summary: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 