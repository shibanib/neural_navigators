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

if __name__ == '__main__':
    app.run(debug=True, port=5000) 