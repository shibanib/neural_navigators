# Steinmetz Dataset Analysis Dashboard

An interactive web application for analyzing and visualizing the Steinmetz neural dataset. This dashboard allows users to select sessions, run various analyses, and generate visualizations to explore neural activity patterns.

## Features

- **Interactive Analysis Selection**: Choose which sessions and analysis types to run
- **Multiple Analysis Types**: Spike train, LFP, population dynamics, behavioral, and cross-regional analyses
- **Customizable Parameters**: Configure analysis parameters through an intuitive interface
- **Dynamic Visualizations**: View results through interactive plots
- **Multiple View Modes**: Individual, summary, and comparison visualizations
- **Export Options**: Download analysis results and summary reports

## Installation

1. Clone this repository or copy the webapp folder to your local machine
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Make sure you have the Steinmetz dataset files in the appropriate location
2. Start the web server:
   ```
   python app.py
   ```
3. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```
4. Use the interface to:
   - Select sessions to analyze
   - Choose analysis types
   - Configure analysis parameters
   - Run analyses and view results
   - Export results as needed

## Directory Structure

```
webapp/
├── app.py                    # Main Flask application
├── controllers/              # Backend controllers
│   ├── data_controller.py    # Data loading and management
│   ├── analysis_controller.py  # Analysis operations
│   └── visualization_controller.py  # Visualization generation
├── static/                   # Static assets
│   ├── css/                  # CSS stylesheets
│   └── js/                   # JavaScript files
├── templates/                # HTML templates
│   ├── base.html             # Base template
│   └── index.html            # Main dashboard template
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Dependencies

- Flask: Web framework
- NumPy & SciPy: Numerical computations
- Matplotlib & Seaborn: Visualization
- Scikit-learn: Machine learning tools
- Bootstrap: Frontend styling

## How It Works

1. The dashboard loads available sessions from the SteinmetzDataLoader
2. When a user selects sessions and analyses and clicks "Run Analysis":
   - The analysis pipeline processes the selected sessions
   - Results are returned to the frontend
   - Visualizations are generated based on the results
3. Users can switch between different visualization modes:
   - Individual: View results for a single session
   - Summary: View a dashboard summary for a session
   - Comparison: Compare results across multiple sessions

## Extending

To add new analysis types:
1. Add a new analysis method in `controllers/analysis_controller.py`
2. Register the method in the `available_analyses` dictionary
3. Add a corresponding visualization method in `controllers/visualization_controller.py`
4. Update the frontend to display the new analysis type

## License

This project is available for research and educational purposes.

## Acknowledgments

Based on the Steinmetz dataset and analysis notebooks. 