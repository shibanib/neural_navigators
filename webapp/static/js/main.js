/**
 * Steinmetz Dataset Analysis Dashboard
 * Main JavaScript functionality
 */

// Store global state
let globalState = {
    results: null,
    selectedSession: null,
    selectedAnalyses: []
};

// Initialize dashboard components when document is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');
    
    // Initialize tooltips if using Bootstrap 5
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});

/**
 * Format numbers for display
 * @param {number} value - The number to format
 * @param {number} precision - Number of decimal places
 * @returns {string} Formatted number
 */
function formatNumber(value, precision = 2) {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value !== 'number') return value;
    
    return value.toFixed(precision);
}

/**
 * Download data as a JSON file
 * @param {Object} data - Data to download
 * @param {string} filename - Name of the file
 */
function downloadJson(data, filename) {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", filename);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}

/**
 * Create a simple chart using chart.js if available
 * @param {string} canvasId - ID of the canvas element
 * @param {Object} chartData - Chart data and config
 */
function createChart(canvasId, chartData) {
    if (typeof Chart === 'undefined') {
        console.error('Chart.js not loaded');
        return;
    }
    
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error(`Canvas element with ID ${canvasId} not found`);
        return;
    }
    
    new Chart(canvas, chartData);
}

/**
 * Show a notification to the user
 * @param {string} message - Message to display
 * @param {string} type - Type of notification (success, error, warning, info)
 * @param {number} duration - Duration in milliseconds
 */
function showNotification(message, type = 'info', duration = 3000) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} notification fade-in`;
    notification.innerHTML = message;
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '9999';
    notification.style.minWidth = '300px';
    
    // Add to DOM
    document.body.appendChild(notification);
    
    // Remove after duration
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 500);
    }, duration);
}

/**
 * Toggle the loading spinner
 * @param {boolean} show - Whether to show the spinner
 * @param {string} message - Optional message to display
 */
function toggleLoading(show, message = 'Loading...') {
    const spinner = document.getElementById('loading-spinner');
    if (!spinner) return;
    
    if (show) {
        spinner.style.display = 'block';
        spinner.querySelector('p').textContent = message;
    } else {
        spinner.style.display = 'none';
    }
}

/**
 * Format a timestamp into a readable date string
 * @param {string} timestamp - ISO timestamp
 * @returns {string} Formatted date
 */
function formatDateTime(timestamp) {
    if (!timestamp) return 'N/A';
    
    const date = new Date(timestamp);
    return date.toLocaleString();
}

/**
 * Check if an object is empty
 * @param {Object} obj - Object to check
 * @returns {boolean} True if empty
 */
function isEmptyObject(obj) {
    return Object.keys(obj).length === 0;
}

/**
 * Get a color for a series based on index
 * @param {number} index - Series index
 * @returns {string} Color string
 */
function getSeriesColor(index) {
    const colors = [
        '#4285F4', // Google Blue
        '#EA4335', // Google Red
        '#FBBC05', // Google Yellow
        '#34A853', // Google Green
        '#8A2BE2', // Blue Violet
        '#FF6347', // Tomato
        '#2E8B57', // Sea Green
        '#4682B4', // Steel Blue
        '#D2691E', // Chocolate
        '#9370DB'  // Medium Purple
    ];
    
    return colors[index % colors.length];
}

// Export functions if using modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        formatNumber,
        downloadJson,
        createChart,
        showNotification,
        toggleLoading,
        formatDateTime,
        isEmptyObject,
        getSeriesColor
    };
} 