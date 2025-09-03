"""
AMP Web Interface - Modern Web Application for Multi-Agent Intelligence System

Interactive web interface for adverse media profiling with real-time monitoring.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import asyncio
import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config, APIConfig, SearchConfig
from core.multi_agent_orchestrator import run_multi_agent_analysis, MultiAgentOrchestrator
from core.monitoring import get_system_monitor, get_advanced_logger, initialize_monitoring
from utils.analysis import AnalysisUtils

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'amp_secret_key_change_in_production'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
analysis_sessions = {}
system_monitor = None
advanced_logger = None

def initialize_app():
    """Initialize the web application."""
    global system_monitor, advanced_logger
    
    # Create necessary directories
    Path("static/css").mkdir(parents=True, exist_ok=True)
    Path("static/js").mkdir(parents=True, exist_ok=True)
    Path("templates").mkdir(parents=True, exist_ok=True)
    
    # Initialize monitoring
    system_monitor, advanced_logger = initialize_monitoring()

def load_config() -> Config:
    """Load configuration from environment."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    api_config = APIConfig(
        openai_api_key=os.getenv('OPENAI_API_KEY', ''),
        news_api_key=os.getenv('NEWS_API_KEY', ''),
        serper_api_key=os.getenv('SERPER_API_KEY', '')
    )
    
    search_config = SearchConfig(
        max_articles=int(os.getenv('MAX_ARTICLES', '50')),
        max_age_days=int(os.getenv('MAX_AGE_DAYS', '90'))
    )
    
    return Config(api=api_config, search=search_config)

# Routes
@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/analysis')
def analysis():
    """Analysis page."""
    return render_template('analysis.html')

@app.route('/monitoring')
def monitoring():
    """Monitoring dashboard page."""
    return render_template('monitoring.html')

@app.route('/reports')
def reports():
    """Reports page."""
    reports_dir = Path("data/reports")
    reports = []
    
    if reports_dir.exists():
        for report_file in reports_dir.glob("*.json"):
            reports.append({
                'filename': report_file.name,
                'size': report_file.stat().st_size,
                'modified': datetime.fromtimestamp(report_file.stat().st_mtime).isoformat()
            })
    
    return render_template('reports.html', reports=reports)

@app.route('/api/system/status')
def api_system_status():
    """Get system status."""
    if system_monitor:
        return jsonify(system_monitor.get_system_status())
    return jsonify({'error': 'Monitoring not initialized'})

@app.route('/api/analysis/start', methods=['POST'])
def api_start_analysis():
    """Start a new analysis."""
    data = request.get_json()
    target = data.get('target')
    
    if not target:
        return jsonify({'error': 'Target is required'}), 400
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Store session info
    analysis_sessions[session_id] = {
        'target': target,
        'status': 'starting',
        'start_time': datetime.now().isoformat(),
        'progress': 0,
        'current_phase': 'Initializing',
        'results': None,
        'error': None
    }
    
    # Start analysis in background
    config = load_config()
    additional_info = data.get('additional_info', {})
    
    def run_analysis():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Update status
            analysis_sessions[session_id]['status'] = 'running'
            socketio.emit('analysis_update', {
                'session_id': session_id,
                'status': 'running',
                'progress': 10,
                'phase': 'Data Collection'
            })
            
            # Run analysis
            result = loop.run_until_complete(
                run_multi_agent_analysis(target, config, additional_info)
            )
            
            # Update session with results
            analysis_sessions[session_id].update({
                'status': 'completed',
                'progress': 100,
                'current_phase': 'Completed',
                'results': result,
                'end_time': datetime.now().isoformat()
            })
            
            socketio.emit('analysis_update', {
                'session_id': session_id,
                'status': 'completed',
                'progress': 100,
                'phase': 'Completed',
                'results': result
            })
            
        except Exception as e:
            analysis_sessions[session_id].update({
                'status': 'error',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
            
            socketio.emit('analysis_update', {
                'session_id': session_id,
                'status': 'error',
                'error': str(e)
            })
    
    # Start analysis thread
    thread = threading.Thread(target=run_analysis, daemon=True)
    thread.start()
    
    return jsonify({
        'session_id': session_id,
        'status': 'started'
    })

@app.route('/api/analysis/<session_id>/status')
def api_analysis_status(session_id):
    """Get analysis status."""
    if session_id in analysis_sessions:
        return jsonify(analysis_sessions[session_id])
    return jsonify({'error': 'Session not found'}), 404

@app.route('/api/reports/<filename>')
def api_get_report(filename):
    """Download a report file."""
    reports_dir = Path("data/reports")
    return send_from_directory(reports_dir, filename)

@app.route('/api/performance/<agent_id>')
def api_agent_performance(agent_id):
    """Get agent performance statistics."""
    if advanced_logger:
        return jsonify(advanced_logger.get_performance_stats(agent_id))
    return jsonify({'error': 'Logger not initialized'})

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Client connected: {request.sid}")
    emit('connected', {'status': 'Connected to AMP system'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"Client disconnected: {request.sid}")

@socketio.on('join_monitoring')
def handle_join_monitoring():
    """Join monitoring room for real-time updates."""
    join_room('monitoring')
    emit('joined_monitoring', {'status': 'Joined monitoring room'})

@socketio.on('leave_monitoring')  
def handle_leave_monitoring():
    """Leave monitoring room."""
    leave_room('monitoring')

# Background tasks
def background_monitoring():
    """Background task for real-time monitoring updates."""
    while True:
        if system_monitor:
            status = system_monitor.get_system_status()
            socketio.emit('monitoring_update', status, room='monitoring')
        socketio.sleep(5)  # Update every 5 seconds

# HTML Templates (embedded for simplicity)
def create_templates():
    """Create HTML templates."""
    
    # Base template
    base_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{% block title %}AMP - Adverse Media Profiler{% endblock %}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.0/socket.io.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            .sidebar { min-height: 100vh; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .content { background-color: #f8f9fa; min-height: 100vh; }
            .card { border: none; box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075); }
            .metric-card { transition: transform 0.2s; }
            .metric-card:hover { transform: translateY(-2px); }
            .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
            .status-online { background-color: #28a745; }
            .status-offline { background-color: #dc3545; }
            .status-warning { background-color: #ffc107; }
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <div class="row">
                <!-- Sidebar -->
                <div class="col-md-2 sidebar p-0">
                    <div class="d-flex flex-column">
                        <div class="text-center py-4">
                            <h4 class="text-white">🔍 AMP</h4>
                            <p class="text-white-50">Multi-Agent Intelligence</p>
                        </div>
                        <nav class="nav flex-column">
                            <a class="nav-link text-white" href="/"><i class="fas fa-tachometer-alt me-2"></i>Dashboard</a>
                            <a class="nav-link text-white" href="/analysis"><i class="fas fa-search me-2"></i>Analysis</a>
                            <a class="nav-link text-white" href="/monitoring"><i class="fas fa-desktop me-2"></i>Monitoring</a>
                            <a class="nav-link text-white" href="/reports"><i class="fas fa-file-alt me-2"></i>Reports</a>
                        </nav>
                    </div>
                </div>
                
                <!-- Main Content -->
                <div class="col-md-10 content p-4">
                    {% block content %}{% endblock %}
                </div>
            </div>
        </div>
        
        {% block scripts %}{% endblock %}
    </body>
    </html>
    '''
    
    # Dashboard template
    dashboard_template = '''
    {% extends "base.html" %}
    
    {% block title %}Dashboard - AMP{% endblock %}
    
    {% block content %}
    <div class="d-flex justify-content-between align-items-center mb-4">
        <h1>System Dashboard</h1>
        <div class="d-flex align-items-center">
            <span class="status-indicator status-online me-2"></span>
            <span>System Online</span>
        </div>
    </div>
    
    <div class="row mb-4">
        <div class="col-md-3">
            <div class="card metric-card">
                <div class="card-body text-center">
                    <i class="fas fa-robot fa-2x text-primary mb-2"></i>
                    <h5>Active Agents</h5>
                    <h3 id="active-agents">5</h3>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card metric-card">
                <div class="card-body text-center">
                    <i class="fas fa-tasks fa-2x text-success mb-2"></i>
                    <h5>Completed Tasks</h5>
                    <h3 id="completed-tasks">0</h3>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card metric-card">
                <div class="card-body text-center">
                    <i class="fas fa-exclamation-triangle fa-2x text-warning mb-2"></i>
                    <h5>Active Alerts</h5>
                    <h3 id="active-alerts">0</h3>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card metric-card">
                <div class="card-body text-center">
                    <i class="fas fa-chart-line fa-2x text-info mb-2"></i>
                    <h5>Success Rate</h5>
                    <h3 id="success-rate">100%</h3>
                </div>
            </div>
        </div>
    </div>
    
    <div class="row">
        <div class="col-md-8">
            <div class="card">
                <div class="card-header">
                    <h5>System Performance</h5>
                </div>
                <div class="card-body">
                    <canvas id="performanceChart" height="100"></canvas>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card">
                <div class="card-header">
                    <h5>Recent Activity</h5>
                </div>
                <div class="card-body">
                    <div id="recent-activity">
                        <div class="d-flex justify-content-center">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    {% endblock %}
    
    {% block scripts %}
    <script>
        const socket = io();
        
        // Initialize dashboard
        function updateDashboard() {
            fetch('/api/system/status')
                .then(response => response.json())
                .then(data => {
                    // Update metrics
                    const agentCount = Object.keys(data.agent_health || {}).length;
                    document.getElementById('active-agents').textContent = agentCount;
                    
                    const alerts = data.active_alerts || [];
                    document.getElementById('active-alerts').textContent = alerts.length;
                })
                .catch(error => console.error('Error updating dashboard:', error));
        }
        
        // Initialize chart
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'CPU Usage',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }, {
                    label: 'Memory Usage',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
        
        // Update dashboard every 5 seconds
        updateDashboard();
        setInterval(updateDashboard, 5000);
        
        // Socket events
        socket.on('connect', function() {
            console.log('Connected to server');
            socket.emit('join_monitoring');
        });
        
        socket.on('monitoring_update', function(data) {
            // Update chart with new data
            const now = new Date().toLocaleTimeString();
            
            chart.data.labels.push(now);
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
            }
            
            const cpuUsage = data.system_metrics?.cpu?.latest || 0;
            const memoryUsage = data.system_metrics?.memory?.latest || 0;
            
            chart.data.datasets[0].data.push(cpuUsage);
            chart.data.datasets[1].data.push(memoryUsage);
            
            if (chart.data.datasets[0].data.length > 20) {
                chart.data.datasets[0].data.shift();
                chart.data.datasets[1].data.shift();
            }
            
            chart.update('none');
        });
    </script>
    {% endblock %}
    '''
    
    # Analysis template
    analysis_template = '''
    {% extends "base.html" %}
    
    {% block title %}Analysis - AMP{% endblock %}
    
    {% block content %}
    <h1>Adverse Media Analysis</h1>
    
    <div class="row">
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">
                    <h5>Start New Analysis</h5>
                </div>
                <div class="card-body">
                    <form id="analysisForm">
                        <div class="mb-3">
                            <label for="target" class="form-label">Target Entity</label>
                            <input type="text" class="form-control" id="target" required>
                            <div class="form-text">Enter the name of the person or organization to analyze</div>
                        </div>
                        <div class="mb-3">
                            <label for="targetId" class="form-label">Target ID (Optional)</label>
                            <input type="text" class="form-control" id="targetId">
                            <div class="form-text">CNPJ, CPF, or other identifier</div>
                        </div>
                        <button type="submit" class="btn btn-primary">
                            <i class="fas fa-search me-2"></i>Start Analysis
                        </button>
                    </form>
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <div class="card">
                <div class="card-header">
                    <h5>Analysis Progress</h5>
                </div>
                <div class="card-body">
                    <div id="analysisProgress" style="display: none;">
                        <div class="progress mb-3">
                            <div id="progressBar" class="progress-bar" role="progressbar" style="width: 0%"></div>
                        </div>
                        <p id="currentPhase">Initializing...</p>
                        <p id="analysisTarget"></p>
                    </div>
                    <div id="noAnalysis" class="text-center text-muted">
                        No active analysis
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="row mt-4" id="resultsSection" style="display: none;">
        <div class="col-12">
            <div class="card">
                <div class="card-header">
                    <h5>Analysis Results</h5>
                </div>
                <div class="card-body">
                    <div id="analysisResults"></div>
                </div>
            </div>
        </div>
    </div>
    {% endblock %}
    
    {% block scripts %}
    <script>
        const socket = io();
        let currentSessionId = null;
        
        document.getElementById('analysisForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const target = document.getElementById('target').value;
            const targetId = document.getElementById('targetId').value;
            
            const data = {
                target: target,
                additional_info: targetId ? {id: targetId} : {}
            };
            
            fetch('/api/analysis/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                if (data.session_id) {
                    currentSessionId = data.session_id;
                    showAnalysisProgress(target);
                }
            })
            .catch(error => {
                console.error('Error starting analysis:', error);
                alert('Failed to start analysis');
            });
        });
        
        function showAnalysisProgress(target) {
            document.getElementById('noAnalysis').style.display = 'none';
            document.getElementById('analysisProgress').style.display = 'block';
            document.getElementById('analysisTarget').textContent = `Analyzing: ${target}`;
        }
        
        function updateProgress(progress, phase) {
            const progressBar = document.getElementById('progressBar');
            const currentPhase = document.getElementById('currentPhase');
            
            progressBar.style.width = `${progress}%`;
            progressBar.textContent = `${progress}%`;
            currentPhase.textContent = phase;
        }
        
        function showResults(results) {
            const resultsSection = document.getElementById('resultsSection');
            const analysisResults = document.getElementById('analysisResults');
            
            // Create results summary
            const metadata = results.metadata || {};
            const riskAssessment = results.risk_assessment || {};
            
            let html = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Target Information</h6>
                        <p><strong>Name:</strong> ${metadata.target?.name || 'N/A'}</p>
                        <p><strong>Analysis ID:</strong> ${metadata.target?.analysis_id || 'N/A'}</p>
                        <p><strong>Generated:</strong> ${metadata.generated_at || 'N/A'}</p>
                    </div>
                    <div class="col-md-6">
                        <h6>Risk Assessment</h6>
                        <p><strong>Overall Score:</strong> ${(riskAssessment.overall_risk_score || 0).toFixed(2)}</p>
            `;
            
            if (results.recommendations && results.recommendations.length > 0) {
                html += '<h6>Recommendations</h6><ul>';
                results.recommendations.slice(0, 3).forEach(rec => {
                    html += `<li>${rec}</li>`;
                });
                html += '</ul>';
            }
            
            html += '</div></div>';
            
            analysisResults.innerHTML = html;
            resultsSection.style.display = 'block';
        }
        
        // Socket events
        socket.on('analysis_update', function(data) {
            if (data.session_id === currentSessionId) {
                if (data.progress) {
                    updateProgress(data.progress, data.phase || 'Processing...');
                }
                
                if (data.status === 'completed' && data.results) {
                    showResults(data.results);
                }
                
                if (data.status === 'error') {
                    alert(`Analysis failed: ${data.error}`);
                    document.getElementById('analysisProgress').style.display = 'none';
                    document.getElementById('noAnalysis').style.display = 'block';
                }
            }
        });
    </script>
    {% endblock %}
    '''
    
    # Save templates
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    with open(templates_dir / "base.html", "w", encoding="utf-8") as f:
        f.write(base_template)
    
    with open(templates_dir / "dashboard.html", "w", encoding="utf-8") as f:
        f.write(dashboard_template)
    
    with open(templates_dir / "analysis.html", "w", encoding="utf-8") as f:
        f.write(analysis_template)
    
    # Simple monitoring and reports templates
    with open(templates_dir / "monitoring.html", "w", encoding="utf-8") as f:
        f.write('''
        {% extends "base.html" %}
        {% block title %}Monitoring - AMP{% endblock %}
        {% block content %}
        <h1>System Monitoring</h1>
        <div id="monitoring-data">
            <div class="text-center">
                <div class="spinner-border" role="status"></div>
                <p>Loading monitoring data...</p>
            </div>
        </div>
        {% endblock %}
        ''')
    
    with open(templates_dir / "reports.html", "w", encoding="utf-8") as f:
        f.write('''
        {% extends "base.html" %}
        {% block title %}Reports - AMP{% endblock %}
        {% block content %}
        <h1>Analysis Reports</h1>
        <div class="table-responsive">
            <table class="table table-striped">
                <thead>
                    <tr><th>Filename</th><th>Size</th><th>Modified</th><th>Actions</th></tr>
                </thead>
                <tbody>
                    {% for report in reports %}
                    <tr>
                        <td>{{ report.filename }}</td>
                        <td>{{ "%.1f KB"|format(report.size / 1024) }}</td>
                        <td>{{ report.modified }}</td>
                        <td><a href="/api/reports/{{ report.filename }}" class="btn btn-sm btn-primary">Download</a></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endblock %}
        ''')

def main():
    """Main entry point for web application."""
    initialize_app()
    create_templates()
    
    # Start background monitoring task
    socketio.start_background_task(background_monitoring)
    
    print("🌐 Starting AMP Web Interface...")
    print("📊 Dashboard: http://localhost:5000")
    print("🔍 Analysis: http://localhost:5000/analysis")
    print("📈 Monitoring: http://localhost:5000/monitoring")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
