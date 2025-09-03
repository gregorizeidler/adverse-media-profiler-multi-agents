"""
Advanced Monitoring and Logging System for AMP Multi-Agent Intelligence

Real-time monitoring, performance tracking, and comprehensive logging for all agents.
"""

import logging
import json
import time
import asyncio
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import sqlite3
from contextlib import contextmanager

@dataclass
class PerformanceMetric:
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class AgentHealthCheck:
    agent_id: str
    status: str
    cpu_usage: float
    memory_usage: float
    response_time: float
    errors_count: int
    last_activity: datetime
    uptime: float

@dataclass
class SystemAlert:
    alert_id: str
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    source: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.metrics: deque = deque(maxlen=max_history)
        self.aggregated_metrics: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        with self.lock:
            self.metrics.append(metric)
            self.aggregated_metrics[metric.metric_name].append(metric.value)
    
    def get_metric_stats(self, metric_name: str, window_minutes: int = 5) -> Dict:
        """Get statistics for a metric within a time window."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self.lock:
            recent_values = [
                m.value for m in self.metrics 
                if m.metric_name == metric_name and m.timestamp >= cutoff_time
            ]
        
        if not recent_values:
            return {}
        
        return {
            'count': len(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'avg': sum(recent_values) / len(recent_values),
            'latest': recent_values[-1] if recent_values else 0
        }

class SystemMonitor:
    """Monitors system resources and agent health."""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.health_checks: Dict[str, AgentHealthCheck] = {}
        self.alerts: List[SystemAlert] = []
        self.metrics_collector = MetricsCollector()
        
    def start_monitoring(self, interval: float = 30.0):
        """Start system monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logging.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logging.info("System monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._collect_system_metrics()
                self._check_system_health()
                time.sleep(interval)
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        now = datetime.now()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_collector.record_metric(PerformanceMetric(
            timestamp=now,
            metric_name="system_cpu_usage",
            value=cpu_percent,
            unit="percent"
        ))
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics_collector.record_metric(PerformanceMetric(
            timestamp=now,
            metric_name="system_memory_usage",
            value=memory.percent,
            unit="percent"
        ))
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.metrics_collector.record_metric(PerformanceMetric(
            timestamp=now,
            metric_name="system_disk_usage",
            value=(disk.used / disk.total) * 100,
            unit="percent"
        ))
        
        # Network I/O
        net_io = psutil.net_io_counters()
        self.metrics_collector.record_metric(PerformanceMetric(
            timestamp=now,
            metric_name="network_bytes_sent",
            value=net_io.bytes_sent,
            unit="bytes"
        ))
        self.metrics_collector.record_metric(PerformanceMetric(
            timestamp=now,
            metric_name="network_bytes_recv",
            value=net_io.bytes_recv,
            unit="bytes"
        ))
    
    def _check_system_health(self):
        """Check system health and generate alerts."""
        # Check CPU usage
        cpu_stats = self.metrics_collector.get_metric_stats("system_cpu_usage")
        if cpu_stats and cpu_stats['avg'] > 90:
            self.generate_alert(
                severity="WARNING",
                message=f"High CPU usage detected: {cpu_stats['avg']:.1f}%",
                source="system_monitor"
            )
        
        # Check memory usage
        memory_stats = self.metrics_collector.get_metric_stats("system_memory_usage")
        if memory_stats and memory_stats['avg'] > 90:
            self.generate_alert(
                severity="WARNING",
                message=f"High memory usage detected: {memory_stats['avg']:.1f}%",
                source="system_monitor"
            )
    
    def register_agent_health(self, agent_id: str, health_check: AgentHealthCheck):
        """Register agent health information."""
        self.health_checks[agent_id] = health_check
        
        # Record agent metrics
        now = datetime.now()
        metrics = [
            ("agent_cpu_usage", health_check.cpu_usage, "percent"),
            ("agent_memory_usage", health_check.memory_usage, "MB"),
            ("agent_response_time", health_check.response_time, "seconds"),
            ("agent_errors", health_check.errors_count, "count"),
            ("agent_uptime", health_check.uptime, "seconds")
        ]
        
        for metric_name, value, unit in metrics:
            self.metrics_collector.record_metric(PerformanceMetric(
                timestamp=now,
                metric_name=metric_name,
                value=value,
                unit=unit,
                agent_id=agent_id
            ))
    
    def generate_alert(self, severity: str, message: str, source: str):
        """Generate a system alert."""
        alert = SystemAlert(
            alert_id=f"alert_{len(self.alerts)}_{int(time.time())}",
            severity=severity,
            message=message,
            source=source,
            timestamp=datetime.now()
        )
        self.alerts.append(alert)
        
        # Log the alert
        log_level = {
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }.get(severity, logging.INFO)
        
        logging.log(log_level, f"[ALERT] {message} (Source: {source})")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        return {
            'monitoring_active': self.monitoring,
            'system_metrics': {
                'cpu': self.metrics_collector.get_metric_stats("system_cpu_usage"),
                'memory': self.metrics_collector.get_metric_stats("system_memory_usage"),
                'disk': self.metrics_collector.get_metric_stats("system_disk_usage")
            },
            'agent_health': {aid: asdict(health) for aid, health in self.health_checks.items()},
            'active_alerts': [asdict(alert) for alert in self.alerts if not alert.resolved],
            'total_metrics_collected': len(self.metrics_collector.metrics)
        }

class AdvancedLogger:
    """Advanced logging system with structured logs and metrics."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for structured logs
        self.db_path = self.log_dir / "amp_logs.db"
        self._init_database()
        
        # Setup loggers
        self._setup_loggers()
        
        # Performance tracking
        self.performance_data: Dict[str, List] = defaultdict(list)
        
    def _init_database(self):
        """Initialize SQLite database for structured logging."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    agent_id TEXT,
                    task_id TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    task_id TEXT,
                    operation TEXT NOT NULL,
                    duration REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
                CREATE INDEX IF NOT EXISTS idx_logs_agent ON logs(agent_id);
                CREATE INDEX IF NOT EXISTS idx_perf_agent ON performance_logs(agent_id);
            """)
    
    def _setup_loggers(self):
        """Setup file and console loggers."""
        # Main application logger
        self.app_logger = logging.getLogger('amp')
        self.app_logger.setLevel(logging.DEBUG)
        
        # File handler for general logs
        file_handler = logging.FileHandler(
            self.log_dir / f"amp_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.app_logger.addHandler(file_handler)
        self.app_logger.addHandler(console_handler)
        
        # Agent-specific loggers
        self.agent_loggers = {}
        for agent_type in ['collector', 'triage', 'risk_analyst', 'reporter', 'orchestrator']:
            logger = logging.getLogger(f'amp.{agent_type}')
            logger.setLevel(logging.DEBUG)
            
            # Agent-specific file handler
            agent_file = logging.FileHandler(
                self.log_dir / f"{agent_type}_{datetime.now().strftime('%Y%m%d')}.log",
                encoding='utf-8'
            )
            agent_file.setFormatter(detailed_formatter)
            logger.addHandler(agent_file)
            
            self.agent_loggers[agent_type] = logger
    
    @contextmanager
    def log_operation(self, agent_id: str, operation: str, task_id: str = None):
        """Context manager for logging operation performance."""
        start_time = time.time()
        success = False
        error = None
        
        try:
            self.log_structured(
                level="INFO",
                source=agent_id,
                message=f"Starting {operation}",
                agent_id=agent_id,
                task_id=task_id
            )
            yield
            success = True
            
        except Exception as e:
            error = str(e)
            self.log_structured(
                level="ERROR",
                source=agent_id,
                message=f"Operation {operation} failed: {error}",
                agent_id=agent_id,
                task_id=task_id
            )
            raise
        
        finally:
            duration = time.time() - start_time
            
            # Log performance
            self._log_performance(
                agent_id=agent_id,
                task_id=task_id,
                operation=operation,
                duration=duration,
                success=success,
                metadata={'error': error} if error else None
            )
            
            if success:
                self.log_structured(
                    level="INFO",
                    source=agent_id,
                    message=f"Completed {operation} in {duration:.2f}s",
                    agent_id=agent_id,
                    task_id=task_id
                )
    
    def log_structured(self, level: str, source: str, message: str,
                      agent_id: str = None, task_id: str = None, metadata: Dict = None):
        """Log structured message to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO logs (timestamp, level, source, message, agent_id, task_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                level,
                source,
                message,
                agent_id,
                task_id,
                json.dumps(metadata) if metadata else None
            ))
        
        # Also log to appropriate file logger
        logger = self.agent_loggers.get(agent_id, self.app_logger)
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(log_level, f"[{source}] {message}")
    
    def _log_performance(self, agent_id: str, operation: str, duration: float,
                        success: bool, task_id: str = None, metadata: Dict = None):
        """Log performance data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_logs 
                (timestamp, agent_id, task_id, operation, duration, success, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                agent_id,
                task_id,
                operation,
                duration,
                success,
                json.dumps(metadata) if metadata else None
            ))
    
    def get_performance_stats(self, agent_id: str = None, hours: int = 24) -> Dict:
        """Get performance statistics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT operation, 
                       COUNT(*) as total_ops,
                       AVG(duration) as avg_duration,
                       MIN(duration) as min_duration,
                       MAX(duration) as max_duration,
                       SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_ops
                FROM performance_logs 
                WHERE timestamp > ?
            """
            params = [cutoff_time.isoformat()]
            
            if agent_id:
                query += " AND agent_id = ?"
                params.append(agent_id)
            
            query += " GROUP BY operation"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            stats = {}
            for row in results:
                op, total, avg_dur, min_dur, max_dur, successful = row
                stats[op] = {
                    'total_operations': total,
                    'successful_operations': successful,
                    'success_rate': (successful / total) * 100 if total > 0 else 0,
                    'avg_duration': avg_dur,
                    'min_duration': min_dur,
                    'max_duration': max_dur
                }
            
            return stats
    
    def get_error_summary(self, hours: int = 24) -> Dict:
        """Get error summary."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT agent_id, COUNT(*) as error_count
                FROM logs 
                WHERE level = 'ERROR' AND timestamp > ?
                GROUP BY agent_id
                ORDER BY error_count DESC
            """, [cutoff_time.isoformat()])
            
            agent_errors = dict(cursor.fetchall())
            
            cursor.execute("""
                SELECT COUNT(*) as total_errors
                FROM logs 
                WHERE level = 'ERROR' AND timestamp > ?
            """, [cutoff_time.isoformat()])
            
            total_errors = cursor.fetchone()[0]
            
            return {
                'total_errors': total_errors,
                'errors_by_agent': agent_errors,
                'time_window_hours': hours
            }

class WebMonitoringDashboard:
    """Simple web dashboard for monitoring (optional feature)."""
    
    def __init__(self, system_monitor: SystemMonitor, logger: AdvancedLogger, port: int = 8080):
        self.system_monitor = system_monitor
        self.logger = logger
        self.port = port
        self.app = None
        
    def create_app(self):
        """Create Flask app for monitoring dashboard."""
        try:
            from flask import Flask, jsonify, render_template_string
            
            self.app = Flask(__name__)
            
            @self.app.route('/api/status')
            def api_status():
                return jsonify(self.system_monitor.get_system_status())
            
            @self.app.route('/api/performance/<agent_id>')
            def api_performance(agent_id):
                return jsonify(self.logger.get_performance_stats(agent_id))
            
            @self.app.route('/api/errors')
            def api_errors():
                return jsonify(self.logger.get_error_summary())
            
            @self.app.route('/')
            def dashboard():
                return render_template_string(self._get_dashboard_template())
            
            return self.app
            
        except ImportError:
            logging.warning("Flask not available. Web dashboard disabled.")
            return None
    
    def _get_dashboard_template(self) -> str:
        """Simple HTML template for the dashboard."""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>AMP Monitoring Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric { margin: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px; }
                .error { background: #ffebee; }
                .warning { background: #fff3e0; }
                .success { background: #e8f5e8; }
            </style>
        </head>
        <body>
            <h1>🔍 AMP Monitoring Dashboard</h1>
            <div id="status"></div>
            <div id="performance"></div>
            <div id="errors"></div>
            
            <script>
                async function updateDashboard() {
                    try {
                        const status = await fetch('/api/status').then(r => r.json());
                        const errors = await fetch('/api/errors').then(r => r.json());
                        
                        document.getElementById('status').innerHTML = 
                            `<h2>System Status</h2><pre>${JSON.stringify(status, null, 2)}</pre>`;
                        document.getElementById('errors').innerHTML = 
                            `<h2>Error Summary</h2><pre>${JSON.stringify(errors, null, 2)}</pre>`;
                    } catch (e) {
                        console.error('Dashboard update failed:', e);
                    }
                }
                
                updateDashboard();
                setInterval(updateDashboard, 30000); // Update every 30 seconds
            </script>
        </body>
        </html>
        '''
    
    def start_server(self):
        """Start the monitoring web server."""
        if self.create_app():
            logging.info(f"Starting web monitoring dashboard on port {self.port}")
            self.app.run(host='0.0.0.0', port=self.port, debug=False)
        else:
            logging.warning("Web dashboard could not be started")

# Global monitoring instance
_global_monitor = None
_global_logger = None

def get_system_monitor() -> SystemMonitor:
    """Get global system monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SystemMonitor()
    return _global_monitor

def get_advanced_logger() -> AdvancedLogger:
    """Get global advanced logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = AdvancedLogger()
    return _global_logger

def initialize_monitoring():
    """Initialize the monitoring system."""
    monitor = get_system_monitor()
    logger = get_advanced_logger()
    
    monitor.start_monitoring()
    
    logger.log_structured(
        level="INFO",
        source="monitoring_system",
        message="Monitoring system initialized"
    )
    
    return monitor, logger

def shutdown_monitoring():
    """Shutdown the monitoring system."""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()
    
    logger = get_advanced_logger()
    logger.log_structured(
        level="INFO",
        source="monitoring_system",
        message="Monitoring system shutdown"
    )
