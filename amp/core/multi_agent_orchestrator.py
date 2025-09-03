import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import uuid
from pathlib import Path

from crewai import Crew, Agent, Task
from agents.agent_factory import AgentFactory
from agents.tools import AdvancedAnalysisTools
from utils.analysis import MediaMention
from config.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AgentStatus:
    agent_id: str
    state: AgentState
    current_task: Optional[str] = None
    progress: float = 0.0
    last_update: datetime = None
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()
        if self.performance_metrics is None:
            self.performance_metrics = {}

@dataclass
class SharedState:
    """Shared state between all agents"""
    target_info: Dict[str, Any]
    collected_data: Dict[str, List[Dict]] = None
    analysis_results: Dict[str, Any] = None
    confidence_scores: Dict[str, float] = None
    risk_assessment: Dict[str, Any] = None
    network_data: Dict[str, Any] = None
    final_report: Optional[Dict] = None
    
    def __post_init__(self):
        if self.collected_data is None:
            self.collected_data = {}
        if self.analysis_results is None:
            self.analysis_results = {}
        if self.confidence_scores is None:
            self.confidence_scores = {}
        if self.risk_assessment is None:
            self.risk_assessment = {}
        if self.network_data is None:
            self.network_data = {}

@dataclass
class AgentTask:
    task_id: str
    agent_id: str
    task_type: str
    priority: TaskPriority
    data: Dict[str, Any]
    dependencies: List[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.dependencies is None:
            self.dependencies = []

class MultiAgentOrchestrator:
    """Advanced multi-agent orchestrator with real async processing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.agents = AgentFactory.get_all_specialized_agents()
        self.agent_status: Dict[str, AgentStatus] = {}
        self.shared_state = SharedState(target_info={})
        self.task_queue: List[AgentTask] = []
        self.completed_tasks: Dict[str, AgentTask] = {}
        self.running_tasks: Dict[str, AgentTask] = {}
        
        # Threading and async components
        self.executor = ThreadPoolExecutor(max_workers=len(self.agents))
        self.state_lock = threading.Lock()
        self.task_lock = threading.Lock()
        
        # Communication channels
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.event_bus = asyncio.Queue()
        
        # Performance monitoring
        self.performance_metrics = {
            'start_time': None,
            'end_time': None,
            'total_tasks_completed': 0,
            'total_errors': 0,
            'agent_performance': {}
        }
        
        self._initialize_agents()
        self._setup_communication_channels()
    
    def _initialize_agents(self):
        """Initialize agent status tracking."""
        for agent_id in self.agents.keys():
            self.agent_status[agent_id] = AgentStatus(
                agent_id=agent_id,
                state=AgentState.IDLE
            )
            logger.info(f"Initialized agent: {agent_id}")
    
    def _setup_communication_channels(self):
        """Setup communication channels between agents."""
        for agent_id in self.agents.keys():
            self.message_queues[agent_id] = asyncio.Queue()
        logger.info("Communication channels established")
    
    async def start_analysis(self, target: str, additional_info: Dict = None) -> Dict:
        """
        Start comprehensive multi-agent analysis.
        
        Args:
            target: Target entity for analysis
            additional_info: Additional information about the target
            
        Returns:
            Final analysis report
        """
        logger.info(f"Starting multi-agent analysis for target: {target}")
        self.performance_metrics['start_time'] = datetime.now()
        
        # Initialize shared state
        self.shared_state.target_info = {
            'name': target,
            'analysis_id': str(uuid.uuid4()),
            'start_time': datetime.now().isoformat(),
            **(additional_info or {})
        }
        
        try:
            # Phase 1: Data Collection (Parallel)
            await self._execute_collection_phase()
            
            # Phase 2: Data Processing and Triage
            await self._execute_triage_phase()
            
            # Phase 3: Specialized Analysis (Parallel)
            await self._execute_analysis_phase()
            
            # Phase 4: Report Generation
            await self._execute_reporting_phase()
            
            # Performance summary
            self.performance_metrics['end_time'] = datetime.now()
            self._generate_performance_report()
            
            return self.shared_state.final_report
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            await self._handle_analysis_error(e)
            raise
    
    async def _execute_collection_phase(self):
        """Execute parallel data collection phase."""
        logger.info("Phase 1: Starting parallel data collection")
        
        # Create collection tasks
        collection_tasks = [
            AgentTask(
                task_id=f"collect_news_{uuid.uuid4().hex[:8]}",
                agent_id='collector',
                task_type='collect_news',
                priority=TaskPriority.HIGH,
                data={'target': self.shared_state.target_info['name']}
            ),
            AgentTask(
                task_id=f"collect_social_{uuid.uuid4().hex[:8]}",
                agent_id='collector', 
                task_type='collect_social_media',
                priority=TaskPriority.MEDIUM,
                data={'target': self.shared_state.target_info['name']}
            ),
            AgentTask(
                task_id=f"collect_records_{uuid.uuid4().hex[:8]}",
                agent_id='collector',
                task_type='collect_public_records',
                priority=TaskPriority.HIGH,
                data={'target': self.shared_state.target_info['name']}
            )
        ]
        
        # Execute tasks in parallel
        results = await self._execute_tasks_parallel(collection_tasks)
        
        # Aggregate collected data
        self.shared_state.collected_data = {
            'news_articles': [],
            'social_mentions': [],
            'public_records': [],
            'documents': []
        }
        
        for task_result in results:
            if task_result.task_type == 'collect_news':
                self.shared_state.collected_data['news_articles'].extend(
                    task_result.result.get('articles', [])
                )
            elif task_result.task_type == 'collect_social_media':
                self.shared_state.collected_data['social_mentions'].extend(
                    task_result.result.get('mentions', [])
                )
            elif task_result.task_type == 'collect_public_records':
                self.shared_state.collected_data['public_records'].extend(
                    task_result.result.get('records', [])
                )
        
        logger.info(f"Collection completed: {len(self.shared_state.collected_data)} data sources")
    
    async def _execute_triage_phase(self):
        """Execute data triage and entity disambiguation."""
        logger.info("Phase 2: Starting data triage and disambiguation")
        
        triage_task = AgentTask(
            task_id=f"triage_{uuid.uuid4().hex[:8]}",
            agent_id='triage',
            task_type='entity_disambiguation',
            priority=TaskPriority.CRITICAL,
            data={
                'collected_data': self.shared_state.collected_data,
                'target_info': self.shared_state.target_info
            }
        )
        
        result = await self._execute_single_task(triage_task)
        
        # Update shared state with verified data
        self.shared_state.collected_data = result.result.get('verified_data', {})
        self.shared_state.confidence_scores['triage'] = result.result.get('confidence', 0.0)
        
        logger.info("Triage phase completed")
    
    async def _execute_analysis_phase(self):
        """Execute specialized analysis in parallel."""
        logger.info("Phase 3: Starting parallel specialized analysis")
        
        # Create analysis tasks
        analysis_tasks = [
            AgentTask(
                task_id=f"analyze_financial_{uuid.uuid4().hex[:8]}",
                agent_id='risk_analyst',
                task_type='financial_analysis',
                priority=TaskPriority.HIGH,
                data={
                    'data': self.shared_state.collected_data,
                    'target_info': self.shared_state.target_info
                }
            ),
            AgentTask(
                task_id=f"analyze_legal_{uuid.uuid4().hex[:8]}",
                agent_id='risk_analyst', 
                task_type='legal_analysis',
                priority=TaskPriority.HIGH,
                data={
                    'data': self.shared_state.collected_data,
                    'target_info': self.shared_state.target_info
                }
            ),
            AgentTask(
                task_id=f"analyze_reputation_{uuid.uuid4().hex[:8]}",
                agent_id='risk_analyst',
                task_type='reputational_analysis',
                priority=TaskPriority.MEDIUM,
                data={
                    'data': self.shared_state.collected_data,
                    'target_info': self.shared_state.target_info
                }
            ),
            AgentTask(
                task_id=f"analyze_network_{uuid.uuid4().hex[:8]}",
                agent_id='risk_analyst',
                task_type='network_analysis',
                priority=TaskPriority.MEDIUM,
                data={
                    'data': self.shared_state.collected_data,
                    'target_info': self.shared_state.target_info
                }
            )
        ]
        
        # Execute analysis tasks in parallel
        results = await self._execute_tasks_parallel(analysis_tasks)
        
        # Aggregate analysis results
        for task_result in results:
            if task_result.task_type == 'financial_analysis':
                self.shared_state.analysis_results['financial'] = task_result.result
            elif task_result.task_type == 'legal_analysis':
                self.shared_state.analysis_results['legal'] = task_result.result
            elif task_result.task_type == 'reputational_analysis':
                self.shared_state.analysis_results['reputational'] = task_result.result
            elif task_result.task_type == 'network_analysis':
                self.shared_state.network_data = task_result.result
        
        # Calculate overall risk assessment
        await self._calculate_overall_risk_assessment()
        
        logger.info("Analysis phase completed")
    
    async def _execute_reporting_phase(self):
        """Execute final report generation."""
        logger.info("Phase 4: Starting report generation")
        
        reporting_task = AgentTask(
            task_id=f"report_{uuid.uuid4().hex[:8]}",
            agent_id='reporter',
            task_type='comprehensive_report',
            priority=TaskPriority.CRITICAL,
            data={
                'target_info': self.shared_state.target_info,
                'collected_data': self.shared_state.collected_data,
                'analysis_results': self.shared_state.analysis_results,
                'risk_assessment': self.shared_state.risk_assessment,
                'network_data': self.shared_state.network_data,
                'confidence_scores': self.shared_state.confidence_scores
            }
        )
        
        result = await self._execute_single_task(reporting_task)
        self.shared_state.final_report = result.result
        
        # Save report to file
        await self._save_final_report()
        
        logger.info("Reporting phase completed")
    
    async def _execute_tasks_parallel(self, tasks: List[AgentTask]) -> List[AgentTask]:
        """Execute multiple tasks in parallel."""
        logger.info(f"Executing {len(tasks)} tasks in parallel")
        
        # Create coroutines for all tasks
        task_coroutines = [self._execute_single_task(task) for task in tasks]
        
        # Execute all tasks concurrently
        completed_tasks = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        # Handle results and exceptions
        successful_tasks = []
        for i, result in enumerate(completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Task {tasks[i].task_id} failed: {str(result)}")
                tasks[i].error = str(result)
                self.performance_metrics['total_errors'] += 1
            else:
                successful_tasks.append(result)
                self.performance_metrics['total_tasks_completed'] += 1
        
        return successful_tasks
    
    async def _execute_single_task(self, task: AgentTask) -> AgentTask:
        """Execute a single task using the appropriate agent."""
        logger.info(f"Executing task {task.task_id} for agent {task.agent_id}")
        
        with self.task_lock:
            self.running_tasks[task.task_id] = task
        
        try:
            # Update agent status
            await self._update_agent_status(task.agent_id, AgentState.RUNNING, task.task_id)
            
            task.started_at = datetime.now()
            
            # Execute task based on type
            result = await self._dispatch_task(task)
            
            task.completed_at = datetime.now()
            task.result = result
            
            # Update agent status
            await self._update_agent_status(task.agent_id, AgentState.COMPLETED)
            
            # Move to completed tasks
            with self.task_lock:
                del self.running_tasks[task.task_id]
                self.completed_tasks[task.task_id] = task
            
            logger.info(f"Task {task.task_id} completed successfully")
            return task
            
        except Exception as e:
            task.error = str(e)
            await self._update_agent_status(task.agent_id, AgentState.ERROR, error_message=str(e))
            
            with self.task_lock:
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
            
            logger.error(f"Task {task.task_id} failed: {str(e)}")
            raise
    
    async def _dispatch_task(self, task: AgentTask) -> Dict:
        """Dispatch task to appropriate handler based on task type."""
        task_handlers = {
            'collect_news': self._handle_news_collection,
            'collect_social_media': self._handle_social_collection,
            'collect_public_records': self._handle_records_collection,
            'entity_disambiguation': self._handle_entity_disambiguation,
            'financial_analysis': self._handle_financial_analysis,
            'legal_analysis': self._handle_legal_analysis,
            'reputational_analysis': self._handle_reputational_analysis,
            'network_analysis': self._handle_network_analysis,
            'comprehensive_report': self._handle_report_generation
        }
        
        handler = task_handlers.get(task.task_type)
        if not handler:
            raise ValueError(f"Unknown task type: {task.task_type}")
        
        return await handler(task)
    
    # Task handlers
    async def _handle_news_collection(self, task: AgentTask) -> Dict:
        """Handle news collection task."""
        target = task.data['target']
        
        # Simulate API calls (replace with real implementations)
        await asyncio.sleep(2)  # Simulate network delay
        
        from agents.tools import ExternalAPITools
        articles = ExternalAPITools.search_news_api(target)
        
        return {'articles': articles, 'source': 'news_api'}
    
    async def _handle_social_collection(self, task: AgentTask) -> Dict:
        """Handle social media collection task."""
        target = task.data['target']
        
        await asyncio.sleep(1.5)
        
        from agents.tools import ExternalAPITools
        mentions = ExternalAPITools.search_social_media(target)
        
        return {'mentions': mentions, 'source': 'social_media'}
    
    async def _handle_records_collection(self, task: AgentTask) -> Dict:
        """Handle public records collection task."""
        target = task.data['target']
        
        await asyncio.sleep(3)  # Simulate slower database queries
        
        from agents.tools import ExternalAPITools
        records = ExternalAPITools.query_public_records(target)
        
        return {'records': records, 'source': 'public_records'}
    
    async def _handle_entity_disambiguation(self, task: AgentTask) -> Dict:
        """Handle entity disambiguation task."""
        collected_data = task.data['collected_data']
        target_info = task.data['target_info']
        
        await asyncio.sleep(1)
        
        # Extract all mentions for disambiguation
        all_mentions = []
        for source_type, data_list in collected_data.items():
            all_mentions.extend(data_list)
        
        verified_data = AdvancedAnalysisTools.entity_disambiguation(all_mentions, target_info)
        confidence = AdvancedAnalysisTools.confidence_scoring({
            'mentions': verified_data,
            'sources': [m.get('source', '') for m in verified_data]
        })
        
        return {
            'verified_data': {'verified_mentions': verified_data},
            'confidence': confidence.get('overall_confidence', 0.0)
        }
    
    async def _handle_financial_analysis(self, task: AgentTask) -> Dict:
        """Handle financial risk analysis task."""
        data = task.data['data']
        
        await asyncio.sleep(2)
        
        # Flatten data for analysis
        all_data = []
        for source_type, items in data.items():
            all_data.extend(items)
        
        return AdvancedAnalysisTools.financial_risk_analysis(all_data)
    
    async def _handle_legal_analysis(self, task: AgentTask) -> Dict:
        """Handle legal risk analysis task."""
        data = task.data['data']
        
        await asyncio.sleep(1.5)
        
        all_data = []
        for source_type, items in data.items():
            all_data.extend(items)
        
        return AdvancedAnalysisTools.legal_risk_analysis(all_data)
    
    async def _handle_reputational_analysis(self, task: AgentTask) -> Dict:
        """Handle reputational risk analysis task."""
        data = task.data['data']
        
        await asyncio.sleep(1)
        
        all_data = []
        for source_type, items in data.items():
            all_data.extend(items)
        
        return AdvancedAnalysisTools.reputational_risk_analysis(all_data)
    
    async def _handle_network_analysis(self, task: AgentTask) -> Dict:
        """Handle network analysis task."""
        data = task.data['data']
        
        await asyncio.sleep(2.5)
        
        # Extract entities for network analysis
        entities = []
        for source_type, items in data.items():
            for item in items:
                entities.append({
                    'name': item.get('name', 'Unknown'),
                    'connections': item.get('connections', [])
                })
        
        return AdvancedAnalysisTools.network_analysis(entities)
    
    async def _handle_report_generation(self, task: AgentTask) -> Dict:
        """Handle comprehensive report generation."""
        await asyncio.sleep(1)
        
        return AdvancedAnalysisTools.compile_comprehensive_report(task.data)
    
    async def _calculate_overall_risk_assessment(self):
        """Calculate overall risk assessment from all analyses."""
        financial_risk = self.shared_state.analysis_results.get('financial', {}).get('fraud_risk', {}).get('score', 0)
        legal_risk = self.shared_state.analysis_results.get('legal', {}).get('criminal_risk', {}).get('score', 0)
        reputational_risk = self.shared_state.analysis_results.get('reputational', {}).get('risk_score', 0)
        
        # Calculate weighted average
        weights = {'financial': 0.4, 'legal': 0.35, 'reputational': 0.25}
        overall_score = (
            financial_risk * weights['financial'] +
            legal_risk * weights['legal'] +
            reputational_risk * weights['reputational']
        )
        
        risk_level = 'LOW'
        if overall_score > 0.7:
            risk_level = 'HIGH'
        elif overall_score > 0.4:
            risk_level = 'MEDIUM'
        
        self.shared_state.risk_assessment = {
            'overall_score': overall_score,
            'risk_level': risk_level,
            'component_scores': {
                'financial': financial_risk,
                'legal': legal_risk,
                'reputational': reputational_risk
            },
            'calculated_at': datetime.now().isoformat()
        }
    
    async def _update_agent_status(self, agent_id: str, state: AgentState, 
                                 current_task: str = None, error_message: str = None):
        """Update agent status thread-safely."""
        with self.state_lock:
            if agent_id in self.agent_status:
                self.agent_status[agent_id].state = state
                self.agent_status[agent_id].last_update = datetime.now()
                
                if current_task:
                    self.agent_status[agent_id].current_task = current_task
                
                if error_message:
                    self.agent_status[agent_id].error_message = error_message
    
    async def _save_final_report(self):
        """Save the final report to file."""
        if not self.shared_state.final_report:
            return
        
        # Create reports directory
        reports_dir = Path("data/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        target_name = self.shared_state.target_info.get('name', 'unknown').replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{target_name}_analysis_{timestamp}.json"
        
        report_path = reports_dir / filename
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.shared_state.final_report, f, 
                     ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Final report saved to: {report_path}")
    
    def _generate_performance_report(self):
        """Generate performance metrics report."""
        if self.performance_metrics['start_time'] and self.performance_metrics['end_time']:
            duration = self.performance_metrics['end_time'] - self.performance_metrics['start_time']
            
            performance_summary = {
                'total_duration_seconds': duration.total_seconds(),
                'tasks_completed': self.performance_metrics['total_tasks_completed'],
                'errors_encountered': self.performance_metrics['total_errors'],
                'success_rate': self.performance_metrics['total_tasks_completed'] / 
                              max(1, self.performance_metrics['total_tasks_completed'] + 
                                  self.performance_metrics['total_errors']),
                'agents_utilized': len(self.agents),
                'final_report_size': len(str(self.shared_state.final_report)) if self.shared_state.final_report else 0
            }
            
            logger.info(f"Performance Summary: {performance_summary}")
            self.performance_metrics.update(performance_summary)
    
    async def _handle_analysis_error(self, error: Exception):
        """Handle analysis errors gracefully."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'agent_states': {aid: asdict(status) for aid, status in self.agent_status.items()},
            'running_tasks': list(self.running_tasks.keys()),
            'completed_tasks': list(self.completed_tasks.keys())
        }
        
        # Save error report
        error_path = Path("data/reports/errors") 
        error_path.mkdir(parents=True, exist_ok=True)
        
        error_file = error_path / f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2, default=str)
        
        logger.error(f"Error report saved to: {error_file}")
    
    def get_system_status(self) -> Dict:
        """Get current system status for monitoring."""
        with self.state_lock:
            return {
                'orchestrator_status': 'running',
                'agents': {aid: asdict(status) for aid, status in self.agent_status.items()},
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'shared_state_keys': list(self.shared_state.collected_data.keys()),
                'performance_metrics': self.performance_metrics,
                'timestamp': datetime.now().isoformat()
            }
    
    async def graceful_shutdown(self):
        """Gracefully shutdown the orchestrator."""
        logger.info("Initiating graceful shutdown...")
        
        # Cancel running tasks
        with self.task_lock:
            for task_id in list(self.running_tasks.keys()):
                logger.info(f"Cancelling task: {task_id}")
                # In a real implementation, you'd signal task cancellation
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Orchestrator shutdown complete")

# Async wrapper for easy usage
async def run_multi_agent_analysis(target: str, config: Config, 
                                 additional_info: Dict = None) -> Dict:
    """
    Convenient function to run multi-agent analysis.
    
    Args:
        target: Target entity for analysis
        config: System configuration
        additional_info: Additional information about target
        
    Returns:
        Complete analysis report
    """
    orchestrator = MultiAgentOrchestrator(config)
    
    try:
        return await orchestrator.start_analysis(target, additional_info)
    finally:
        await orchestrator.graceful_shutdown()
