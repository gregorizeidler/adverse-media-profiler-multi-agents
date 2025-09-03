from crewai import Agent
from typing import Dict, Any
from .tools import LocalProcessingTools, ExternalAPITools, AdvancedAnalysisTools

class AgentFactory:
    """Factory for creating specialized agents with proper tool assignments."""
    
    @staticmethod
    def create_collector_agent() -> Agent:
        """Creates the data collection agent."""
        return Agent(
            role='Data Collector',
            goal='Collect comprehensive information about target entities from multiple sources',
            backstory="""You are an elite intelligence collector with expertise in 
            gathering information from diverse sources including news outlets, 
            social media, public records, and databases. Your methodology is thorough, 
            systematic, and focuses on credible sources.""",
            tools=[
                LocalProcessingTools.analyze_local_news,
                LocalProcessingTools.process_documents,
                ExternalAPITools.search_news_api,
                ExternalAPITools.search_social_media,
                ExternalAPITools.query_public_records
            ],
            verbose=True,
            allow_delegation=True,
            max_execution_time=300,
            step_callback=lambda step: print(f"[COLLECTOR] {step.tool}: {step.tool_input}")
        )
    
    @staticmethod
    def create_triage_agent() -> Agent:
        """Creates the entity triage and disambiguation agent."""
        return Agent(
            role='Entity Triage Specialist',
            goal='Filter, verify, and disambiguate entity mentions to ensure accuracy',
            backstory="""You are a meticulous fact-checker and entity resolution expert. 
            Your specialty is distinguishing between entities with similar names, 
            filtering false positives, and ensuring data quality. You have an eye 
            for detail and never accept information at face value.""",
            tools=[
                AdvancedAnalysisTools.entity_disambiguation,
                AdvancedAnalysisTools.source_verification,
                AdvancedAnalysisTools.confidence_scoring
            ],
            verbose=True,
            allow_delegation=False,
            max_execution_time=180,
            step_callback=lambda step: print(f"[TRIAGE] {step.tool}: Processing {len(str(step.tool_input))} chars")
        )
    
    @staticmethod
    def create_risk_analyst_agent() -> Agent:
        """Creates the comprehensive risk analysis agent."""
        return Agent(
            role='Multi-Domain Risk Analyst',
            goal='Conduct deep risk analysis across financial, legal, and reputational dimensions',
            backstory="""You are a seasoned risk analyst with expertise spanning 
            financial crimes, regulatory compliance, and reputational management. 
            You can identify subtle patterns and assess complex risk scenarios 
            with precision and insight.""",
            tools=[
                AdvancedAnalysisTools.financial_risk_analysis,
                AdvancedAnalysisTools.legal_risk_analysis,
                AdvancedAnalysisTools.reputational_risk_analysis,
                AdvancedAnalysisTools.network_analysis,
                AdvancedAnalysisTools.sentiment_analysis,
                AdvancedAnalysisTools.temporal_pattern_analysis
            ],
            verbose=True,
            allow_delegation=True,
            max_execution_time=600,
            step_callback=lambda step: print(f"[RISK_ANALYST] {step.tool}: Analyzing risks")
        )
    
    @staticmethod
    def create_reporting_agent() -> Agent:
        """Creates the final reporting and synthesis agent."""
        return Agent(
            role='Intelligence Report Compiler',
            goal='Synthesize all findings into comprehensive, actionable intelligence reports',
            backstory="""You are an expert intelligence analyst and report writer 
            with years of experience in synthesizing complex information into 
            clear, actionable reports. You understand the needs of decision-makers 
            and present findings in a structured, professional manner.""",
            tools=[
                AdvancedAnalysisTools.compile_comprehensive_report,
                AdvancedAnalysisTools.generate_executive_summary,
                AdvancedAnalysisTools.create_risk_matrix,
                AdvancedAnalysisTools.generate_recommendations
            ],
            verbose=True,
            allow_delegation=False,
            max_execution_time=240,
            step_callback=lambda step: print(f"[REPORTER] {step.tool}: Compiling findings")
        )
    
    @staticmethod
    def create_orchestrator_agent() -> Agent:
        """Creates the master orchestrator agent."""
        return Agent(
            role='Master Orchestrator',
            goal='Coordinate all agents and ensure optimal workflow execution',
            backstory="""You are the maestro of this intelligence operation. 
            With deep understanding of each agent's capabilities and the overall 
            mission objectives, you orchestrate the perfect symphony of analysis. 
            You make real-time decisions about workflow, prioritization, and 
            quality control.""",
            tools=[
                AdvancedAnalysisTools.workflow_coordination,
                AdvancedAnalysisTools.quality_assessment,
                AdvancedAnalysisTools.progress_monitoring
            ],
            verbose=True,
            allow_delegation=True,
            max_execution_time=120,
            step_callback=lambda step: print(f"[ORCHESTRATOR] {step.tool}: Coordinating workflow")
        )

    @staticmethod
    def get_all_specialized_agents() -> Dict[str, Agent]:
        """Returns all agents in a dictionary for easy access."""
        return {
            'orchestrator': AgentFactory.create_orchestrator_agent(),
            'collector': AgentFactory.create_collector_agent(), 
            'triage': AgentFactory.create_triage_agent(),
            'risk_analyst': AgentFactory.create_risk_analyst_agent(),
            'reporter': AgentFactory.create_reporting_agent()
        }