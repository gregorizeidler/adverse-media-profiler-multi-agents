from crewai import Agent
from typing import List
from .tools import LocalProcessingTools

def create_orchestrator_agent() -> Agent:
    return Agent(
        role='Orchestrator',
        goal='Coordinate and manage the workflow of all other agents',
        backstory="""You are the Maestro, the brain of the operation. Your role is to 
        coordinate all other agents, manage the flow of information, and ensure the 
        successful completion of the investigation.""",
        verbose=True
    )

def create_media_hunter_agent() -> Agent:
    return Agent(
        role='Media Hunter',
        goal='Search and analyze unstructured data sources for relevant information',
        backstory="""You are an expert in finding and analyzing information from various
        unstructured sources. You focus on local documents, news archives, and other
        text-based sources.""",
        tools=[
            LocalProcessingTools.analyze_local_news,
            LocalProcessingTools.process_documents
        ],
        verbose=True
    )

def create_list_checker_agent() -> Agent:
    return Agent(
        role='List Checker',
        goal='Verify target against structured databases and lists',
        backstory="""You are a specialist in checking and cross-referencing information
        against structured databases, including locally stored sanctions lists, PEP lists,
        and corporate registries.""",
        tools=[LocalProcessingTools.check_local_databases],
        verbose=True
    )

def create_triage_entity_agent() -> Agent:
    return Agent(
        role='Triage and Entity Specialist',
        goal='Filter and verify entity mentions, removing false positives',
        backstory="""You are an expert in entity disambiguation and verification. Your
        job is to ensure that all mentions actually refer to the target entity.""",
        tools=[LocalProcessingTools.entity_disambiguation],
        verbose=True
    )

def create_financial_risk_analyst() -> Agent:
    return Agent(
        role='Financial Risk Analyst',
        goal='Analyze financial risks and identify potential financial crimes',
        backstory="""You are a financial crime specialist focused on identifying patterns
        of fraud, money laundering, and other financial irregularities.""",
        tools=[LocalProcessingTools.analyze_financial_risk],
        verbose=True
    )

def create_legal_risk_analyst() -> Agent:
    return Agent(
        role='Legal & Criminal Risk Analyst',
        goal='Analyze legal and criminal risks',
        backstory="""You are a legal expert specialized in identifying criminal activities,
        investigations, and legal proceedings.""",
        tools=[LocalProcessingTools.analyze_legal_risk],
        verbose=True
    )

def create_reputational_risk_analyst() -> Agent:
    return Agent(
        role='Reputational & Ethical Risk Analyst',
        goal='Analyze reputational and ethical risks',
        backstory="""You are an expert in identifying and analyzing reputational risks,
        ethical concerns, and ESG-related issues.""",
        tools=[LocalProcessingTools.analyze_reputational_risk],
        verbose=True
    )

def create_connections_detective() -> Agent:
    return Agent(
        role='Connections Detective',
        goal='Map and analyze relationships between entities',
        backstory="""You are a network analysis specialist who excels at identifying
        and mapping connections between people, companies, and organizations.""",
        tools=[LocalProcessingTools.analyze_connections],
        verbose=True
    )

def create_quality_auditor() -> Agent:
    return Agent(
        role='Quality Auditor',
        goal='Verify and validate all findings for accuracy',
        backstory="""You are a meticulous fact-checker responsible for ensuring the
        accuracy and reliability of all findings and conclusions.""",
        tools=[LocalProcessingTools.verify_findings],
        verbose=True
    )

def create_dossier_compiler() -> Agent:
    return Agent(
        role='Dossier Compiler',
        goal='Create comprehensive final reports',
        backstory="""You are an expert report writer who excels at synthesizing complex
        information into clear, structured, and actionable intelligence reports.""",
        tools=[LocalProcessingTools.compile_report],
        verbose=True
    )

def get_all_agents() -> List[Agent]:
    """Returns all agents in the correct execution order."""
    return [
        create_orchestrator_agent(),
        create_media_hunter_agent(),
        create_list_checker_agent(),
        create_triage_entity_agent(),
        create_financial_risk_analyst(),
        create_legal_risk_analyst(),
        create_reputational_risk_analyst(),
        create_connections_detective(),
        create_quality_auditor(),
        create_dossier_compiler()
    ]
