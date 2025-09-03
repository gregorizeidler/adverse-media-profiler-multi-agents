from typing import List
from crewai import Task
from datetime import datetime
from .agent_factory import AgentFactory
from utils.analysis import MediaMention

class TaskFactory:
    @staticmethod
    def create_collection_task(target: str, start_date: datetime = None) -> Task:
        return Task(
            description=f"""Search for any adverse media mentions about {target}.
            Focus on credible news sources, official records, and verified information.
            If a start date is provided, only look for mentions after that date.
            Collect all relevant articles, ensuring to capture the full text and metadata.""",
            agent=AgentFactory.create_collector_agent()
        )

    @staticmethod
    def create_triage_task(mentions: List[MediaMention]) -> Task:
        return Task(
            description="""Review all collected media mentions and filter out:
            1. False positives and irrelevant mentions
            2. Mentions about different entities with similar names
            3. Duplicate or redundant information
            Preserve the most relevant and significant mentions.""",
            agent=AgentFactory.create_triage_agent()
        )

    @staticmethod
    def create_analysis_task(mentions: List[MediaMention]) -> Task:
        return Task(
            description="""Analyze each media mention to:
            1. Assess the credibility of the source
            2. Evaluate the severity of the risk
            3. Categorize the type of risk
            4. Identify any connections or patterns between mentions
            Provide a detailed risk assessment for each mention.""",
            agent=AgentFactory.create_risk_analyst_agent()
        )

    @staticmethod
    def create_reporting_task(analyzed_mentions: List[MediaMention]) -> Task:
        return Task(
            description="""Generate a comprehensive adverse media report that includes:
            1. Executive summary of findings
            2. Detailed analysis of each significant risk
            3. Risk scoring and categorization
            4. Source credibility assessment
            5. Recommendations based on findings
            Format the report in a clear, professional structure.""",
            agent=AgentFactory.create_reporting_agent()
        )
