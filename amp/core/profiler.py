from typing import List, Optional
from datetime import datetime
from crewai import Crew
from config.config import Config
from core.tasks import TaskFactory
from utils.analysis import MediaMention

class AdverseMediaProfiler:
    def __init__(self, config: Config):
        self.config = config
        self.task_factory = TaskFactory()

    def run_analysis(self, target: str, start_date: Optional[datetime] = None) -> str:
        """
        Run a complete adverse media analysis for the given target.
        
        Args:
            target: The name of the person or organization to analyze
            start_date: Optional date to limit the search to mentions after this date
            
        Returns:
            str: The final analysis report
        """
        # Create tasks
        collection_task = self.task_factory.create_collection_task(target, start_date)
        
        # Initialize the crew with the collector agent
        crew = Crew(
            tasks=[collection_task],
            verbose=True
        )
        
        # Run initial collection
        collected_mentions = crew.kickoff()
        
        if not collected_mentions:
            return "No adverse media mentions found."
        
        # Create and run triage task
        triage_task = self.task_factory.create_triage_task(collected_mentions)
        crew.tasks = [triage_task]
        filtered_mentions = crew.kickoff()
        
        # Create and run analysis task
        analysis_task = self.task_factory.create_analysis_task(filtered_mentions)
        crew.tasks = [analysis_task]
        analyzed_mentions = crew.kickoff()
        
        # Create and run reporting task
        reporting_task = self.task_factory.create_reporting_task(analyzed_mentions)
        crew.tasks = [reporting_task]
        final_report = crew.kickoff()
        
        return final_report
