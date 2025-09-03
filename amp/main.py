from crewai import Agent, Task, Crew
from agents.agent_definitions import get_all_agents
from pathlib import Path
import argparse
import json

class AdverseMediaProfiler:
    def __init__(self, target_info: dict):
        self.target_info = target_info
        self.data_dir = Path("data")
        self.setup_directories()
        self.agents = get_all_agents()
        
    def setup_directories(self):
        """
        Sets up the necessary directory structure.
        """
        dirs = ["news", "documents", "databases", "reports"]
        for dir_name in dirs:
            (self.data_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def create_tasks(self) -> list:
        """
        Creates the sequence of tasks for the crew to execute.
        """
        [orchestrator, media_hunter, list_checker, triage_entity, 
         financial_analyst, legal_analyst, reputation_analyst,
         connections_detective, quality_auditor, dossier_compiler] = self.agents
        
        tasks = []
        
        # Fase de Coleta (Paralela)
        tasks.append(Task(
            description=f"Search and analyze local sources for information about {self.target_info['name']}",
            agent=media_hunter
        ))
        
        tasks.append(Task(
            description=f"Check local databases for any matches with {self.target_info['name']}",
            agent=list_checker
        ))
        
        # Fase de Triagem
        tasks.append(Task(
            description="Filter and verify all entity mentions, remove false positives",
            agent=triage_entity
        ))
        
        # Specialized Analysis Phase (Parallel)
        tasks.append(Task(
            description="Analyze financial risks and identify potential financial crimes",
            agent=financial_analyst
        ))
        
        tasks.append(Task(
            description="Analyze legal and criminal risks",
            agent=legal_analyst
        ))
        
        tasks.append(Task(
            description="Analyze reputational and ethical risks",
            agent=reputation_analyst
        ))
        
        # Network Analysis Phase
        tasks.append(Task(
            description="Map and analyze relationships between identified entities",
            agent=connections_detective
        ))
        
        # Fase de Auditoria
        tasks.append(Task(
            description="Verify and validate all findings for accuracy and reliability",
            agent=quality_auditor
        ))
        
        # Compilation Phase
        tasks.append(Task(
            description="Compile all verified findings into a comprehensive final report",
            agent=dossier_compiler
        ))
        
        return tasks
    
    def run(self) -> dict:
        """
        Executes the complete analysis process.
        """
        # Create tasks
        tasks = self.create_tasks()
        
        # Criar e executar o crew
        crew = Crew(
            agents=self.agents,
            tasks=tasks,
            verbose=True
        )
        
        # Executar o processo
        result = crew.kickoff()
        
        # Salvar o resultado
        report_path = self.data_dir / "reports" / f"{self.target_info['name']}_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result

def main():
    parser = argparse.ArgumentParser(description="Adverse Media Profiler")
    parser.add_argument("--name", required=True, help="Nome do alvo")
    parser.add_argument("--id", help="Identificador do alvo (CNPJ, CPF, etc.)")
    args = parser.parse_args()
    
    target_info = {
        "name": args.name,
        "id": args.id
    }
    
    profiler = AdverseMediaProfiler(target_info)
    result = profiler.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
