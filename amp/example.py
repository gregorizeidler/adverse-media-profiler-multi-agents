import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from config.config import Config, APIConfig, SearchConfig
from core.profiler import AdverseMediaProfiler

def main():
    # Load environment variables
    load_dotenv()
    
    # Create configuration
    config = Config(
        api=APIConfig(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            news_api_key=os.getenv('NEWS_API_KEY'),
            serper_api_key=os.getenv('SERPER_API_KEY')
        ),
        search=SearchConfig(
            max_articles=50,
            max_age_days=90
        )
    )
    
    # Initialize the profiler
    profiler = AdverseMediaProfiler(config)
    
    # Example: Run analysis for a target
    target = "Example Corp"
    start_date = datetime.now() - timedelta(days=90)  # Look back 90 days
    
    print(f"Starting adverse media analysis for {target}")
    report = profiler.run_analysis(target, start_date)
    
    print("\nAnalysis Report:")
    print("-" * 50)
    print(report)

if __name__ == "__main__":
    main()
