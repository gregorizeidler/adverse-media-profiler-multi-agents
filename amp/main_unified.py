#!/usr/bin/env python3
"""
AMP (Adverse Media Profiler) - Unified Main Entry Point

Advanced Multi-Agent Intelligence System for Adverse Media Screening and Risk Assessment.
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import Config, APIConfig, SearchConfig
from core.multi_agent_orchestrator import run_multi_agent_analysis
from utils.analysis import AnalysisUtils

# ASCII Art Banner
BANNER = """
    ░█████╗░███╗░░░███╗██████╗░
    ██╔══██╗████╗░████║██╔══██╗
    ███████║██╔████╔██║██████╔╝
    ██╔══██║██║╚██╔╝██║██╔═══╝░
    ██║░░██║██║░╚═╝░██║██║░░░░░
    ╚═╝░░╚═╝╚═╝░░░░░╚═╝╚═╝░░░░░
    
    Adverse Media Profiler
    Multi-Agent Intelligence System
"""

def print_banner():
    """Print the application banner."""
    print("\033[94m" + BANNER + "\033[0m")

def load_environment_config() -> Config:
    """
    Load configuration from environment variables.
    
    Returns:
        Config object with API keys and settings
    """
    # Try to load from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not available, use system environment variables
    
    # Create API configuration
    api_config = APIConfig(
        openai_api_key=os.getenv('OPENAI_API_KEY', ''),
        news_api_key=os.getenv('NEWS_API_KEY', ''),
        serper_api_key=os.getenv('SERPER_API_KEY', '')
    )
    
    # Create search configuration
    search_config = SearchConfig(
        max_articles=int(os.getenv('MAX_ARTICLES', '50')),
        max_age_days=int(os.getenv('MAX_AGE_DAYS', '90'))
    )
    
    return Config(api=api_config, search=search_config)

def validate_config(config: Config) -> bool:
    """
    Validate that required configuration is present.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        True if configuration is valid
    """
    missing_keys = []
    
    if not config.api.openai_api_key:
        missing_keys.append('OPENAI_API_KEY')
    
    if missing_keys:
        print(f"\033[91mError: Missing required environment variables: {', '.join(missing_keys)}\033[0m")
        print("Please set these environment variables or create a .env file.")
        return False
    
    return True

def setup_directories():
    """Setup required directories for data and reports."""
    directories = [
        "data/news",
        "data/documents", 
        "data/databases",
        "data/reports",
        "data/reports/errors",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("📁 Directory structure initialized")

def print_analysis_progress(phase: str, message: str):
    """Print formatted progress messages."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\033[93m[{timestamp}] {phase}:\033[0m {message}")

async def interactive_analysis():
    """Run interactive analysis mode."""
    print("\n🔍 \033[96mInteractive Analysis Mode\033[0m")
    print("=" * 50)
    
    target = input("Enter target entity name: ").strip()
    if not target:
        print("Target name cannot be empty.")
        return
    
    print(f"\nStarting analysis for: \033[94m{target}\033[0m")
    
    # Additional information (optional)
    target_id = input("Enter target ID (CNPJ, CPF, etc.) [optional]: ").strip()
    
    additional_info = {}
    if target_id:
        additional_info['id'] = target_id
    
    # Load configuration
    config = load_environment_config()
    if not validate_config(config):
        return
    
    try:
        # Run analysis
        print("\n🚀 Starting multi-agent analysis...")
        result = await run_multi_agent_analysis(target, config, additional_info)
        
        # Display summary
        print_analysis_summary(result)
        
        # Offer to view detailed report
        view_detail = input("\nView detailed report? (y/n): ").lower() == 'y'
        if view_detail:
            print_detailed_report(result)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n\033[91m❌ Analysis failed: {str(e)}\033[0m")
        print("Check logs for detailed error information.")

def print_analysis_summary(report: Dict[str, Any]):
    """Print a summary of the analysis results."""
    print("\n" + "="*60)
    print("📊 \033[92mANALYSIS SUMMARY\033[0m")
    print("="*60)
    
    # Extract key information
    metadata = report.get('metadata', {})
    risk_assessment = report.get('risk_assessment', {})
    
    print(f"Target: \033[94m{metadata.get('target', {}).get('name', 'Unknown')}\033[0m")
    print(f"Analysis ID: {metadata.get('target', {}).get('analysis_id', 'N/A')}")
    print(f"Generated: {metadata.get('generated_at', 'N/A')}")
    
    # Risk assessment
    overall_score = risk_assessment.get('overall_risk_score', 0)
    risk_level = "LOW"
    color = "\033[92m"  # Green
    
    if overall_score > 0.7:
        risk_level = "HIGH"
        color = "\033[91m"  # Red
    elif overall_score > 0.4:
        risk_level = "MEDIUM"
        color = "\033[93m"  # Yellow
    
    print(f"\nOverall Risk Level: {color}{risk_level}\033[0m")
    print(f"Risk Score: {overall_score:.2f}")
    
    # Component risks
    financial_risk = risk_assessment.get('financial_risk', {})
    legal_risk = risk_assessment.get('legal_risk', {})
    reputational_risk = risk_assessment.get('reputational_risk', {})
    
    print(f"\n📈 Risk Breakdown:")
    if financial_risk:
        print(f"  • Financial: {get_risk_score(financial_risk):.2f}")
    if legal_risk:
        print(f"  • Legal: {get_risk_score(legal_risk):.2f}")
    if reputational_risk:
        print(f"  • Reputational: {get_risk_score(reputational_risk):.2f}")
    
    # Recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        print(f"\n💡 Key Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
            print(f"  {i}. {rec}")
    
    # Network analysis
    network_analysis = report.get('network_analysis', {})
    if network_analysis:
        network_size = network_analysis.get('network_size', 0)
        connections = network_analysis.get('connection_count', 0)
        print(f"\n🕸️  Network Analysis:")
        print(f"  • Entities identified: {network_size}")
        print(f"  • Connections mapped: {connections}")
    
    print("="*60)

def get_risk_score(risk_data: Dict) -> float:
    """Extract risk score from risk data structure."""
    if isinstance(risk_data, dict):
        # Try different possible keys
        for key in ['overall_score', 'score', 'risk_score']:
            if key in risk_data:
                return risk_data[key]
        
        # If it's a nested structure, try to find scores in sub-categories
        scores = []
        for value in risk_data.values():
            if isinstance(value, dict) and 'score' in value:
                scores.append(value['score'])
        
        if scores:
            return sum(scores) / len(scores)
    
    return 0.0

def print_detailed_report(report: Dict[str, Any]):
    """Print detailed analysis report."""
    print("\n" + "="*80)
    print("📋 \033[92mDETAILED ANALYSIS REPORT\033[0m")
    print("="*80)
    
    # Pretty print the JSON report
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))

async def batch_analysis(targets_file: str):
    """Run batch analysis from a file containing multiple targets."""
    print(f"\n📊 \033[96mBatch Analysis Mode\033[0m")
    print(f"Processing targets from: {targets_file}")
    
    try:
        with open(targets_file, 'r', encoding='utf-8') as f:
            targets = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File {targets_file} not found.")
        return
    
    config = load_environment_config()
    if not validate_config(config):
        return
    
    print(f"Found {len(targets)} targets to analyze")
    
    results = []
    
    for i, target in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] Analyzing: {target}")
        
        try:
            result = await run_multi_agent_analysis(target, config)
            results.append({
                'target': target,
                'success': True,
                'result': result
            })
            
            # Quick summary
            overall_score = result.get('risk_assessment', {}).get('overall_risk_score', 0)
            print(f"  ✅ Completed - Risk Score: {overall_score:.2f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            results.append({
                'target': target,
                'success': False,
                'error': str(e)
            })
    
    # Save batch results
    batch_results_file = f"data/reports/batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(batch_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📄 Batch results saved to: {batch_results_file}")
    
    # Summary statistics
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n📊 Batch Summary:")
    print(f"  • Total analyzed: {len(results)}")
    print(f"  • Successful: {successful}")
    print(f"  • Failed: {failed}")
    print(f"  • Success rate: {(successful/len(results)*100):.1f}%")

def create_sample_data():
    """Create sample data files for testing."""
    print("🗂️ Creating sample data files...")
    
    # Sample news articles
    sample_news = [
        {
            "title": "Example Corp Under Investigation",
            "content": "Financial authorities are investigating Example Corp for potential fraud...",
            "source": "Financial Times",
            "date": "2024-01-15",
            "url": "https://example.com/news1"
        }
    ]
    
    # Sample sanctions list
    sample_sanctions = [
        {
            "name": "Bad Actor Corp",
            "id": "123.456/0001-00",
            "list_name": "OFAC",
            "date_listed": "2024-01-01",
            "reason": "Financial sanctions",
            "status": "active"
        }
    ]
    
    # Save sample data
    Path("data/news").mkdir(parents=True, exist_ok=True)
    Path("data/databases").mkdir(parents=True, exist_ok=True)
    
    with open("data/news/sample_article.txt", "w", encoding="utf-8") as f:
        f.write(sample_news[0]["content"])
    
    with open("data/databases/sanctions.json", "w", encoding="utf-8") as f:
        json.dump(sample_sanctions, f, indent=2, ensure_ascii=False)
    
    print("✅ Sample data created")

async def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="AMP - Adverse Media Profiler: Multi-Agent Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_unified.py                          # Interactive mode
  python main_unified.py --target "Example Corp" # Direct analysis
  python main_unified.py --batch targets.txt     # Batch analysis
  python main_unified.py --setup                 # Setup sample data
        """
    )
    
    parser.add_argument('--target', '-t', help='Target entity name for analysis')
    parser.add_argument('--id', help='Target ID (CNPJ, CPF, etc.)')
    parser.add_argument('--batch', '-b', help='Batch analysis from file')
    parser.add_argument('--setup', action='store_true', help='Setup sample data')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode (no banner)')
    parser.add_argument('--output', '-o', help='Output file for results')
    
    args = parser.parse_args()
    
    # Print banner unless in quiet mode
    if not args.quiet:
        print_banner()
    
    # Setup directories
    setup_directories()
    
    # Handle setup mode
    if args.setup:
        create_sample_data()
        return
    
    # Handle batch mode
    if args.batch:
        await batch_analysis(args.batch)
        return
    
    # Handle direct analysis mode
    if args.target:
        config = load_environment_config()
        if not validate_config(config):
            return
        
        additional_info = {}
        if args.id:
            additional_info['id'] = args.id
        
        print(f"🎯 Analyzing target: \033[94m{args.target}\033[0m")
        
        try:
            result = await run_multi_agent_analysis(args.target, config, additional_info)
            
            # Save result if output specified
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                print(f"📄 Results saved to: {args.output}")
            else:
                print_analysis_summary(result)
            
        except Exception as e:
            print(f"\033[91m❌ Analysis failed: {str(e)}\033[0m")
            return 1
        
        return 0
    
    # Default to interactive mode
    await interactive_analysis()
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code or 0)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\033[91m💥 Fatal error: {str(e)}\033[0m")
        sys.exit(1)
