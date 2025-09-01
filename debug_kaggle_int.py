#!/usr/bin/env python3
"""
Debug DataCollectorAgent integration with Kaggle
"""

import sys
from pathlib import Path
from rich.console import Console

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

console = Console()

def debug_agent_integration():
    """Debug step-by-step how DataCollectorAgent processes Kaggle data"""
    
    console.print("[bold blue]🔍 Debugging Agent Integration[/bold blue]\n")
    
    # Add cache management option
    console.print("[bold yellow]Cache Management:[/bold yellow]")
    console.print("This will clear all cached API results to test fresh data.")
    console.print("Press Enter to continue, or Ctrl+C to exit...")
    try:
        input()
    except KeyboardInterrupt:
        console.print("\nExiting...")
        return
    
    try:
        from config.settings import load_settings
        from agents.data_collector import DataCollectorAgent
        
        # Load settings
        settings = load_settings()
        
        # Initialize agent with verbose mode
        console.print("🤖 Initializing DataCollectorAgent...")
        agent = DataCollectorAgent(settings, verbose=True)
        
        # CLEAR THE CACHE - this is the fix!
        console.print("\n🧹 Clearing cache to test fresh results...")
        agent.kaggle.clear_cache()
        agent.data_gov.clear_cache()
        console.print("✅ Cache cleared!")
        
        # Test the search parameters generation
        console.print("\n📝 Testing query parsing...")
        query = "education"
        search_params = agent._parse_query_with_llm(query)
        console.print(f"Parsed search params: {search_params}")
        
        # Test Kaggle connector directly through agent
        console.print(f"\n🏆 Testing Kaggle connector through agent (fresh, no cache)...")
        kaggle_results = agent.kaggle.search(search_params)
        console.print(f"Direct Kaggle results: {len(kaggle_results)}")
        
        if kaggle_results:
            console.print("First 3 Kaggle results:")
            for i, result in enumerate(kaggle_results[:3], 1):
                console.print(f"   {i}. {result.get('title', 'No title')}")
                console.print(f"      Source: {result.get('source_api', 'unknown')}")
        else:
            console.print("❌ Still no Kaggle results - API issue persists")
        
        # Test data.gov connector
        console.print(f"\n🏛️ Testing data.gov connector...")
        datagov_results = agent.data_gov.search(search_params)
        console.print(f"data.gov results: {len(datagov_results)}")
        
        # Test the combine step
        console.print(f"\n🔄 Testing result combination...")
        all_datasets = agent._combine_results(datagov_results, kaggle_results)
        console.print(f"Combined datasets: {len(all_datasets)}")
        
        # Check source breakdown in combined results
        source_counts = {}
        for dataset in all_datasets:
            source = dataset.source
            source_counts[source] = source_counts.get(source, 0) + 1
        
        console.print(f"Source breakdown after combining: {source_counts}")
        
        # Test ranking step
        console.print(f"\n⚖️ Testing ranking...")
        ranked_datasets = agent._rank_datasets(all_datasets, query, search_params)
        console.print(f"Ranked datasets: {len(ranked_datasets)}")
        
        # Check source breakdown after ranking
        source_counts_ranked = {}
        for dataset in ranked_datasets:
            source = dataset.source
            source_counts_ranked[source] = source_counts_ranked.get(source, 0) + 1
        
        console.print(f"Source breakdown after ranking: {source_counts_ranked}")
        
        # Show top 5 ranked results with source info
        console.print(f"\n🏆 Top 5 results after ranking:")
        for i, dataset in enumerate(ranked_datasets[:5], 1):
            console.print(f"   {i}. [{dataset.source}] {dataset.title}")
            console.print(f"      Score: {dataset.overall_score:.2f}")
        
        # Now test the full discover_datasets method WITH cache cleared
        console.print(f"\n🎯 Testing full discover_datasets method (with cleared cache)...")
        
        # Create a fresh agent to ensure no cached data
        fresh_agent = DataCollectorAgent(settings, verbose=True)
        fresh_agent.kaggle.clear_cache()
        fresh_agent.data_gov.clear_cache()
        
        df_results = fresh_agent.discover_datasets(query, max_results=10)
        
        if not df_results.empty:
            all_results = fresh_agent.get_all_results()
            source_counts_final = all_results['source'].value_counts()
            
            console.print(f"\n[bold]Final Results in DataFrame:[/bold]")
            console.print(f"Total: {len(all_results)}")
            for source, count in source_counts_final.items():
                console.print(f"   • {source}: {count} datasets")
                
            # Show sample of each source
            for source in source_counts_final.index:
                source_data = all_results[all_results['source'] == source]
                console.print(f"\n[bold]Sample {source} results:[/bold]")
                for i, (_, row) in enumerate(source_data.head(2).iterrows(), 1):
                    console.print(f"   {i}. {row['title'][:60]}...")
        else:
            console.print("[red]❌ discover_datasets returned empty DataFrame[/red]")
            
    except Exception as e:
        console.print(f"[red]❌ Debug failed: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    debug_agent_integration()