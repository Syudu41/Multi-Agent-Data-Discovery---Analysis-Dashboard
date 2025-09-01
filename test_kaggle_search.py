#!/usr/bin/env python3
"""
Test script specifically for Kaggle search functionality
"""

import sys
from pathlib import Path
from rich.console import Console

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

console = Console()

def test_kaggle_search():
    """Test Kaggle search specifically"""
    
    console.print("[bold blue]🔍 Kaggle Search Test[/bold blue]\n")
    
    try:
        from config.settings import load_settings
        from tools.data_sources.kaggle_connector import KaggleConnector
        
        # Load settings
        settings = load_settings()
        
        # Initialize Kaggle connector directly
        kaggle = KaggleConnector(
            username=settings.kaggle_username,
            api_key=settings.kaggle_key,
            cache_dir=settings.cache_dir,
            verbose=True
        )
        
        # Test different search terms
        test_terms = [
            {"keywords": ["education"], "priority_keywords": ["education"]},
            {"keywords": ["covid"], "priority_keywords": ["covid"]},
            {"keywords": ["climate"], "priority_keywords": ["climate"]},
            {"keywords": ["data"], "priority_keywords": ["data"]},
        ]
        
        for i, search_params in enumerate(test_terms, 1):
            keyword = search_params["keywords"][0]
            console.print(f"\n[cyan]Test {i}: Searching Kaggle for '{keyword}'[/cyan]")
            
            try:
                results = kaggle.search(search_params)
                console.print(f"[green]✅ Found {len(results)} results[/green]")
                
                if results:
                    console.print("First 3 results:")
                    for j, result in enumerate(results[:3], 1):
                        console.print(f"   {j}. {result.get('title', 'No title')}")
                        console.print(f"      ID: {result.get('id', 'No ID')}")
                        console.print(f"      Downloads: {result.get('downloadCount', 0)}")
                        console.print(f"      Votes: {result.get('voteCount', 0)}")
                else:
                    console.print("   [yellow]No results found for this term[/yellow]")
                    
            except Exception as e:
                console.print(f"[red]❌ Search failed: {e}[/red]")
                import traceback
                console.print(traceback.format_exc())
        
        # Test with agent
        console.print(f"\n[cyan]Testing with full agent integration:[/cyan]")
        from agents.data_collector import DataCollectorAgent
        
        agent = DataCollectorAgent(settings, verbose=True)
        
        # Simple search
        df_results = agent.discover_datasets("education", max_results=10)
        
        if not df_results.empty:
            all_results = agent.get_all_results()
            source_counts = all_results['source'].value_counts()
            
            console.print(f"\n[bold]Final Results Summary:[/bold]")
            console.print(f"Total datasets found: {len(all_results)}")
            
            for source, count in source_counts.items():
                console.print(f"   • {source}: {count} datasets")
                
            # Show some Kaggle results if any
            kaggle_results = all_results[all_results['source'] == 'kaggle']
            if not kaggle_results.empty:
                console.print(f"\n[bold]Sample Kaggle Results:[/bold]")
                for i, (_, row) in enumerate(kaggle_results.head(3).iterrows(), 1):
                    console.print(f"   {i}. {row['title']}")
                    console.print(f"      Score: {row['overall_score']}")
            else:
                console.print("[red]❌ No Kaggle results in final output![/red]")
        else:
            console.print("[red]❌ Agent returned no results[/red]")
            
    except Exception as e:
        console.print(f"[red]❌ Test failed: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    test_kaggle_search()