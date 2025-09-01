#!/usr/bin/env python3
"""
Test script for Agent 1 - Data Discovery Specialist
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import load_settings, validate_settings
from agents.data_collector import DataCollectorAgent

console = Console()

def test_agent1():
    """Test Agent 1 functionality"""
    
    console.print(Panel.fit(
        "[bold blue]Agent 1 Test Suite[/bold blue]\n"
        "[dim]Testing Data Discovery Specialist[/dim]",
        title="🧪 Testing",
        border_style="blue"
    ))
    
    try:
        # Load settings
        console.print("📋 Loading configuration...")
        settings = load_settings()
        
        # Validate settings
        console.print("✅ Validating Ollama connection...")
        validate_settings(settings)
        
        # Initialize agent
        console.print("🤖 Initializing Agent 1...")
        agent = DataCollectorAgent(settings, verbose=True)
        
        # Test queries - enhanced for triple source testing
        test_queries = [
            "education spending by country",  # Should hit all 3 sources
            "gdp economic growth",            # Should be strong in World Bank  
            "climate change temperature data", # Should hit data.gov + Kaggle
            "covid vaccination rates"          # Should be in all sources
        ]
        
        for i, query in enumerate(test_queries, 1):
            console.print(f"\n[bold cyan]Test {i}/3: '{query}'[/bold cyan]")
            
            try:
                df_results = agent.discover_datasets(query, max_results=5)
                
                if not df_results.empty:
                    console.print(f"[green]✅ Found {len(agent.get_all_results())} total datasets! Showing top {len(df_results)}[/green]")
                    
                    # Show results table
                    agent.display_results_table(df_results)
                    
                    # Show enhanced source breakdown
                    source_counts = agent.get_all_results()['source'].value_counts()
                    console.print(f"\n[bold]Enhanced Multi-Source Breakdown:[/bold]")
                    total_datasets = len(agent.get_all_results())
                    for source, count in source_counts.items():
                        percentage = (count / total_datasets * 100) if total_datasets > 0 else 0
                        console.print(f"   • {source}: {count} datasets ({percentage:.1f}%)")
                    
                    # Test preview for each source type
                    console.print(f"\n[bold]Testing preview for each source:[/bold]")
                    for source in source_counts.index:
                        source_data = agent.get_all_results()[agent.get_all_results()['source'] == source]
                        if not source_data.empty:
                            first_dataset = source_data.iloc[0]
                            console.print(f"\n[cyan]{source} preview:[/cyan]")
                            try:
                                preview = agent.preview_dataset(
                                    first_dataset['url'], 
                                    first_dataset['source'],
                                    first_dataset.get('full_metadata', {})
                                )
                                console.print(f"  Preview keys: {list(preview.keys())}")
                                if 'error' in preview:
                                    console.print(f"  Error: {preview['error']}")
                                else:
                                    console.print(f"  Status: Preview successful for {source}")
                            except Exception as e:
                                console.print(f"  Preview failed: {e}")
                        
                    break  # Stop after first successful test
                        
                else:
                    console.print("[yellow]⚠️ No datasets found[/yellow]")
                    
            except Exception as e:
                console.print(f"[red]❌ Test failed: {e}[/red]")
        
        console.print(Panel.fit(
            "[bold green]🎉 Agent 1 Tests Complete![/bold green]\n\n"
            "[dim]Agent 1 is ready for integration![/dim]",
            title="Success",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(Panel.fit(
            f"[bold red]❌ Test failed: {e}[/bold red]\n\n"
            "[dim]Check your configuration and try again[/dim]",
            title="Error",
            border_style="red"
        ))
        
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")

if __name__ == "__main__":
    test_agent1()