#!/usr/bin/env python3
"""
PersonalDataAI - Multi-Agent Data Discovery & Analysis
Main CLI Entry Point
"""

import sys
import os
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.data_collector import DataCollectorAgent
from config.settings import load_settings

console = Console()

@click.command()
@click.option('--max-results', '-m', default=10, help='Maximum number of datasets to find')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def main(max_results: int, verbose: bool):
    """
    PersonalDataAI - Ask natural language questions about public datasets
    
    Interactive mode: The system will ask you for your question
    
    Example:
    python main.py
    > What would you like to know? How does education spending correlate with literacy rates?
    """
    
    # Load configuration
    try:
        settings = load_settings()
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)
    
    # Display startup banner
    console.print(Panel.fit(
        "[bold blue]PersonalDataAI[/bold blue]\n"
        "[dim]Multi-Agent Data Discovery & Analysis[/dim]\n\n"
        f"[green]Model:[/green] {settings.ollama_model}\n"
        f"[green]Max Results:[/green] {max_results}\n\n"
        "[dim]💡 Example questions:[/dim]\n"
        "[dim]• How does education spending correlate with literacy rates?[/dim]\n"
        "[dim]• Climate change temperature data by country[/dim]\n"
        "[dim]• Population demographics United States[/dim]",
        title="🤖 Agent System Ready",
        border_style="blue"
    ))
    
    while True:
        try:
            # Interactive prompt for user query
            console.print("\n" + "="*60)
            query = console.input(
                "[bold cyan]❓ What would you like to know about data?[/bold cyan]\n"
                "[dim](Press Ctrl+C to exit)[/dim]\n"
                "[yellow]>>[/yellow] "
            ).strip()
            
            if not query:
                console.print("[yellow]Please enter a question![/yellow]")
                continue
                
            console.print(f"\n[green]🔍 Searching for:[/green] [bold]{query}[/bold]")
            
            # Initialize Agent 1: Data Discovery
            agent = DataCollectorAgent(settings, verbose=verbose)
            console.print("\n[yellow]🔍 Agent 1: Data Discovery Specialist - Starting...[/yellow]")
            
            # Execute data discovery
            datasets = agent.discover_datasets(query, max_results=max_results)
            
            if not datasets:
                console.print("[red]No datasets found for your query. Try rephrasing or using different keywords.[/red]")
                continue
                
            # Display results
            console.print(f"\n[green]✅ Found {len(datasets)} relevant datasets![/green]")
            
            # Show dataset options
            console.print("\n[bold cyan]📊 Available Datasets:[/bold cyan]")
            for i, dataset in enumerate(datasets, 1):
                console.print(f"[dim]{i}.[/dim] [bold]{dataset['title']}[/bold]")
                console.print(f"   [green]Source:[/green] {dataset['source']}")
                console.print(f"   [green]Score:[/green] {dataset['relevance_score']:.2f}/10")
                if dataset.get('description'):
                    desc = dataset['description'][:100] + "..." if len(dataset['description']) > 100 else dataset['description']
                    console.print(f"   [dim]{desc}[/dim]")
                console.print()
            
            # Ask if user wants to continue
            console.print(Panel.fit(
                "[bold green]🎉 Agent 1 Complete![/bold green]\n\n"
                "[dim]Next Steps:[/dim]\n"
                "• Agent 2: Data Processing (Coming Soon)\n"
                "• Agent 3: Pattern Analysis (Coming Soon)\n"
                "• Agent 4: Report Generation (Coming Soon)",
                title="Status Update",
                border_style="green"
            ))
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]👋 Thanks for using PersonalDataAI![/yellow]")
            break
            
        except Exception as e:
            console.print(f"[red]❌ Error in Agent 1: {e}[/red]")
            if verbose:
                import traceback
                console.print(f"[red]{traceback.format_exc()}[/red]")
            console.print("[dim]Try again with a different query...[/dim]")

if __name__ == "__main__":
    main()