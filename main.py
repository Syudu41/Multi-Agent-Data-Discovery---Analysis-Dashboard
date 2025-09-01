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
@click.option('--clear-cache', '-c', is_flag=True, help='Clear cached results before searching')
def main(max_results: int, verbose: bool, clear_cache: bool):
    """
    PersonalDataAI - Ask natural language questions about public datasets
    
    Interactive mode: The system will ask you for your question
    
    Options:
    --clear-cache: Clear cached results to get fresh data from APIs
    --verbose: Show detailed output
    --max-results: Maximum results to show initially
    
    Example:
    python main.py --clear-cache --verbose
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
        f"[green]Max Results:[/green] {max_results}\n"
        f"[green]Clear Cache:[/green] {'Yes' if clear_cache else 'No'}\n\n"
        "[dim]💡 Example questions:[/dim]\n"
        "[dim]• How does education spending correlate with literacy rates?[/dim]\n"
        "[dim]• Climate change temperature data by country[/dim]\n"
        "[dim]• Population demographics United States[/dim]",
        title="🤖 Agent System Ready",
        border_style="blue"
    ))
    
    # Initialize agent once
    agent = DataCollectorAgent(settings, verbose=verbose)
    
    # Clear cache if requested
    if clear_cache:
        console.print("\n[yellow]🧹 Clearing cached results...[/yellow]")
        agent.kaggle.clear_cache()
        agent.data_gov.clear_cache()
        console.print("[green]✅ Cache cleared![/green]")
    
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
            
            console.print("\n[yellow]🔍 Agent 1: Data Discovery Specialist - Starting...[/yellow]")
            
            # Execute data discovery
            df_results = agent.discover_datasets(query, max_results=max_results)
            
            if df_results.empty:
                console.print("[yellow]📭 No relevant datasets found for your query.[/yellow]")
                console.print("[dim]💡 Try these suggestions:[/dim]")
                console.print("[dim]  • Use simpler keywords (e.g., 'education' instead of 'educational attainment')[/dim]")
                console.print("[dim]  • Try different terms (e.g., 'climate' instead of 'global warming')[/dim]")
                console.print("[dim]  • Use broader categories (e.g., 'health' instead of 'cardiovascular disease')[/dim]")
                continue
                
            # Display results in table format
            console.print(f"\n[green]✅ Found {len(agent.get_all_results())} total datasets! Showing top {len(df_results)}[/green]")
            
            # Show source breakdown
            all_results = agent.get_all_results()
            source_counts = all_results['source'].value_counts()
            console.print(f"\n[bold]📊 Source Breakdown:[/bold]")
            for source, count in source_counts.items():
                console.print(f"   • {source}: {count} datasets")
            
            console.print("\n[bold cyan]📊 Available Datasets:[/bold cyan]")
            
            agent.display_results_table(df_results)
            
            # Options for user
            console.print("\n[bold yellow]Options:[/bold yellow]")
            console.print("• [dim]Type 'more' to see more results[/dim]")
            console.print("• [dim]Type 'all' to see all results[/dim]")  
            console.print("• [dim]Type 'preview <rank>' to preview a dataset[/dim]")
            console.print("• [dim]Press Enter for a new search[/dim]")
            
            while True:
                user_input = console.input("\n[yellow]>>[/yellow] ").strip().lower()
                
                if not user_input:
                    break  # New search
                elif user_input == 'more':
                    try:
                        more_results = agent.get_more_results(count=10)
                        if not more_results.empty:
                            console.print(f"\n[green]Next {len(more_results)} results (showing {agent.current_display_position-len(more_results)+1}-{agent.current_display_position}):[/green]")
                            agent.display_results_table(more_results)
                        else:
                            console.print("[yellow]No more results available[/yellow]")
                    except Exception as e:
                        console.print(f"[red]Error: {e}[/red]")
                        
                elif user_input == 'all':
                    try:
                        all_results = agent.get_all_results()
                        console.print(f"\n[green]All {len(all_results)} results:[/green]")
                        agent.display_results_table(all_results, max_description_length=60)  # Shorter descriptions for full list
                    except Exception as e:
                        console.print(f"[red]Error: {e}[/red]")
                        
                elif user_input.startswith('preview '):
                    try:
                        rank = int(user_input.split()[1])
                        all_results = agent.get_all_results()
                        if 1 <= rank <= len(all_results):
                            dataset = all_results.iloc[rank-1]
                            preview = agent.preview_dataset(
                                dataset['url'], 
                                dataset['source'], 
                                dataset.get('full_metadata', {})
                            )
                            
                            console.print(f"\n[bold cyan]Preview of: {dataset['title']}[/bold cyan]")
                            console.print(f"[green]Source:[/green] {dataset['source']}")
                            console.print(f"[green]URL:[/green] {dataset['url']}")
                            
                            if 'error' in preview:
                                console.print(f"[yellow]⚠️ {preview['error']}[/yellow]")
                                
                                # Show fallback info if available
                                if 'fallback_info' in preview:
                                    console.print("[green]Available Info:[/green]")
                                    fallback = preview['fallback_info']
                                    for key, value in fallback.items():
                                        console.print(f"   • {key.replace('_', ' ').title()}: {value}")
                                
                                if 'note' in preview:
                                    console.print(f"[dim]{preview['note']}[/dim]")
                            
                            elif dataset['source'] == 'data.gov':
                                console.print(f"[green]Total Resources:[/green] {preview['total_resources']}")
                                console.print(f"[green]Available Formats:[/green] {', '.join(preview['formats']) if preview['formats'] else 'N/A'}")
                                if preview['resources']:
                                    console.print("\n[bold]Resources:[/bold]")
                                    for res in preview['resources'][:3]:  # Show first 3
                                        console.print(f"  • {res['name']} ({res['format']})")
                            
                            elif dataset['source'] == 'kaggle':
                                console.print(f"[green]Total Files:[/green] {preview.get('total_files', 'N/A')}")
                                if preview.get('file_count_fallback', 0) > 0:
                                    console.print(f"[green]Metadata File Count:[/green] {preview['file_count_fallback']}")
                                console.print(f"[green]Downloads:[/green] {preview.get('download_count', 'N/A')}")
                                console.print(f"[green]Votes:[/green] {preview.get('vote_count', 'N/A')}")
                                if preview.get('dataset_ref'):
                                    console.print(f"[green]Dataset ID:[/green] {preview['dataset_ref']}")
                                if preview.get('files'):
                                    console.print("\n[bold]Files:[/bold]")
                                    for file in preview['files'][:3]:  # Show first 3
                                        console.print(f"  • {file['name']} ({file.get('size', 'Unknown size')})")
                            
                            elif dataset['source'] == 'worldbank':
                                console.print(f"[green]Indicator ID:[/green] {preview.get('indicator_id', 'N/A')}")
                                console.print(f"[green]Source:[/green] {preview.get('source_organization', 'World Bank')}")
                                console.print(f"[green]Last Updated:[/green] {preview.get('last_updated', 'N/A')}")
                                if preview.get('topics'):
                                    console.print(f"[green]Topics:[/green] {', '.join(preview['topics'][:3])}")
                                if preview.get('note'):
                                    console.print(f"[dim]{preview['note']}[/dim]")
                        else:
                            console.print(f"[red]Invalid rank. Please choose 1-{len(all_results)}[/red]")
                    except (ValueError, IndexError):
                        console.print("[red]Invalid command. Use: preview <rank_number>[/red]")
                    except Exception as e:
                        console.print(f"[red]Error: {e}[/red]")
                else:
                    console.print("[yellow]Unknown command. Try 'more', 'all', 'preview <rank>', or press Enter for new search.[/yellow]")
            
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