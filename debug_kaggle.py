#!/usr/bin/env python3
"""
Debug script for Kaggle API connection
"""

import sys
from pathlib import Path
from rich.console import Console

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

console = Console()

def debug_kaggle():
    """Debug Kaggle API step by step"""
    
    console.print("[bold blue]🔍 Kaggle API Debug[/bold blue]\n")
    
    # Step 1: Check if kaggle package is installed
    try:
        import kaggle
        console.print("✅ Kaggle package installed")
    except ImportError:
        console.print("❌ Kaggle package not found. Install with: pip install kaggle")
        return
    
    # Step 2: Check if kaggle.json exists
    import os
    kaggle_json_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path("kaggle.json"),
        Path(".") / "kaggle.json"
    ]
    
    kaggle_json_found = False
    for path in kaggle_json_paths:
        if path.exists():
            console.print(f"✅ Found kaggle.json at: {path}")
            kaggle_json_found = True
            break
    
    if not kaggle_json_found:
        console.print("❌ kaggle.json not found in expected locations:")
        for path in kaggle_json_paths:
            console.print(f"   • {path}")
        console.print("\nPlace your kaggle.json in ~/.kaggle/kaggle.json")
        return
    
    # Step 3: Try to authenticate
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        console.print("✅ Kaggle API authenticated successfully")
    except Exception as e:
        console.print(f"❌ Kaggle authentication failed: {e}")
        return
    
    # Step 4: Test basic API call
    try:
        console.print("🔍 Testing basic API call...")
        competitions = api.competitions_list()
        console.print(f"✅ Basic API call works - found {len(competitions)} competitions")
    except Exception as e:
        console.print(f"❌ Basic API call failed: {e}")
        return
    
    # Step 5: Test dataset search
    try:
        console.print("🔍 Testing dataset search...")
        
        # First try a simple dataset_list call (no parameters)
        try:
            datasets = api.dataset_list()  # Try without any parameters first
            console.print(f"✅ Simple dataset_list(): found {len(datasets)} datasets")
            if datasets:
                console.print(f"   First result: {getattr(datasets[0], 'title', 'No title')}")
        except Exception as e:
            console.print(f"❌ Simple dataset_list() failed: {e}")
            
        # Try with sort parameter
        try:
            datasets = api.dataset_list(sort_by='hottest')  # Fixed sort parameter
            console.print(f"✅ dataset_list(sort_by='hottest'): found {len(datasets)} datasets")
            if datasets:
                console.print(f"   First result: {getattr(datasets[0], 'title', 'No title')}")
        except Exception as e:
            console.print(f"❌ dataset_list with sort failed: {e}")
        
        # Try different search terms
        search_terms = ["education", "climate", "population"]
        
        for term in search_terms:
            try:
                # Try search without sort first
                datasets = api.dataset_list(search=term)
                console.print(f"✅ Search '{term}' (no sort): found {len(datasets)} datasets")
                
                if datasets:
                    first_title = getattr(datasets[0], 'title', 'No title')
                    console.print(f"   First result: {first_title}")
                    
                    # Debug first dataset attributes
                    console.print(f"   Dataset attributes: {[attr for attr in dir(datasets[0]) if not attr.startswith('_')]}")
                    break
                else:
                    # Try with sort if no results
                    datasets = api.dataset_list(search=term, sort_by='hottest')
                    console.print(f"✅ Search '{term}' (with sort): found {len(datasets)} datasets")
                    if datasets:
                        first_title = getattr(datasets[0], 'title', 'No title')
                        console.print(f"   First result: {first_title}")
                        break
                        
            except Exception as e:
                console.print(f"❌ Search '{term}' failed: {e}")
    
    except Exception as e:
        console.print(f"❌ Dataset search test failed: {e}")
        return
    
    # Step 6: Test our connector
    try:
        console.print("\n🔍 Testing our Kaggle connector...")
        from config.settings import load_settings
        from tools.data_sources.kaggle_connector import KaggleConnector
        
        settings = load_settings()
        connector = KaggleConnector(
            username=settings.kaggle_username,
            api_key=settings.kaggle_key,
            cache_dir=settings.cache_dir,
            verbose=True
        )
        
        search_params = {
            "keywords": ["education", "spending"],
            "domain": "general",
            "data_types": ["any"],
            "geographic_scope": "any", 
            "time_relevance": "any",
            "priority_keywords": ["education"]
        }
        
        results = connector.search(search_params)
        console.print(f"✅ Our connector found {len(results)} datasets")
        
        if results:
            console.print("First 3 results:")
            for i, result in enumerate(results[:3], 1):
                console.print(f"   {i}. {result.get('title', 'No title')}")
                console.print(f"      Source: {result.get('source_api', 'unknown')}")
                console.print(f"      URL: {result.get('url', 'No URL')}")
        else:
            console.print("   [yellow]No results from our connector - this could be due to search term mismatch[/yellow]")
            console.print("   [dim]Try running with a simple term like 'data' or 'covid'[/dim]")
        
    except Exception as e:
        console.print(f"❌ Our connector failed: {e}")
        import traceback
        console.print(f"Error details: {traceback.format_exc()}")
        return
    
    console.print("\n✅ [bold green]All Kaggle tests passed![/bold green]")

if __name__ == "__main__":
    debug_kaggle()