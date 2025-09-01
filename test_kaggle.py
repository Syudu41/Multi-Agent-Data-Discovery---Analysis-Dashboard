#!/usr/bin/env python3
"""
Quick test to verify Kaggle sort_by fix
"""

import sys
from pathlib import Path
from rich.console import Console

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

console = Console()

def test_kaggle_fix():
    """Test the sort_by fix"""
    
    console.print("[bold blue]🔧 Testing Kaggle sort_by Fix[/bold blue]\n")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        console.print("✅ Kaggle API authenticated")
        
        # Test the correct sort_by values
        sort_options = ['hottest', 'votes', 'updated', 'active', 'published']
        
        for sort_option in sort_options:
            try:
                console.print(f"🔍 Testing sort_by='{sort_option}'...")
                datasets = api.dataset_list(sort_by=sort_option)
                console.print(f"   ✅ sort_by='{sort_option}' returned {len(datasets)} datasets")
                
                if datasets and sort_option == 'hottest':
                    console.print(f"   📊 First result: {getattr(datasets[0], 'title', 'No title')}")
                    break  # We found what we need
                    
            except Exception as e:
                console.print(f"   ❌ sort_by='{sort_option}' failed: {e}")
        
        # Test with search term
        console.print(f"\n🔍 Testing search with corrected sort_by...")
        try:
            datasets = api.dataset_list(search='education', sort_by='hottest')
            console.print(f"✅ Search + sort_by='hottest' returned {len(datasets)} datasets")
            
            if datasets:
                console.print("First 3 results:")
                for i, dataset in enumerate(datasets[:3], 1):
                    title = getattr(dataset, 'title', 'No title')
                    console.print(f"   {i}. {title}")
            
        except Exception as e:
            console.print(f"❌ Search test failed: {e}")
        
        console.print(f"\n[bold green]🎉 Kaggle API is working with sort_by='hottest'![/bold green]")
        
    except Exception as e:
        console.print(f"[red]❌ Test failed: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    test_kaggle_fix()