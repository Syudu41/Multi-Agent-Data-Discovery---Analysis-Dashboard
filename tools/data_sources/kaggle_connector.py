"""
Kaggle API connector

Searches Kaggle datasets using the official Kaggle API
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from rich.console import Console

from .base_connector import BaseConnector

class KaggleConnector(BaseConnector):
    """Connector for Kaggle datasets using official API"""
    
    def __init__(self, username: Optional[str] = None, api_key: Optional[str] = None, 
                 cache_dir: Path = Path("./cache"), verbose: bool = False):
        super().__init__(cache_dir, verbose=verbose)
        
        self.username = username
        self.api_key = api_key
        self.console = Console() if verbose else None
        
        # Try to initialize Kaggle API
        self.kaggle_api = self._setup_kaggle_api()
        
        if self.verbose:
            status = "✅ Connected" if self.kaggle_api else "❌ Not available"
            self.console.print(f"[blue]🏆 Kaggle connector initialized - {status}[/blue]")
    
    def _setup_kaggle_api(self):
        """Setup Kaggle API client"""
        
        try:
            # Import kaggle API
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            
            # Setup authentication
            if self.username and self.api_key:
                # Use provided credentials
                os.environ['KAGGLE_USERNAME'] = self.username
                os.environ['KAGGLE_KEY'] = self.api_key
                
            # Authenticate
            api.authenticate()
            
            if self.verbose:
                self.console.print("[green]✅ Kaggle API authenticated[/green]")
            
            return api
            
        except ImportError:
            if self.verbose:
                self.console.print("[yellow]⚠️ Kaggle package not installed. Run: pip install kaggle[/yellow]")
            return None
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]⚠️ Kaggle API setup failed: {e}[/yellow]")
                self.console.print("[dim]Get API key from: https://www.kaggle.com/account[/dim]")
            return None
    
    def search(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search Kaggle for datasets"""
        
        if not self.kaggle_api:
            if self.verbose:
                self.console.print("[yellow]⚠️ Kaggle API not available, skipping...[/yellow]")
            return []
        
        if self.verbose:
            self.console.print(f"[dim]🔍 Searching Kaggle with: {search_params}[/dim]")
        
        def _make_request():
            return self._search_datasets(search_params)
        
        return self._make_request_with_cache("search", search_params, _make_request)
    
    def _search_datasets(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Internal method to search Kaggle datasets"""
        
        try:
            # Build search query
            keywords = search_params.get('keywords', [])
            priority_keywords = search_params.get('priority_keywords', [])
            
            # Combine keywords
            all_keywords = list(set(priority_keywords + keywords))
            search_query = ' '.join(all_keywords[:5]) if all_keywords else 'data'  # Limit keywords
            
            # Search parameters
            search_kwargs = {
                'search': search_query,
                'sort_by': 'hotness',  # Most popular/relevant first
                'page': 1,
                'page_size': 50  # Maximum per request
            }
            
            if self.verbose:
                self.console.print(f"[dim]📡 Kaggle API search: '{search_query}'[/dim]")
            
            # Make the API call
            datasets = self.kaggle_api.dataset_list(**search_kwargs)
            
            if self.verbose:
                self.console.print(f"[green]✅ Found {len(datasets)} datasets on Kaggle[/green]")
            
            # Convert to our format
            cleaned_datasets = []
            for dataset in datasets:
                cleaned = self._clean_dataset(dataset)
                if cleaned:
                    cleaned_datasets.append(cleaned)
            
            return cleaned_datasets
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[red]❌ Kaggle API error: {e}[/red]")
            return []
    
    def _clean_dataset(self, raw_dataset) -> Optional[Dict[str, Any]]:
        """Clean and standardize Kaggle dataset information"""
        
        try:
            # Kaggle dataset objects have specific attributes
            dataset_ref = raw_dataset.ref
            title = raw_dataset.title or 'Untitled Dataset'
            subtitle = getattr(raw_dataset, 'subtitle', '')
            
            # Build URL
            dataset_url = f"https://www.kaggle.com/datasets/{dataset_ref}"
            
            # Get metadata
            owner_name = getattr(raw_dataset, 'creatorName', '')
            download_count = getattr(raw_dataset, 'downloadCount', 0)
            vote_count = getattr(raw_dataset, 'voteCount', 0)
            
            # Dates
            creation_date = getattr(raw_dataset, 'creationDate', '')
            last_updated = getattr(raw_dataset, 'lastUpdated', '')
            
            # Size and files
            total_bytes = getattr(raw_dataset, 'totalBytes', 0)
            file_count = getattr(raw_dataset, 'fileCount', 0)
            
            # Tags
            tags = getattr(raw_dataset, 'tags', [])
            if hasattr(tags, '__iter__') and not isinstance(tags, str):
                tag_names = [tag.name if hasattr(tag, 'name') else str(tag) for tag in tags]
            else:
                tag_names = []
            
            cleaned = {
                'id': dataset_ref,
                'title': title,
                'description': subtitle,
                'subtitle': subtitle,  # Keep original field name
                'url': dataset_url,
                'owner': owner_name,
                'downloadCount': download_count,
                'voteCount': vote_count,
                'creationDate': str(creation_date) if creation_date else '',
                'lastUpdated': str(last_updated) if last_updated else '',
                'totalBytes': total_bytes,
                'fileCount': file_count,
                'tags': tag_names,
                'source_api': 'kaggle',
                'raw_data': {
                    'ref': dataset_ref,
                    'title': title,
                    'subtitle': subtitle,
                    'owner': owner_name,
                    'downloadCount': download_count,
                    'voteCount': vote_count,
                    'totalBytes': total_bytes,
                    'fileCount': file_count
                }
            }
            
            return cleaned
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]⚠️ Error cleaning Kaggle dataset: {e}[/yellow]")
            return None
    
    def get_dataset_details(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific Kaggle dataset"""
        
        if not self.kaggle_api:
            return {}
        
        def _make_request():
            return self._get_dataset_details(dataset_id)
        
        return self._make_request_with_cache(
            "details", 
            {"dataset_id": dataset_id}, 
            _make_request
        )
    
    def _get_dataset_details(self, dataset_id: str) -> Dict[str, Any]:
        """Internal method to get Kaggle dataset details"""
        
        try:
            # dataset_id should be in format "owner/dataset-name"
            if '/' not in dataset_id:
                return {}
            
            owner, dataset_name = dataset_id.split('/', 1)
            
            # Get dataset metadata
            dataset_metadata = self.kaggle_api.dataset_metadata(owner, dataset_name)
            
            # Get dataset files list
            files = self.kaggle_api.dataset_list_files(owner, dataset_name)
            
            return {
                'metadata': dataset_metadata,
                'files': [{'name': f.name, 'size': f.size} for f in files.files] if files else []
            }
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[red]❌ Error getting Kaggle dataset details: {e}[/red]")
            return {}
    
    def test_connection(self) -> bool:
        """Test if the Kaggle API is accessible"""
        
        if not self.kaggle_api:
            return False
        
        try:
            # Try to get user info as a simple API test
            self.kaggle_api.competitions_list(page=1, page_size=1)
            return True
            
        except Exception:
            return False
    
    def download_dataset(self, dataset_id: str, download_path: Path) -> bool:
        """Download a Kaggle dataset (for future use)"""
        
        if not self.kaggle_api:
            return False
        
        try:
            if '/' not in dataset_id:
                return False
            
            owner, dataset_name = dataset_id.split('/', 1)
            download_path.mkdir(parents=True, exist_ok=True)
            
            self.kaggle_api.dataset_download_files(
                owner, 
                dataset_name, 
                path=str(download_path),
                unzip=True
            )
            
            return True
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[red]❌ Error downloading dataset: {e}[/red]")
            return False