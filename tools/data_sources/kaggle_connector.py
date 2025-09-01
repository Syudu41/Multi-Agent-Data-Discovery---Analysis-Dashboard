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
            
            # Search parameters - Fixed sort_by value
            search_kwargs = {
                'search': search_query,
                'sort_by': 'hottest'  # Fixed: was 'hotness', should be 'hottest'
            }
            
            try:
                if self.verbose:
                    self.console.print(f"[dim]📡 Kaggle API search: '{search_query}' with {len(all_keywords)} keywords[/dim]")
                
                # Try search with sort parameter first
                try:
                    datasets = self.kaggle_api.dataset_list(search=search_query, sort_by='hottest')
                    if self.verbose:
                        self.console.print(f"[green]✅ Kaggle API (with sort) returned {len(datasets)} datasets[/green]")
                except Exception as sort_error:
                    if self.verbose:
                        self.console.print(f"[yellow]⚠️ Sort failed, trying without: {sort_error}[/yellow]")
                    # Try without sort parameter
                    datasets = self.kaggle_api.dataset_list(search=search_query)
                    if self.verbose:
                        self.console.print(f"[green]✅ Kaggle API (no sort) returned {len(datasets)} datasets[/green]")
                    
                # If we got no results, try simpler searches
                if len(datasets) == 0 and all_keywords:
                    if self.verbose:
                        self.console.print("[yellow]⚠️ No datasets returned from Kaggle - trying broader search[/yellow]")
                        
                    # Try with just the first keyword
                    broader_search = all_keywords[0]
                    if self.verbose:
                        self.console.print(f"[dim]🔄 Retrying with broader search: '{broader_search}'[/dim]")
                    
                    try:
                        datasets = self.kaggle_api.dataset_list(search=broader_search, sort_by='hottest')
                    except:
                        datasets = self.kaggle_api.dataset_list(search=broader_search)
                    
                    if self.verbose:
                        self.console.print(f"[green]✅ Broader search returned {len(datasets)} datasets[/green]")
                        
                    # If still no results, try without search term (get popular datasets)
                    if len(datasets) == 0:
                        if self.verbose:
                            self.console.print("[yellow]⚠️ Still no results - getting popular datasets[/yellow]")
                        try:
                            datasets = self.kaggle_api.dataset_list(sort_by='hottest')[:50]  # Get top 50 popular
                        except:
                            datasets = self.kaggle_api.dataset_list()[:50]  # Fallback without sort
                        
                        if self.verbose:
                            self.console.print(f"[green]✅ Popular datasets fallback returned {len(datasets)} datasets[/green]")
                            
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[red]❌ Kaggle API error: {e}[/red]")
                
                # Try a simple fallback - get some popular datasets
                try:
                    if self.verbose:
                        self.console.print("[dim]🔄 Trying simple dataset list as fallback[/dim]")
                    try:
                        datasets = self.kaggle_api.dataset_list(sort_by='hottest')[:20]
                    except:
                        datasets = self.kaggle_api.dataset_list()[:20]  # Ultimate fallback
                        
                    if self.verbose:
                        self.console.print(f"[green]✅ Fallback returned {len(datasets)} datasets[/green]")
                except Exception as fallback_error:
                    if self.verbose:
                        self.console.print(f"[red]❌ All Kaggle attempts failed: {fallback_error}[/red]")
                    return []
            
            if self.verbose:
                self.console.print(f"[green]✅ Found {len(datasets)} datasets on Kaggle[/green]")
            
            # Convert to our format (limit to first 100 for performance)
            cleaned_datasets = []
            max_datasets = min(len(datasets), 100)
            
            for i, dataset in enumerate(datasets[:max_datasets]):
                if self.verbose and i < 3:  # Debug first 3 datasets
                    self.console.print(f"[dim]🔍 Processing dataset {i+1}: {getattr(dataset, 'title', 'No title')}[/dim]")
                
                cleaned = self._clean_dataset(dataset)
                if cleaned:
                    cleaned_datasets.append(cleaned)
                elif self.verbose:
                    self.console.print(f"[yellow]⚠️ Failed to clean dataset {i+1}[/yellow]")
            
            if self.verbose:
                self.console.print(f"[green]✅ Successfully processed {len(cleaned_datasets)} of {max_datasets} Kaggle datasets[/green]")
            
            return cleaned_datasets
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[red]❌ Kaggle API error: {e}[/red]")
            return []
    
    def _clean_dataset(self, raw_dataset) -> Optional[Dict[str, Any]]:
        """Clean and standardize Kaggle dataset information"""
        
        try:
            # Kaggle dataset objects have specific attributes
            dataset_ref = getattr(raw_dataset, 'ref', 'unknown/unknown')
            title = getattr(raw_dataset, 'title', None) or 'Untitled Dataset'
            subtitle = getattr(raw_dataset, 'subtitle', '')
            
            if self.verbose:
                self.console.print(f"[dim]   📝 Dataset ref: {dataset_ref}, title: {title[:50]}...[/dim]")
            
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
            
            if self.verbose:
                self.console.print(f"[dim]   ✅ Successfully cleaned dataset: {title[:30]}...[/dim]")
            
            return cleaned
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]⚠️ Error cleaning Kaggle dataset: {e}[/yellow]")
                self.console.print(f"[dim]   Dataset attributes: {dir(raw_dataset)}[/dim]")
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
            # Try to get a small list of datasets as a simple API test
            datasets = self.kaggle_api.dataset_list()
            return len(datasets) >= 0  # Even 0 results means the API is working
            
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