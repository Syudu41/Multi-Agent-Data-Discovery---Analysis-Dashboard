"""
Data.gov connector using CKAN API

Searches the US government's open data catalog
"""

import requests
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from rich.console import Console

from .base_connector import BaseConnector

class DataGovConnector(BaseConnector):
    """Connector for data.gov using CKAN API"""
    
    def __init__(self, api_base: str = "https://catalog.data.gov/api/3", 
                 cache_dir: Path = Path("./cache"), verbose: bool = False):
        super().__init__(cache_dir, verbose=verbose)
        
        self.api_base = api_base.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PersonalDataAI/1.0'
        })
        
        self.console = Console() if verbose else None
        
        if self.verbose:
            self.console.print(f"[green]🏛️ Data.gov connector initialized - API: {self.api_base}[/green]")
    
    def search(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search data.gov for datasets"""
        
        if self.verbose:
            self.console.print(f"[dim]🔍 Searching data.gov with: {search_params}[/dim]")
        
        def _make_request():
            return self._search_datasets(search_params)
        
        return self._make_request_with_cache("search", search_params, _make_request)
    
    def _search_datasets(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Internal method to search datasets"""
        
        # Build search query from parameters
        query_parts = []
        
        # Add keywords
        keywords = search_params.get('keywords', [])
        priority_keywords = search_params.get('priority_keywords', [])
        
        # Combine and prioritize keywords
        all_keywords = list(set(priority_keywords + keywords))
        if all_keywords:
            # Use OR for broader results
            query_parts.append(' OR '.join(all_keywords))
        
        # Add domain-specific terms
        domain = search_params.get('domain', '')
        if domain and domain != 'general':
            query_parts.append(domain)
        
        # Combine query parts
        query = ' '.join(query_parts) if query_parts else 'data'
        
        # CKAN API parameters
        params = {
            'q': query,
            'rows': 100,  # Increased to 100 results per request
            'sort': 'score desc, metadata_modified desc',  # Relevance + recency
            'facet': 'false',  # Don't need facet counts
        }
        
        try:
            url = f"{self.api_base}/action/package_search"
            
            if self.verbose:
                self.console.print(f"[dim]📡 GET {url}?q={query[:50]}...[/dim]")
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('success'):
                raise Exception(f"API returned success=false: {data.get('error', 'Unknown error')}")
            
            datasets = data.get('result', {}).get('results', [])
            
            if self.verbose:
                self.console.print(f"[green]✅ Found {len(datasets)} datasets on data.gov[/green]")
            
            # Clean and enhance results
            cleaned_datasets = []
            for dataset in datasets:
                cleaned = self._clean_dataset(dataset)
                if cleaned:
                    cleaned_datasets.append(cleaned)
            
            return cleaned_datasets
            
        except requests.exceptions.RequestException as e:
            if self.verbose:
                self.console.print(f"[red]❌ Data.gov API error: {e}[/red]")
            return []
        
        except Exception as e:
            if self.verbose:
                self.console.print(f"[red]❌ Data.gov processing error: {e}[/red]")
            return []
    
    def _clean_dataset(self, raw_dataset: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean and standardize dataset information"""
        
        try:
            # Extract basic information
            dataset_id = raw_dataset.get('id', '')
            title = raw_dataset.get('title', 'Untitled Dataset')
            notes = raw_dataset.get('notes', '')
            
            # Get organization info
            org = raw_dataset.get('organization', {})
            org_name = org.get('title', org.get('name', '')) if org else ''
            
            # Build dataset URL
            dataset_name = raw_dataset.get('name', dataset_id)
            dataset_url = f"https://catalog.data.gov/dataset/{dataset_name}"
            
            # Extract resources (actual data files)
            resources = raw_dataset.get('resources', [])
            download_urls = []
            file_formats = set()
            
            for resource in resources:
                if resource.get('url'):
                    download_urls.append(resource['url'])
                if resource.get('format'):
                    file_formats.add(resource['format'].upper())
            
            # Get dates
            created = raw_dataset.get('metadata_created', '')
            modified = raw_dataset.get('metadata_modified', '')
            
            # Get tags
            tags = [tag.get('display_name', tag.get('name', '')) 
                   for tag in raw_dataset.get('tags', [])]
            
            cleaned = {
                'id': dataset_id,
                'title': title,
                'description': notes,
                'notes': notes,  # Keep original field name
                'url': dataset_url,
                'organization': org_name,
                'resources': resources,
                'download_urls': download_urls,
                'formats': list(file_formats),
                'tags': tags,
                'metadata_created': created,
                'metadata_modified': modified,
                'source_api': 'data.gov',
                'raw_data': raw_dataset  # Keep original for debugging
            }
            
            return cleaned
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]⚠️ Error cleaning dataset: {e}[/yellow]")
            return None
    
    def get_dataset_details(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific dataset"""
        
        def _make_request():
            return self._get_dataset_details(dataset_id)
        
        return self._make_request_with_cache(
            "details", 
            {"dataset_id": dataset_id}, 
            _make_request
        )
    
    def _get_dataset_details(self, dataset_id: str) -> Dict[str, Any]:
        """Internal method to get dataset details"""
        
        try:
            url = f"{self.api_base}/action/package_show"
            params = {'id': dataset_id}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('success'):
                raise Exception(f"API returned success=false: {data.get('error', 'Unknown error')}")
            
            dataset = data.get('result', {})
            return self._clean_dataset(dataset) or {}
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[red]❌ Error getting dataset details: {e}[/red]")
            return {}
    
    def test_connection(self) -> bool:
        """Test if the API is accessible"""
        
        try:
            url = f"{self.api_base}/action/status_show"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            return data.get('success', False)
            
        except Exception:
            return False
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get information about the API"""
        
        try:
            url = f"{self.api_base}/action/status_show"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            
            return response.json().get('result', {})
            
        except Exception as e:
            return {"error": str(e)}