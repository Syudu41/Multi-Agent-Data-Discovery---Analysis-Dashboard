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
        """Internal method to search datasets using multiple endpoints"""
        
        all_results = []
        
        # Build search query from parameters
        query_parts = []
        
        # Add keywords with priority handling
        keywords = search_params.get('keywords', [])
        priority_keywords = search_params.get('priority_keywords', [])
        preserved_phrases = search_params.get('preserved_phrases', [])
        
        # Combine all query elements
        all_query_terms = preserved_phrases + priority_keywords + keywords[:10]  # Limit for performance
        unique_terms = list(dict.fromkeys(all_query_terms))  # Remove duplicates, preserve order
        
        if unique_terms:
            # For data.gov, use top terms for focused search
            query = ' '.join(unique_terms[:5])  # Use top 5 terms
        else:
            query = 'data'
        
        if self.verbose:
            self.console.print(f"[dim]🔍 Data.gov multi-endpoint search with: '{query}'[/dim]")
        
        # Endpoint 1: Package search (main datasets)
        package_results = self._search_packages(query, search_params)
        all_results.extend(package_results)
        
        # Endpoint 2: Organization-based search (for domain-specific queries)
        if search_params.get('domain') != 'general':
            org_results = self._search_by_organization(search_params)
            all_results.extend(org_results)
        
        # Endpoint 3: Group/topic search (for categorized content)
        group_results = self._search_by_groups(search_params)
        all_results.extend(group_results)
        
        # Deduplicate results by ID
        seen_ids = set()
        deduplicated_results = []
        for result in all_results:
            result_id = result.get('id', '')
            if result_id and result_id not in seen_ids:
                seen_ids.add(result_id)
                deduplicated_results.append(result)
        
        if self.verbose:
            self.console.print(f"[green]✅ Data.gov multi-endpoint: {len(all_results)} raw, {len(deduplicated_results)} deduplicated[/green]")
        
        return deduplicated_results
    
    def _search_packages(self, query: str, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search main package endpoint"""
        
        params = {
            'q': query,
            'rows': 150,  # Increased from 100
            'sort': 'score desc, metadata_modified desc',
            'facet': 'false',
        }
        
        # Add geographic filters if specified
        geo_scope = search_params.get('geographic_scope', '')
        if geo_scope == 'national':
            params['fq'] = 'organization_type:federal'
        elif geo_scope == 'local':
            params['fq'] = 'organization_type:(state OR local)'
        
        try:
            url = f"{self.api_base}/action/package_search"
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            if data.get('success'):
                datasets = data.get('result', {}).get('results', [])
                return [self._clean_dataset(dataset) for dataset in datasets if self._clean_dataset(dataset)]
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]⚠️ Package search error: {e}[/yellow]")
        
        return []
    
    def _search_by_organization(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search by relevant organizations for domain-specific queries"""
        
        # Map domains to relevant organizations
        domain_orgs = {
            'education': ['department-of-education', 'national-center-for-education-statistics'],
            'health': ['centers-for-disease-control-and-prevention', 'national-institutes-of-health'],
            'economy': ['bureau-of-economic-analysis', 'bureau-of-labor-statistics', 'census-bureau'],
            'climate': ['national-oceanic-and-atmospheric-administration', 'environmental-protection-agency'],
            'technology': ['national-institute-of-standards-and-technology', 'national-science-foundation']
        }
        
        domain = search_params.get('domain', '')
        relevant_orgs = domain_orgs.get(domain, [])
        
        if not relevant_orgs:
            return []
        
        org_results = []
        for org_name in relevant_orgs[:2]:  # Limit to 2 orgs for performance
            try:
                params = {
                    'q': '*:*',  # All datasets from this org
                    'fq': f'organization:{org_name}',
                    'rows': 25,  # Fewer per org
                    'sort': 'metadata_modified desc'
                }
                
                url = f"{self.api_base}/action/package_search"
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if data.get('success'):
                    datasets = data.get('result', {}).get('results', [])
                    for dataset in datasets:
                        cleaned = self._clean_dataset(dataset)
                        if cleaned:
                            org_results.append(cleaned)
                
                if self.verbose:
                    self.console.print(f"[dim]📊 {org_name}: {len(datasets)} datasets[/dim]")
                            
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]⚠️ Org search error for {org_name}: {e}[/yellow]")
                continue
        
        return org_results
    
    def _search_by_groups(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search by topic groups for better categorization"""
        
        # Map domains to data.gov groups/topics
        domain_groups = {
            'education': ['education'],
            'health': ['health', 'safety'],
            'economy': ['economy', 'finance', 'business'],
            'climate': ['environment', 'energy'],
            'technology': ['science-and-research', 'information-and-communications']
        }
        
        domain = search_params.get('domain', '')
        relevant_groups = domain_groups.get(domain, [])
        
        if not relevant_groups:
            return []
        
        group_results = []
        for group_name in relevant_groups[:2]:  # Limit for performance
            try:
                params = {
                    'q': '*:*',
                    'fq': f'groups:{group_name}',
                    'rows': 20,
                    'sort': 'metadata_modified desc'
                }
                
                url = f"{self.api_base}/action/package_search"
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if data.get('success'):
                    datasets = data.get('result', {}).get('results', [])
                    for dataset in datasets:
                        cleaned = self._clean_dataset(dataset)
                        if cleaned:
                            group_results.append(cleaned)
                
                if self.verbose:
                    self.console.print(f"[dim]🏷️ Group {group_name}: {len(datasets)} datasets[/dim]")
                            
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]⚠️ Group search error for {group_name}: {e}[/yellow]")
                continue
        
        return group_results
    
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