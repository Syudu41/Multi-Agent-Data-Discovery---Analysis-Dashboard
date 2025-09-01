"""
World Bank API connector

Searches World Bank Open Data for economic and development datasets
"""

import requests
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from rich.console import Console

from .base_connector import BaseConnector

class WorldBankConnector(BaseConnector):
    """Connector for World Bank Open Data API"""
    
    def __init__(self, api_base: str = "https://api.worldbank.org/v2", 
                 cache_dir: Path = Path("./cache"), verbose: bool = False):
        super().__init__(cache_dir, verbose=verbose)
        
        self.api_base = api_base.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PersonalDataAI/1.0',
            'Accept': 'application/json'
        })
        
        self.console = Console() if verbose else None
        
        if self.verbose:
            self.console.print(f"[green]🌍 World Bank connector initialized - API: {self.api_base}[/green]")
    
    def search(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search World Bank for indicators/datasets"""
        
        if self.verbose:
            self.console.print(f"[dim]🔍 Searching World Bank with: {search_params}[/dim]")
        
        def _make_request():
            return self._search_indicators(search_params)
        
        return self._make_request_with_cache("search", search_params, _make_request)
    
    def _search_indicators(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Internal method to search World Bank indicators"""
        
        # Build search query from parameters
        keywords = search_params.get('keywords', [])
        priority_keywords = search_params.get('priority_keywords', [])
        
        # Combine keywords for search
        all_keywords = priority_keywords + keywords
        search_terms = list(dict.fromkeys(all_keywords))  # Remove duplicates, preserve order
        
        if not search_terms:
            search_terms = ['economy']  # Default fallback
        
        # Use primary keyword for search
        search_query = search_terms[0]
        
        try:
            # World Bank indicators API
            url = f"{self.api_base}/indicators"
            params = {
                'format': 'json',
                'per_page': 200,  # Get more results
                'page': 1
            }
            
            if self.verbose:
                self.console.print(f"[dim]📡 GET {url} (searching for indicators)[/dim]")
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if len(data) < 2:
                if self.verbose:
                    self.console.print("[yellow]⚠️ No data returned from World Bank API[/yellow]")
                return []
            
            indicators = data[1]  # World Bank API returns [metadata, data]
            
            if self.verbose:
                self.console.print(f"[green]✅ Found {len(indicators)} World Bank indicators[/green]")
            
            # Filter indicators based on search terms
            relevant_indicators = []
            for indicator in indicators:
                if self._is_relevant(indicator, search_terms):
                    relevant_indicators.append(indicator)
            
            if self.verbose:
                self.console.print(f"[green]✅ Filtered to {len(relevant_indicators)} relevant indicators[/green]")
            
            # Clean and enhance results
            cleaned_indicators = []
            for indicator in relevant_indicators:
                cleaned = self._clean_indicator(indicator)
                if cleaned:
                    cleaned_indicators.append(cleaned)
            
            return cleaned_indicators
            
        except requests.exceptions.RequestException as e:
            if self.verbose:
                self.console.print(f"[red]❌ World Bank API error: {e}[/red]")
            return []
        
        except Exception as e:
            if self.verbose:
                self.console.print(f"[red]❌ World Bank processing error: {e}[/red]")
            return []
    
    def _is_relevant(self, indicator: Dict[str, Any], search_terms: List[str]) -> bool:
        """Check if indicator is relevant to search terms"""
        
        # Combine name and description for searching
        name = (indicator.get('name', '') or '').lower()
        desc = (indicator.get('sourceNote', '') or '').lower()
        combined_text = f"{name} {desc}"
        
        # Check if any search term appears in the indicator
        for term in search_terms:
            if term.lower() in combined_text:
                return True
        
        # Also check for common related terms
        term_mappings = {
            'education': ['education', 'school', 'literacy', 'enrollment'],
            'health': ['health', 'mortality', 'life expectancy', 'disease'],
            'economy': ['gdp', 'economic', 'growth', 'income', 'poverty'],
            'population': ['population', 'demographic', 'birth', 'death'],
            'environment': ['environment', 'emissions', 'energy', 'forest'],
            'climate': ['climate', 'temperature', 'carbon', 'emissions']
        }
        
        for term in search_terms:
            if term.lower() in term_mappings:
                for related_term in term_mappings[term.lower()]:
                    if related_term in combined_text:
                        return True
        
        return False
    
    def _clean_indicator(self, raw_indicator: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean and standardize World Bank indicator information"""
        
        try:
            indicator_id = raw_indicator.get('id', '')
            name = raw_indicator.get('name', 'Unnamed Indicator')
            description = raw_indicator.get('sourceNote', '')
            
            # Get source information
            source_info = raw_indicator.get('source', {})
            source_name = source_info.get('value', 'World Bank') if source_info else 'World Bank'
            
            # Get topic information
            topics = raw_indicator.get('topics', [])
            topic_names = []
            if topics:
                for topic in topics:
                    if isinstance(topic, dict) and 'value' in topic:
                        topic_names.append(topic['value'])
            
            # Build URL to data
            data_url = f"https://data.worldbank.org/indicator/{indicator_id}"
            
            cleaned = {
                'id': indicator_id,
                'title': name,
                'description': description,
                'notes': description,  # Keep original field name
                'url': data_url,
                'source_organization': source_name,
                'topics': topic_names,
                'indicator_type': 'World Bank Indicator',
                'last_updated': raw_indicator.get('lastupdated', ''),
                'source_api': 'worldbank',
                'raw_data': raw_indicator  # Keep original for debugging
            }
            
            return cleaned
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]⚠️ Error cleaning World Bank indicator: {e}[/yellow]")
            return None
    
    def get_dataset_details(self, indicator_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific indicator"""
        
        def _make_request():
            return self._get_indicator_details(indicator_id)
        
        return self._make_request_with_cache(
            "details", 
            {"indicator_id": indicator_id}, 
            _make_request
        )
    
    def _get_indicator_details(self, indicator_id: str) -> Dict[str, Any]:
        """Internal method to get indicator details"""
        
        try:
            url = f"{self.api_base}/indicators/{indicator_id}"
            params = {'format': 'json'}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if len(data) < 2:
                return {}
            
            indicator = data[1][0] if data[1] else {}
            return self._clean_indicator(indicator) or {}
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[red]❌ Error getting indicator details: {e}[/red]")
            return {}
    
    def test_connection(self) -> bool:
        """Test if the World Bank API is accessible"""
        
        try:
            url = f"{self.api_base}/countries"
            params = {'format': 'json', 'per_page': 1}
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            return len(data) >= 2  # Should return metadata + data
            
        except Exception:
            return False
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get information about the API"""
        
        try:
            url = f"{self.api_base}/sources"
            params = {'format': 'json', 'per_page': 10}
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            return {
                'sources_available': len(data[1]) if len(data) > 1 else 0,
                'api_base': self.api_base
            }
            
        except Exception as e:
            return {"error": str(e)}