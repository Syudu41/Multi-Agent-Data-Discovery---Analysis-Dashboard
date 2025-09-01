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
        """Internal method to search World Bank using multiple endpoints"""
        
        all_results = []
        
        # Build search strategy from enhanced parameters
        keywords = search_params.get('keywords', [])
        priority_keywords = search_params.get('priority_keywords', [])
        preserved_phrases = search_params.get('preserved_phrases', [])
        domain = search_params.get('domain', '')
        
        # Combine search terms intelligently
        primary_terms = preserved_phrases + priority_keywords[:3]  # Top phrases + keywords
        if not primary_terms:
            primary_terms = keywords[:2] if keywords else ['economy']
        
        if self.verbose:
            self.console.print(f"[dim]🌍 World Bank multi-endpoint search with: {primary_terms}[/dim]")
        
        # Search Strategy 1: Indicators API (main economic indicators)
        indicator_results = self._search_indicators_endpoint(primary_terms, search_params)
        all_results.extend(indicator_results)
        
        # Search Strategy 2: Projects API (development projects with data)
        if domain in ['economy', 'health', 'education', 'climate']:
            project_results = self._search_projects_endpoint(primary_terms, search_params)
            all_results.extend(project_results)
        
        # Search Strategy 3: Country-specific data (if geographic scope indicates)
        if search_params.get('geographic_scope') in ['national', 'international']:
            country_results = self._search_country_data(primary_terms, search_params)
            all_results.extend(country_results)
        
        return all_results
    
    def _search_indicators_endpoint(self, search_terms: List[str], search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search World Bank Indicators API with enhanced strategy"""
        
        try:
            # Enhanced indicator search with more results
            url = f"{self.api_base}/indicators"
            params = {
                'format': 'json',
                'per_page': 300,  # Increased from 200
                'page': 1
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            if len(data) < 2:
                return []
            
            indicators = data[1]
            
            # Enhanced relevance filtering
            relevant_indicators = []
            for indicator in indicators:
                if self._is_enhanced_relevant(indicator, search_terms, search_params):
                    relevant_indicators.append(indicator)
            
            # Clean and return
            cleaned_indicators = []
            for indicator in relevant_indicators:
                cleaned = self._clean_indicator(indicator)
                if cleaned:
                    cleaned_indicators.append(cleaned)
            
            if self.verbose:
                self.console.print(f"[dim]📊 Indicators endpoint: {len(cleaned_indicators)} relevant[/dim]")
            
            return cleaned_indicators
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]⚠️ Indicators search error: {e}[/yellow]")
            return []
    
    def _search_projects_endpoint(self, search_terms: List[str], search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search World Bank Projects API for development projects"""
        
        try:
            # World Bank Projects API
            url = f"{self.api_base}/projects"
            params = {
                'format': 'json',
                'per_page': 100,
                'page': 1
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            if len(data) < 2:
                return []
            
            projects = data[1]
            
            # Filter projects by relevance to search terms
            relevant_projects = []
            for project in projects:
                if self._is_project_relevant(project, search_terms, search_params):
                    relevant_projects.append(project)
            
            # Convert projects to our dataset format
            project_datasets = []
            for project in relevant_projects:
                cleaned = self._clean_project(project)
                if cleaned:
                    project_datasets.append(cleaned)
            
            if self.verbose:
                self.console.print(f"[dim]🏗️ Projects endpoint: {len(project_datasets)} relevant[/dim]")
            
            return project_datasets
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]⚠️ Projects search error: {e}[/yellow]")
            return []
    
    def _search_country_data(self, search_terms: List[str], search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search country-specific World Bank data"""
        
        try:
            # Get country data overview
            url = f"{self.api_base}/countries"
            params = {
                'format': 'json',
                'per_page': 50  # Focus on major countries
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if len(data) < 2:
                return []
            
            countries = data[1]
            
            # Create country data entries for relevant countries
            country_datasets = []
            for country in countries[:20]:  # Top 20 countries
                if country.get('capitalCity'):  # Filter for actual countries
                    country_dataset = self._create_country_dataset(country, search_terms, search_params)
                    if country_dataset:
                        country_datasets.append(country_dataset)
            
            if self.verbose:
                self.console.print(f"[dim]🏁 Country endpoint: {len(country_datasets)} datasets[/dim]")
            
            return country_datasets
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]⚠️ Country search error: {e}[/yellow]")
            return []
    
    def _is_enhanced_relevant(self, indicator: Dict[str, Any], search_terms: List[str], search_params: Dict[str, Any]) -> bool:
        """Enhanced relevance checking with domain intelligence"""
        
        name = (indicator.get('name', '') or '').lower()
        desc = (indicator.get('sourceNote', '') or '').lower()
        combined_text = f"{name} {desc}"
        
        # Check direct term matches (higher weight for phrases)
        for term in search_terms:
            if term.lower() in combined_text:
                return True
        
        # Domain-specific enhanced matching
        domain = search_params.get('domain', '')
        domain_mappings = {
            'economy': ['gdp', 'economic', 'growth', 'income', 'poverty', 'unemployment', 'inflation', 'trade', 'fiscal'],
            'education': ['education', 'school', 'literacy', 'enrollment', 'completion', 'expenditure on education'],
            'health': ['health', 'mortality', 'life expectancy', 'disease', 'nutrition', 'immunization', 'maternal'],
            'population': ['population', 'demographic', 'birth', 'death', 'migration', 'urban', 'rural'],
            'climate': ['environment', 'emissions', 'energy', 'forest', 'renewable', 'carbon', 'co2'],
            'technology': ['technology', 'innovation', 'research', 'development', 'internet', 'mobile']
        }
        
        if domain in domain_mappings:
            for domain_term in domain_mappings[domain]:
                if domain_term in combined_text:
                    return True
        
        return False
    
    def _is_project_relevant(self, project: Dict[str, Any], search_terms: List[str], search_params: Dict[str, Any]) -> bool:
        """Check if World Bank project is relevant to search"""
        
        project_name = (project.get('project_name', '') or '').lower()
        
        # Check for direct matches
        for term in search_terms:
            if term.lower() in project_name:
                return True
        
        # Check project status - prefer active/closed projects with data
        status = project.get('status', '').lower()
        if status in ['active', 'closed']:
            # Check if project has sector alignment
            sectors = project.get('sector', [])
            domain = search_params.get('domain', '')
            
            sector_mappings = {
                'health': ['health', 'social protection'],
                'education': ['education'],
                'economy': ['finance', 'industry', 'public administration'],
                'climate': ['environment', 'energy', 'transport']
            }
            
            if domain in sector_mappings:
                for sector in sectors:
                    sector_name = sector.get('Name', '').lower()
                    for domain_sector in sector_mappings[domain]:
                        if domain_sector in sector_name:
                            return True
        
        return False
    
    def _clean_project(self, raw_project: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert World Bank project to dataset format"""
        
        try:
            project_id = raw_project.get('id', '')
            project_name = raw_project.get('project_name', 'Unnamed Project')
            
            # Create description from available info
            description_parts = []
            if raw_project.get('abstract'):
                description_parts.append(raw_project['abstract'])
            if raw_project.get('country') and raw_project['country'].get('value'):
                description_parts.append(f"Country: {raw_project['country']['value']}")
            
            description = ' | '.join(description_parts) or 'World Bank development project'
            
            # Build project URL
            project_url = f"https://projects.worldbank.org/en/projects-operations/project-detail/{project_id}"
            
            cleaned = {
                'id': project_id,
                'title': f"WB Project: {project_name}",
                'description': description,
                'notes': description,
                'url': project_url,
                'source_organization': 'World Bank Projects',
                'topics': [raw_project.get('sector', {}).get('Name', 'Development')],
                'indicator_type': 'World Bank Project',
                'last_updated': raw_project.get('boardapprovaldate', ''),
                'source_api': 'worldbank',
                'raw_data': raw_project
            }
            
            return cleaned
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]⚠️ Error cleaning project: {e}[/yellow]")
            return None
    
    def _create_country_dataset(self, country: Dict[str, Any], search_terms: List[str], search_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create country data entry"""
        
        try:
            country_code = country.get('id', '')
            country_name = country.get('name', '')
            
            if not country_code or not country_name:
                return None
            
            # Create country data description
            description = f"Comprehensive World Bank data for {country_name} including economic, social, and development indicators"
            
            # Country data URL
            country_url = f"https://data.worldbank.org/country/{country_code.lower()}"
            
            cleaned = {
                'id': f"country_{country_code}",
                'title': f"World Bank Data: {country_name}",
                'description': description,
                'notes': description,
                'url': country_url,
                'source_organization': 'World Bank Country Data',
                'topics': ['Country Data', 'Development Indicators'],
                'indicator_type': 'World Bank Country Profile',
                'last_updated': '2024',
                'source_api': 'worldbank',
                'raw_data': country
            }
            
            return cleaned
            
        except Exception as e:
            return None
    
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