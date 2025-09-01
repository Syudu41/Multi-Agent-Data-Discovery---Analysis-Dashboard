"""
Agent 1: Data Discovery Specialist

Responsible for:
- Understanding user queries using LLM
- Searching data.gov and Kaggle APIs
- Ranking datasets by relevance, quality, recency
- Presenting top datasets for selection
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from tools.data_sources.data_gov_connector import DataGovConnector
from tools.data_sources.kaggle_connector import KaggleConnector
from tools.llm_client import OllamaClient

@dataclass
class Dataset:
    """Represents a discovered dataset"""
    title: str
    description: str
    source: str  # 'data.gov' or 'kaggle'
    url: str
    relevance_score: float
    quality_score: float
    recency_score: float
    overall_score: float
    metadata: Dict[str, Any]
    
class DataCollectorAgent:
    """Agent 1: Data Discovery Specialist"""
    
    def __init__(self, settings, verbose: bool = False):
        self.settings = settings
        self.verbose = verbose
        self.console = Console()
        
        # Initialize LLM client
        self.llm = OllamaClient(
            host=settings.ollama_host,
            model=settings.ollama_model,
            verbose=verbose
        )
        
        # Initialize data source connectors
        self.data_gov = DataGovConnector(
            api_base=settings.data_gov_api_base,
            cache_dir=settings.cache_dir,
            verbose=verbose
        )
        
        self.kaggle = KaggleConnector(
            username=settings.kaggle_username,
            api_key=settings.kaggle_key,
            cache_dir=settings.cache_dir,
            verbose=verbose
        )
    
    def discover_datasets(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Main method: Discover relevant datasets for a user query
        
        Args:
            query: Natural language query from user
            max_results: Maximum number of datasets to return
            
        Returns:
            List of ranked datasets with metadata
        """
        
        if self.verbose:
            self.console.print(f"[dim]🔍 Starting dataset discovery for: '{query}'[/dim]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Step 1: Parse query using LLM
            task = progress.add_task("🧠 Understanding query with AI...", total=None)
            search_params = self._parse_query_with_llm(query)
            progress.update(task, completed=True)
            
            if self.verbose:
                self.console.print(f"[dim]📝 Parsed query: {search_params}[/dim]")
            
            # Step 2: Search data sources in parallel
            task = progress.add_task("🌐 Searching data.gov...", total=None)
            data_gov_results = self.data_gov.search(search_params)
            progress.update(task, completed=True)
            
            task = progress.add_task("📊 Searching Kaggle...", total=None)
            kaggle_results = self.kaggle.search(search_params)
            progress.update(task, completed=True)
            
            # Step 3: Combine and rank results
            task = progress.add_task("⚖️ Ranking datasets by relevance...", total=None)
            all_datasets = self._combine_results(data_gov_results, kaggle_results)
            ranked_datasets = self._rank_datasets(all_datasets, query, search_params)
            progress.update(task, completed=True)
        
        # Return top results
        top_results = ranked_datasets[:max_results]
        
        if self.verbose:
            self.console.print(f"[dim]✅ Found {len(ranked_datasets)} total datasets, returning top {len(top_results)}[/dim]")
        
        return [self._dataset_to_dict(dataset) for dataset in top_results]
    
    def _parse_query_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to extract search parameters from natural language query"""
        
        prompt = f"""
        You are a data discovery expert. Parse this natural language query into structured search parameters.
        
        Query: "{query}"
        
        Extract the following information and respond in JSON format:
        {{
            "keywords": ["list", "of", "relevant", "search", "terms"],
            "domain": "primary subject area (education, health, economics, etc.)",
            "data_types": ["statistical", "survey", "time-series", "geographic", etc.],
            "geographic_scope": "local/national/international/global",
            "time_relevance": "current/historical/any",
            "priority_keywords": ["most", "important", "terms"]
        }}
        
        Focus on extracting concrete search terms that would help find relevant datasets.
        """
        
        try:
            response = self.llm.generate(prompt)
            # Try to parse JSON response
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_text = response[json_start:json_end]
                return json.loads(json_text)
            else:
                # Fallback: basic keyword extraction
                return self._basic_keyword_extraction(query)
                
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]⚠️ LLM parsing failed: {e}. Using fallback.[/yellow]")
            return self._basic_keyword_extraction(query)
    
    def _basic_keyword_extraction(self, query: str) -> Dict[str, Any]:
        """Fallback method for keyword extraction"""
        
        # Simple keyword extraction
        import re
        words = re.findall(r'\b\w+\b', query.lower())
        # Remove common stop words
        stop_words = {'how', 'does', 'what', 'where', 'when', 'why', 'is', 'are', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return {
            "keywords": keywords,
            "domain": "general",
            "data_types": ["any"],
            "geographic_scope": "any",
            "time_relevance": "any",
            "priority_keywords": keywords[:3]  # Top 3 words
        }
    
    def _combine_results(self, data_gov_results: List[Dict], kaggle_results: List[Dict]) -> List[Dataset]:
        """Combine results from different sources into Dataset objects"""
        
        datasets = []
        
        # Convert data.gov results
        for result in data_gov_results:
            datasets.append(Dataset(
                title=result.get('title', 'Unknown Title'),
                description=result.get('notes', result.get('description', '')),
                source='data.gov',
                url=result.get('url', ''),
                relevance_score=0.0,  # Will be calculated later
                quality_score=self._calculate_quality_score(result, 'data.gov'),
                recency_score=self._calculate_recency_score(result),
                overall_score=0.0,  # Will be calculated later
                metadata=result
            ))
        
        # Convert Kaggle results  
        for result in kaggle_results:
            datasets.append(Dataset(
                title=result.get('title', result.get('name', 'Unknown Title')),
                description=result.get('description', result.get('subtitle', '')),
                source='kaggle',
                url=result.get('url', ''),
                relevance_score=0.0,  # Will be calculated later
                quality_score=self._calculate_quality_score(result, 'kaggle'),
                recency_score=self._calculate_recency_score(result),
                overall_score=0.0,  # Will be calculated later
                metadata=result
            ))
        
        return datasets
    
    def _rank_datasets(self, datasets: List[Dataset], original_query: str, search_params: Dict) -> List[Dataset]:
        """Rank datasets by relevance, quality, and recency"""
        
        for dataset in datasets:
            # Calculate relevance score
            dataset.relevance_score = self._calculate_relevance_score(
                dataset, original_query, search_params
            )
            
            # Calculate overall score (weighted average)
            dataset.overall_score = (
                dataset.relevance_score * 0.5 +  # 50% relevance
                dataset.quality_score * 0.3 +     # 30% quality  
                dataset.recency_score * 0.2        # 20% recency
            )
        
        # Sort by overall score (descending)
        return sorted(datasets, key=lambda d: d.overall_score, reverse=True)
    
    def _calculate_relevance_score(self, dataset: Dataset, query: str, search_params: Dict) -> float:
        """Calculate relevance score based on text matching"""
        
        # Combine title and description for scoring
        text = f"{dataset.title} {dataset.description}".lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Direct query match
        if query_lower in text:
            score += 3.0
        
        # Individual keyword matches
        for keyword in search_params.get('keywords', []):
            if keyword.lower() in text:
                score += 1.0
        
        # Priority keyword matches (weighted higher)
        for keyword in search_params.get('priority_keywords', []):
            if keyword.lower() in text:
                score += 2.0
        
        # Domain match
        domain = search_params.get('domain', '').lower()
        if domain and domain in text:
            score += 1.5
        
        # Normalize to 0-10 scale
        return min(score, 10.0)
    
    def _calculate_quality_score(self, result: Dict, source: str) -> float:
        """Calculate quality score based on metadata"""
        
        score = 5.0  # Base score
        
        if source == 'data.gov':
            # Government data generally high quality
            score += 2.0
            
            # Check for additional quality indicators
            if result.get('resources'):
                score += 1.0  # Has downloadable resources
            if result.get('organization'):
                score += 1.0  # Has organization info
                
        elif source == 'kaggle':
            # Kaggle scoring based on votes/downloads
            if 'voteCount' in result:
                votes = result.get('voteCount', 0)
                score += min(votes / 10, 3.0)  # Up to 3 points for popularity
                
            if 'downloadCount' in result:
                downloads = result.get('downloadCount', 0)  
                score += min(downloads / 1000, 2.0)  # Up to 2 points for downloads
        
        return min(score, 10.0)
    
    def _calculate_recency_score(self, result: Dict) -> float:
        """Calculate recency score based on last update"""
        
        # Default score for unknown dates
        score = 5.0
        
        try:
            from dateutil import parser
            import datetime
            
            # Try to find a date field
            date_fields = ['metadata_modified', 'lastUpdated', 'creationDate', 'updated']
            date_str = None
            
            for field in date_fields:
                if field in result:
                    date_str = result[field]
                    break
            
            if date_str:
                date_obj = parser.parse(date_str)
                now = datetime.datetime.now(date_obj.tzinfo or datetime.timezone.utc)
                days_ago = (now - date_obj).days
                
                # Score based on recency (higher for more recent)
                if days_ago < 30:
                    score = 10.0
                elif days_ago < 180:
                    score = 8.0  
                elif days_ago < 365:
                    score = 6.0
                elif days_ago < 1095:  # 3 years
                    score = 4.0
                else:
                    score = 2.0
                    
        except Exception:
            # If date parsing fails, keep default score
            pass
            
        return score
    
    def _dataset_to_dict(self, dataset: Dataset) -> Dict[str, Any]:
        """Convert Dataset object to dictionary for output"""
        
        return {
            'title': dataset.title,
            'description': dataset.description,
            'source': dataset.source,
            'url': dataset.url,
            'relevance_score': round(dataset.relevance_score, 2),
            'quality_score': round(dataset.quality_score, 2),
            'recency_score': round(dataset.recency_score, 2),
            'overall_score': round(dataset.overall_score, 2),
            'metadata': dataset.metadata
        }