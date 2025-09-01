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
from rich.table import Table
import pandas as pd
from pathlib import Path

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
        
        # Storage for results
        self.last_results_df = None
        self.last_query = None
    
    def discover_datasets(self, query: str, max_results: int = 10) -> pd.DataFrame:
        """
        Main method: Discover relevant datasets for a user query
        
        Args:
            query: Natural language query from user
            max_results: Maximum number of datasets to return for display
            
        Returns:
            DataFrame of ranked datasets with metadata
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
        if self.verbose:
            self.console.print(f"[dim]✅ Found {len(ranked_datasets)} total datasets[/dim]")
        
        # Convert to DataFrame
        df_data = []
        for i, dataset in enumerate(ranked_datasets):
            df_data.append({
                'rank': i + 1,
                'title': dataset.title,
                'description': dataset.description[:200] + "..." if len(dataset.description) > 200 else dataset.description,
                'source': dataset.source,
                'url': dataset.url,
                'relevance_score': round(dataset.relevance_score, 2),
                'quality_score': round(dataset.quality_score, 2),
                'recency_score': round(dataset.recency_score, 2),
                'overall_score': round(dataset.overall_score, 2),
                'full_metadata': dataset.metadata
            })
        
        df = pd.DataFrame(df_data)
        
        # Save full results to files
        self._save_results(df, query)
        
        # Store in instance for later access
        self.last_results_df = df
        self.last_query = query
        
        return df.head(max_results)  # Return top N for display
    
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
    
    def _save_results(self, df: pd.DataFrame, query: str):
        """Save results to files for later access"""
        
        try:
            # Create results directory
            results_dir = self.settings.cache_dir / "results"
            results_dir.mkdir(exist_ok=True)
            
            # Generate filename from query
            import re
            safe_query = re.sub(r'[^\w\s-]', '', query).strip()
            safe_query = re.sub(r'[-\s]+', '_', safe_query)[:50]
            timestamp = int(time.time())
            
            filename_base = f"{safe_query}_{timestamp}"
            
            # Save as CSV
            csv_path = results_dir / f"{filename_base}.csv"
            df_export = df.drop('full_metadata', axis=1)  # Remove complex metadata for CSV
            df_export.to_csv(csv_path, index=False)
            
            # Save as JSON (with full metadata)
            json_path = results_dir / f"{filename_base}.json"
            df.to_json(json_path, orient='records', indent=2)
            
            if self.verbose:
                self.console.print(f"[dim]💾 Results saved: {csv_path.name}[/dim]")
                
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]⚠️ Could not save results: {e}[/yellow]")
    
    def get_more_results(self, start_rank: int = 11, count: int = 10) -> pd.DataFrame:
        """Get more results from the last search"""
        
        if self.last_results_df is None:
            raise ValueError("No previous search results available")
        
        end_rank = start_rank + count - 1
        return self.last_results_df.iloc[start_rank-1:end_rank].copy()
    
    def get_all_results(self) -> pd.DataFrame:
        """Get all results from the last search"""
        
        if self.last_results_df is None:
            raise ValueError("No previous search results available")
        
        return self.last_results_df.copy()
    
    def preview_dataset(self, dataset_url: str, source: str) -> Dict[str, Any]:
        """
        Get basic preview of dataset contents and structure
        Note: This is basic preview - full data processing is Agent 2's job
        """
        
        try:
            if source == 'data.gov':
                # For data.gov, try to get dataset details
                dataset_id = dataset_url.split('/')[-1]
                details = self.data_gov.get_dataset_details(dataset_id)
                
                preview = {
                    'source': 'data.gov',
                    'resources': [],
                    'formats': set(),
                    'total_resources': 0
                }
                
                for resource in details.get('resources', []):
                    if resource.get('url') and resource.get('format'):
                        preview['resources'].append({
                            'name': resource.get('name', 'Unknown'),
                            'format': resource.get('format', 'Unknown'),
                            'size': resource.get('size', 'Unknown'),
                            'url': resource.get('url')
                        })
                        preview['formats'].add(resource['format'].upper())
                
                preview['total_resources'] = len(preview['resources'])
                preview['formats'] = list(preview['formats'])
                
                return preview
                
            elif source == 'kaggle':
                # For Kaggle, try to get file list
                dataset_id = dataset_url.split('/')[-1]
                details = self.kaggle.get_dataset_details(dataset_id)
                
                return {
                    'source': 'kaggle',
                    'files': details.get('files', []),
                    'total_files': len(details.get('files', [])),
                    'metadata': details.get('metadata', {})
                }
                
        except Exception as e:
            return {
                'error': f"Could not preview dataset: {e}",
                'note': "Full data access and column analysis will be available in Agent 2"
            }
    
    def display_results_table(self, df: pd.DataFrame, max_description_length: int = 80):
        """Display results in a nice table format"""
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Rank", style="dim", width=4)
        table.add_column("Title", style="bold", max_width=30)
        table.add_column("Source", style="green", width=10)
        table.add_column("Score", style="yellow", width=6)
        table.add_column("Description", style="dim", max_width=max_description_length)
        
        for _, row in df.iterrows():
            desc = row['description']
            if len(desc) > max_description_length:
                desc = desc[:max_description_length-3] + "..."
            
            table.add_row(
                str(row['rank']),
                row['title'],
                row['source'],
                str(row['overall_score']),
                desc
            )
        
        self.console.print(table)