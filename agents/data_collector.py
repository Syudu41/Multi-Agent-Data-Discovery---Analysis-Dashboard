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
from tools.data_sources.world_bank_connector import WorldBankConnector
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
    
    # Domain-specific synonym dictionaries
    DOMAIN_SYNONYMS = {
        'education': {
            'primary_terms': ['education', 'school', 'learning', 'academic', 'student', 'teacher', 'university', 'college'],
            'synonyms': ['educational', 'pedagogy', 'curriculum', 'enrollment', 'literacy', 'graduation', 'achievement', 'performance']
        },
        'health': {
            'primary_terms': ['health', 'medical', 'healthcare', 'medicine', 'hospital', 'patient', 'disease', 'clinical'],
            'synonyms': ['wellness', 'mortality', 'morbidity', 'epidemic', 'treatment', 'therapy', 'diagnosis', 'symptoms']
        },
        'economy': {
            'primary_terms': ['economy', 'economic', 'gdp', 'finance', 'financial', 'income', 'employment', 'business'],
            'synonyms': ['growth', 'development', 'inflation', 'unemployment', 'productivity', 'trade', 'investment', 'revenue']
        },
        'climate': {
            'primary_terms': ['climate', 'weather', 'temperature', 'environmental', 'environment', 'emissions', 'carbon'],
            'synonyms': ['greenhouse', 'warming', 'pollution', 'sustainability', 'renewable', 'energy', 'atmosphere', 'ecology']
        },
        'technology': {
            'primary_terms': ['technology', 'tech', 'digital', 'software', 'computer', 'internet', 'data', 'ai'],
            'synonyms': ['machine learning', 'artificial intelligence', 'algorithms', 'programming', 'automation', 'innovation']
        },
        'population': {
            'primary_terms': ['population', 'demographic', 'census', 'people', 'residents', 'citizens', 'community'],
            'synonyms': ['migration', 'birth', 'death', 'age', 'ethnicity', 'household', 'family', 'social']
        }
    }
    
    # Multi-word phrases that should be preserved as units
    IMPORTANT_PHRASES = [
        'machine learning', 'artificial intelligence', 'climate change', 'global warming',
        'public health', 'mental health', 'economic growth', 'gdp growth', 'unemployment rate',
        'birth rate', 'death rate', 'life expectancy', 'income inequality', 'poverty rate',
        'renewable energy', 'carbon emissions', 'greenhouse gas', 'education spending',
        'healthcare costs', 'housing market', 'labor force', 'consumer price'
    ]
    
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
        
        # Initialize World Bank connector
        self.worldbank = WorldBankConnector(
            cache_dir=settings.cache_dir,
            verbose=verbose
        )
        
        # Storage for results
        self.last_results_df = None
        self.last_query = None
        self.current_display_position = 0  # Track where user is in results
    
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
            
            if self.verbose:
                self.console.print(f"[dim]📊 data.gov returned {len(data_gov_results)} results[/dim]")
            
            task = progress.add_task("📊 Searching Kaggle...", total=None)
            kaggle_results = self.kaggle.search(search_params)
            progress.update(task, completed=True)
            
            if self.verbose:
                self.console.print(f"[dim]🏆 Kaggle returned {len(kaggle_results)} results[/dim]")
            
            task = progress.add_task("🌍 Searching World Bank...", total=None)
            worldbank_results = self.worldbank.search(search_params)
            progress.update(task, completed=True)
            
            if self.verbose:
                self.console.print(f"[dim]🌍 World Bank returned {len(worldbank_results)} results[/dim]")
            
            # Step 3: Combine and rank results
            task = progress.add_task("⚖️ Ranking datasets by relevance...", total=None)
            all_datasets = self._combine_results(data_gov_results, kaggle_results, worldbank_results)
            
            if self.verbose:
                source_breakdown = {}
                for dataset in all_datasets:
                    source = dataset.source
                    source_breakdown[source] = source_breakdown.get(source, 0) + 1
                self.console.print(f"[dim]📈 Combined results by source: {source_breakdown}[/dim]")
            
            ranked_datasets = self._rank_datasets(all_datasets, query, search_params)
            progress.update(task, completed=True)
        
        # Return results with proper error handling
        if len(ranked_datasets) == 0:
            # No results found
            if self.verbose:
                self.console.print("[yellow]⚠️ No datasets found for this query[/yellow]")
            
            # Return empty DataFrame with proper structure
            empty_df = pd.DataFrame(columns=[
                'rank', 'title', 'description', 'source', 'url', 
                'relevance_score', 'quality_score', 'recency_score', 
                'overall_score', 'full_metadata'
            ])
            
            self.last_results_df = empty_df
            self.last_query = query
            self.current_display_position = 0
            
            return empty_df
        
        if self.verbose:
            self.console.print(f"[dim]✅ Found {len(ranked_datasets)} total datasets[/dim]")
        
        # Convert to DataFrame
        df_data = []
        for i, dataset in enumerate(ranked_datasets):
            try:
                df_data.append({
                    'rank': i + 1,
                    'title': dataset.title or 'Untitled Dataset',
                    'description': (dataset.description[:200] + "..." if len(dataset.description or '') > 200 
                                  else dataset.description or 'No description available'),
                    'source': dataset.source,
                    'url': dataset.url or 'N/A',
                    'relevance_score': round(dataset.relevance_score, 2),
                    'quality_score': round(dataset.quality_score, 2),
                    'recency_score': round(dataset.recency_score, 2),
                    'overall_score': round(dataset.overall_score, 2),
                    'full_metadata': dataset.metadata or {}
                })
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]⚠️ Error processing dataset {i+1}: {e}[/yellow]")
                continue
        
        df = pd.DataFrame(df_data)
        
        # Save full results to files
        self._save_results(df, query)
        
        # Store in instance for later access
        self.last_results_df = df
        self.last_query = query
        self.current_display_position = max_results  # Track what we've shown
        
        return df.head(max_results)  # Return top N for display
    
    def _parse_query_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to extract search parameters from natural language query"""
        
        prompt = f"""
        You are a data discovery expert. Parse this natural language query into structured search parameters.
        
        Query: "{query}"
        
        Extract the following information and respond in JSON format:
        {{
            "keywords": ["list", "of", "relevant", "search", "terms"],
            "domain": "primary subject area (education, health, economics, climate, technology, population, etc.)",
            "data_types": ["statistical", "survey", "time-series", "geographic", etc.],
            "geographic_scope": "local/national/international/global",
            "time_relevance": "current/historical/future/any",
            "priority_keywords": ["most", "important", "terms"],
            "preserved_phrases": ["important", "multi-word", "phrases"],
            "query_complexity": "simple or complex"
        }}
        
        Focus on:
        1. Preserving important multi-word phrases like "machine learning", "climate change"
        2. Including domain-specific synonyms and related terms
        3. Identifying the primary subject area accurately
        4. Extracting geographic and temporal context
        """
        
        try:
            response = self.llm.generate(prompt)
            # Try to parse JSON response
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_text = response[json_start:json_end]
                llm_result = json.loads(json_text)
                
                # Enhance LLM result with our domain intelligence
                enhanced_result = self._enhance_llm_result(llm_result, query)
                
                if self.verbose:
                    self.console.print(f"[dim]🤖 LLM enhanced result: {len(enhanced_result.get('keywords', []))} keywords, domain: {enhanced_result.get('domain')}[/dim]")
                
                return enhanced_result
            else:
                # Fallback to enhanced extraction
                return self._basic_keyword_extraction(query)
                
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]⚠️ LLM parsing failed: {e}. Using enhanced fallback.[/yellow]")
            return self._basic_keyword_extraction(query)
    
    def _enhance_llm_result(self, llm_result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Enhance LLM results with our domain intelligence"""
        
        # Get domain from LLM or detect it ourselves
        llm_domain = llm_result.get('domain', 'general')
        
        # If LLM detected a domain, use our synonym expansion
        if llm_domain != 'general' and llm_domain in self.DOMAIN_SYNONYMS:
            domain_info = self.DOMAIN_SYNONYMS[llm_domain]
            
            # Add domain synonyms to keywords if not already present
            current_keywords = llm_result.get('keywords', [])
            for synonym in domain_info['synonyms'][:5]:  # Add top 5 synonyms
                if synonym not in current_keywords:
                    current_keywords.append(synonym)
            
            llm_result['keywords'] = current_keywords
        
        # Ensure we have preserved phrases
        if 'preserved_phrases' not in llm_result:
            preserved_phrases = []
            query_lower = original_query.lower()
            for phrase in self.IMPORTANT_PHRASES:
                if phrase in query_lower:
                    preserved_phrases.append(phrase)
            llm_result['preserved_phrases'] = preserved_phrases
        
        # Add detected domains list
        detected_domains = [llm_domain] if llm_domain != 'general' else []
        llm_result['detected_domains'] = detected_domains
        
        # Ensure all required fields exist
        defaults = {
            "keywords": [],
            "domain": "general", 
            "data_types": ["any"],
            "geographic_scope": "any",
            "time_relevance": "any",
            "priority_keywords": [],
            "preserved_phrases": [],
            "query_complexity": "simple"
        }
        
        for key, default_value in defaults.items():
            if key not in llm_result:
                llm_result[key] = default_value
        
        return llm_result
    
    def _basic_keyword_extraction(self, query: str) -> Dict[str, Any]:
        """Enhanced keyword extraction with domain detection and synonym expansion"""
        
        query_lower = query.lower().strip()
        
        if self.verbose:
            self.console.print(f"[dim]🔍 Enhanced query processing: '{query}'[/dim]")
        
        # Step 1: Preserve important multi-word phrases
        preserved_phrases = []
        remaining_query = query_lower
        
        for phrase in self.IMPORTANT_PHRASES:
            if phrase in remaining_query:
                preserved_phrases.append(phrase)
                # Remove the phrase from remaining query to avoid double-processing
                remaining_query = remaining_query.replace(phrase, ' ')
        
        # Step 2: Extract individual words from remaining query
        import re
        words = re.findall(r'\b\w+\b', remaining_query)
        
        # Enhanced stop words - keep domain-relevant words
        enhanced_stop_words = {
            'how', 'does', 'what', 'where', 'when', 'why', 'is', 'are', 'was', 'were',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'a', 'an', 'this', 'that', 'these', 'those', 'can', 'could', 'would', 'should'
        }
        
        # Keep words longer than 2 chars and not in stop words
        individual_words = [word for word in words 
                          if len(word) > 2 and word not in enhanced_stop_words]
        
        # Step 3: Domain detection
        detected_domains = self._detect_domains(preserved_phrases + individual_words)
        
        # Step 4: Build comprehensive keyword lists
        primary_keywords = preserved_phrases + individual_words  # Original query terms
        
        # Step 5: Add domain-specific synonyms
        expanded_keywords = list(primary_keywords)  # Start with originals
        
        for domain in detected_domains:
            domain_info = self.DOMAIN_SYNONYMS.get(domain, {})
            
            # Add relevant synonyms for this domain
            for synonym in domain_info.get('synonyms', []):
                if synonym not in expanded_keywords:
                    expanded_keywords.append(synonym)
        
        # Step 6: Prioritize keywords
        # Priority 1: Preserved phrases (highest)
        # Priority 2: Individual words from original query
        # Priority 3: Domain synonyms
        priority_keywords = preserved_phrases + individual_words[:5]  # Top 5 individual words
        
        # Step 7: Determine primary domain for context
        primary_domain = detected_domains[0] if detected_domains else 'general'
        
        if self.verbose:
            self.console.print(f"[dim]📝 Preserved phrases: {preserved_phrases}[/dim]")
            self.console.print(f"[dim]🎯 Detected domains: {detected_domains}[/dim]")
            self.console.print(f"[dim]🔑 Priority keywords: {priority_keywords}[/dim]")
            self.console.print(f"[dim]📚 Total expanded keywords: {len(expanded_keywords)}[/dim]")
        
        return {
            "keywords": expanded_keywords,
            "domain": primary_domain,
            "detected_domains": detected_domains,
            "data_types": self._infer_data_types(primary_domain, preserved_phrases),
            "geographic_scope": self._infer_geographic_scope(query_lower),
            "time_relevance": self._infer_time_relevance(query_lower),
            "priority_keywords": priority_keywords,
            "preserved_phrases": preserved_phrases,
            "query_complexity": "complex" if len(preserved_phrases) > 0 else "simple"
        }
    
    def _detect_domains(self, keywords: List[str]) -> List[str]:
        """Detect relevant domains based on keywords"""
        
        detected = []
        keyword_text = ' '.join(keywords).lower()
        
        for domain, terms in self.DOMAIN_SYNONYMS.items():
            # Check if any primary terms or synonyms are in the keywords
            all_domain_terms = terms['primary_terms'] + terms['synonyms']
            
            domain_matches = sum(1 for term in all_domain_terms if term in keyword_text)
            
            # If we find multiple matches or strong single matches, include domain
            if domain_matches >= 2 or any(term in keywords for term in terms['primary_terms'][:3]):
                detected.append(domain)
        
        # Sort by relevance (domains with more matches first)
        detected.sort(key=lambda d: sum(1 for term in self.DOMAIN_SYNONYMS[d]['primary_terms'] 
                                       if term in keyword_text), reverse=True)
        
        return detected[:3]  # Return top 3 most relevant domains
    
    def _infer_data_types(self, primary_domain: str, phrases: List[str]) -> List[str]:
        """Infer expected data types based on domain and phrases"""
        
        data_type_mapping = {
            'economy': ['statistical', 'time-series', 'financial'],
            'health': ['survey', 'statistical', 'clinical'],
            'education': ['survey', 'performance', 'demographic'],
            'climate': ['time-series', 'sensor', 'satellite'],
            'technology': ['transactional', 'behavioral', 'performance'],
            'population': ['census', 'demographic', 'survey']
        }
        
        # Check for specific indicators in phrases
        if any('rate' in phrase for phrase in phrases):
            return ['statistical', 'time-series']
        elif any('survey' in phrase for phrase in phrases):
            return ['survey', 'demographic']
        elif any('spending' in phrase or 'cost' in phrase for phrase in phrases):
            return ['financial', 'statistical']
        
        return data_type_mapping.get(primary_domain, ['any'])
    
    def _infer_geographic_scope(self, query: str) -> str:
        """Infer geographic scope from query"""
        
        # Look for geographic indicators
        if any(term in query for term in ['country', 'countries', 'national', 'by country']):
            return 'international'
        elif any(term in query for term in ['state', 'states', 'usa', 'united states', 'america']):
            return 'national'
        elif any(term in query for term in ['city', 'cities', 'local', 'county', 'district']):
            return 'local'
        elif any(term in query for term in ['global', 'world', 'worldwide', 'international']):
            return 'global'
        
        return 'any'
    
    def _infer_time_relevance(self, query: str) -> str:
        """Infer time relevance from query"""
        
        if any(term in query for term in ['current', 'recent', 'latest', '2024', '2023']):
            return 'current'
        elif any(term in query for term in ['historical', 'past', 'since', 'over time', 'trend']):
            return 'historical'
        elif any(term in query for term in ['forecast', 'prediction', 'future', 'projection']):
            return 'future'
        
        return 'any'
    
    def _combine_results(self, data_gov_results: List[Dict], kaggle_results: List[Dict], 
                        worldbank_results: List[Dict]) -> List[Dataset]:
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
        
        # Convert World Bank results
        for result in worldbank_results:
            datasets.append(Dataset(
                title=result.get('title', 'Unknown Indicator'),
                description=result.get('description', result.get('notes', '')),
                source='worldbank',
                url=result.get('url', ''),
                relevance_score=0.0,  # Will be calculated later
                quality_score=self._calculate_quality_score(result, 'worldbank'),
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
        """Calculate relevance score based on text matching with fuzzy logic"""
        
        # Combine title and description for scoring
        text = f"{dataset.title} {dataset.description}".lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # 1. Exact query match (highest score)
        if query_lower in text:
            score += 4.0
        
        # 2. Fuzzy matching for partial queries
        query_words = query_lower.split()
        text_words = text.split()
        
        for query_word in query_words:
            if len(query_word) > 2:  # Skip very short words
                # Exact word match
                if query_word in text_words:
                    score += 2.0
                    continue
                
                # Partial/fuzzy match
                for text_word in text_words:
                    if len(text_word) > 2:
                        # Check if query word is substring of text word
                        if query_word in text_word:
                            score += 1.0
                            break
                        # Check if text word is substring of query word  
                        elif text_word in query_word:
                            score += 0.8
                            break
                        # Check for similar start (first 3 chars)
                        elif query_word[:3] == text_word[:3] and len(query_word) > 3:
                            score += 0.5
                            break
        
        # 3. Individual keyword matches
        for keyword in search_params.get('keywords', []):
            if keyword.lower() in text:
                score += 1.2
        
        # 4. Priority keyword matches (weighted higher)
        for keyword in search_params.get('priority_keywords', []):
            if keyword.lower() in text:
                score += 2.5
        
        # 5. Domain match bonus
        domain = search_params.get('domain', '').lower()
        if domain and domain != 'general' and domain in text:
            score += 1.8
        
        # 6. Source-specific bonuses
        if dataset.source == 'worldbank':
            # World Bank bonus for economic/development queries
            econ_terms = ['gdp', 'economic', 'development', 'poverty', 'income', 'growth']
            for term in econ_terms:
                if term in query_lower and term in text:
                    score += 1.0
                    break
        
        elif dataset.source == 'kaggle':
            # Kaggle bonus for ML/data science queries
            ml_terms = ['machine learning', 'prediction', 'classification', 'analysis', 'model']
            for term in ml_terms:
                if term in query_lower and term in text:
                    score += 0.8
                    break
        
        # 7. Title vs description weighting
        title_lower = dataset.title.lower()
        if any(word in title_lower for word in query_lower.split()):
            score += 1.5  # Title matches are more important
        
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
        
        elif source == 'worldbank':
            # World Bank data is official and high quality
            score += 2.5  # Higher than government data
            
            # Check for topics/categories
            if result.get('topics') and len(result.get('topics', [])) > 0:
                score += 1.0  # Well-categorized
            
            # Check for recent updates
            if result.get('last_updated'):
                score += 0.5  # Recently maintained
        
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
            results_dir.mkdir(parents=True, exist_ok=True)
            
            if self.verbose:
                self.console.print(f"[dim]📁 Creating results directory: {results_dir}[/dim]")
            
            # Generate filename from query
            import re
            safe_query = re.sub(r'[^\w\s-]', '', query).strip()
            safe_query = re.sub(r'[-\s]+', '_', safe_query)[:50]
            timestamp = int(time.time())
            
            filename_base = f"{safe_query}_{timestamp}"
            
            if self.verbose:
                self.console.print(f"[dim]💾 Saving results as: {filename_base}[/dim]")
            
            # Save as CSV (without metadata for readability)
            csv_path = results_dir / f"{filename_base}.csv"
            df_export = df.drop('full_metadata', axis=1, errors='ignore')  # Remove complex metadata for CSV
            df_export.to_csv(csv_path, index=False, encoding='utf-8')
            
            if self.verbose:
                self.console.print(f"[dim]✅ CSV saved: {csv_path}[/dim]")
            
            # Save as JSON (with full metadata)
            json_path = results_dir / f"{filename_base}.json"
            # Convert DataFrame to dict for JSON serialization
            json_data = []
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                # Handle any non-serializable data
                for key, value in row_dict.items():
                    if hasattr(value, 'to_dict'):
                        row_dict[key] = value.to_dict()
                    elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        row_dict[key] = str(value)
                json_data.append(row_dict)
            
            import json
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            if self.verbose:
                self.console.print(f"[dim]✅ JSON saved: {json_path}[/dim]")
            
            self.console.print(f"[green]💾 Results saved to: {results_dir}[/green]")
                
        except Exception as e:
            self.console.print(f"[red]❌ Failed to save results: {e}[/red]")
            if self.verbose:
                import traceback
                self.console.print(f"[red]{traceback.format_exc()}[/red]")
    
    def get_more_results(self, count: int = 10) -> pd.DataFrame:
        """Get more results from the last search"""
        
        if self.last_results_df is None:
            raise ValueError("No previous search results available")
        
        start_pos = self.current_display_position
        end_pos = start_pos + count
        
        more_data = self.last_results_df.iloc[start_pos:end_pos].copy()
        
        # Update position for next "more" request
        if not more_data.empty:
            self.current_display_position = end_pos
            
        return more_data
    
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
                
            elif dataset['source'] == 'kaggle':
                # For Kaggle, try to get file list
                dataset_id = dataset['url'].split('/')[-1]
                if '/' in dataset.get('id', ''):
                    dataset_id = dataset['id']  # Use the full ref if available
                
                details = self.kaggle.get_dataset_details(dataset_id)
                
                return {
                    'source': 'kaggle',
                    'files': details.get('files', []),
                    'total_files': len(details.get('files', [])),
                    'metadata': details.get('metadata', {}),
                    'file_count_from_metadata': dataset.get('fileCount', 0),  # Fallback from original metadata
                    'download_count': dataset.get('downloadCount', 0),
                    'vote_count': dataset.get('voteCount', 0)
                }
            elif source == 'worldbank':
                # For World Bank, show indicator information
                return {
                    'source': 'worldbank',
                    'indicator_id': dataset.get('id', 'N/A'),
                    'topics': dataset.get('topics', []),
                    'source_organization': dataset.get('source_organization', 'World Bank'),
                    'last_updated': dataset.get('last_updated', 'N/A'),
                    'data_url': dataset_url,
                    'note': 'This is a World Bank development indicator. Access full time-series data via the URL.'
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