"""
Base connector class for data sources
"""

import os
import time
import hashlib
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import diskcache as dc

class BaseConnector(ABC):
    """Base class for data source connectors"""
    
    def __init__(self, cache_dir: Path, cache_duration_hours: int = 24, verbose: bool = False):
        self.cache_dir = cache_dir
        self.cache_duration_hours = cache_duration_hours
        self.verbose = verbose
        
        # Initialize cache
        self.cache = dc.Cache(str(cache_dir / "api_cache"))
    
    @abstractmethod
    def search(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for datasets using the given parameters"""
        pass
    
    @abstractmethod
    def get_dataset_details(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific dataset"""
        pass
    
    def _cache_key(self, method: str, params: Dict[str, Any]) -> str:
        """Generate a cache key for the given method and parameters"""
        
        # Create a stable hash of the parameters
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        return f"{self.__class__.__name__}_{method}_{params_hash}"
    
    def _get_cached(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available and not expired"""
        
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                # Check if cache is still valid
                cached_time, result = cached_data
                age_hours = (time.time() - cached_time) / 3600
                
                if age_hours < self.cache_duration_hours:
                    if self.verbose:
                        print(f"Using cached result (age: {age_hours:.1f}h)")
                    return result
                else:
                    # Cache expired, remove it
                    del self.cache[cache_key]
                    
        except Exception as e:
            if self.verbose:
                print(f"Cache error: {e}")
                
        return None
    
    def _set_cached(self, cache_key: str, result: Any):
        """Cache the result with timestamp"""
        
        try:
            self.cache[cache_key] = (time.time(), result)
        except Exception as e:
            if self.verbose:
                print(f"Cache write error: {e}")
    
    def _make_request_with_cache(self, method: str, params: Dict[str, Any], request_func) -> Any:
        """Make a request with caching support"""
        
        cache_key = self._cache_key(method, params)
        
        # Try to get from cache first
        cached_result = self._get_cached(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Make the actual request
        result = request_func()
        
        # Cache the result
        self._set_cached(cache_key, result)
        
        return result
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "cache_directory": str(self.cache.directory),
            "cache_duration_hours": self.cache_duration_hours
        }