"""
Ollama LLM Client for PersonalDataAI

Handles communication with local Ollama instance
"""

import requests
import json
from typing import Dict, Any, Optional
from rich.console import Console

class OllamaClient:
    """Client for communicating with Ollama LLM"""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.1:8b", verbose: bool = False):
        self.host = host.rstrip('/')
        self.model = model
        self.verbose = verbose
        self.console = Console() if verbose else None
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama is running and model is available"""
        
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception(f"Ollama server not responding: {response.status_code}")
            
            # Check if model is available
            models = response.json()
            available_models = [model["name"] for model in models.get("models", [])]
            
            if self.model not in available_models:
                raise Exception(
                    f"Model '{self.model}' not found.\n"
                    f"Available models: {available_models}\n"
                    f"Install with: ollama pull {self.model}"
                )
                
            if self.verbose:
                self.console.print(f"[green]✅ Connected to Ollama - Model: {self.model}[/green]")
                
        except requests.exceptions.RequestException as e:
            raise Exception(
                f"Cannot connect to Ollama at {self.host}.\n"
                f"Make sure Ollama is running: ollama serve\n"
                f"Error: {e}"
            )
    
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: Optional[int] = None) -> str:
        """
        Generate response from Ollama
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        
        if self.verbose:
            self.console.print(f"[dim]🤖 Querying {self.model}...[/dim]")
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("response", "").strip()
            
            if self.verbose:
                self.console.print(f"[dim]💭 LLM Response ({len(generated_text)} chars): {generated_text[:100]}...[/dim]")
            
            return generated_text
            
        except requests.exceptions.Timeout:
            raise Exception("Ollama request timed out. Try a simpler query or check system resources.")
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama request failed: {e}")
        
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response from Ollama: {e}")
    
    def chat(self, messages: list, temperature: float = 0.1) -> str:
        """
        Chat-style interaction with context
        
        Args:
            messages: List of {"role": "user"/"assistant", "content": "text"}
            temperature: Sampling temperature
            
        Returns:
            Assistant response
        """
        
        # Convert messages to single prompt for simple generate API
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant: "
        
        return self.generate(prompt, temperature=temperature)
    
    def validate_json_response(self, prompt: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Generate and validate JSON response
        
        Args:
            prompt: Prompt that should return JSON
            max_retries: Number of retry attempts
            
        Returns:
            Parsed JSON dictionary
        """
        
        for attempt in range(max_retries + 1):
            try:
                response = self.generate(prompt, temperature=0.1)
                
                # Extract JSON from response
                if "{" in response and "}" in response:
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    json_text = response[json_start:json_end]
                    return json.loads(json_text)
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries:
                    raise Exception(f"Failed to get valid JSON after {max_retries + 1} attempts: {e}")
                
                if self.verbose:
                    self.console.print(f"[yellow]⚠️ JSON parsing failed (attempt {attempt + 1}), retrying...[/yellow]")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        
        try:
            response = requests.post(
                f"{self.host}/api/show",
                json={"name": self.model},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {"error": str(e)}
    
    def list_models(self) -> list:
        """List all available models"""
        
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            models_data = response.json()
            return [model["name"] for model in models_data.get("models", [])]
            
        except Exception as e:
            return []