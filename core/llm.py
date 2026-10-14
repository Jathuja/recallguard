"""Local Ollama client configuration and inference wrapper for RecallGuard.

Enforces local execution using Ollama with the default model 'qwen3.5:0.8b'.
Cloud APIs (OpenAI, Gemini, Anthropic, etc.) are strictly prohibited.
"""

import os
from typing import Any, Dict, List, Optional
import ollama

from core.schemas import OllamaConfig

DEFAULT_MODEL = "qwen3.5:0.8b"
DEFAULT_HOST = "http://localhost:11434"


class OllamaClient:
    """Client for local inference via Ollama."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self.host = host or os.getenv("OLLAMA_HOST", DEFAULT_HOST)
        self.timeout = timeout
        self._client = ollama.Client(host=self.host, timeout=self.timeout)

    @property
    def config(self) -> OllamaConfig:
        """Return the current Ollama configuration schema."""
        return OllamaConfig(
            host=self.host,
            model=self.model,
            timeout_seconds=self.timeout,
        )

    def check_connection(self) -> bool:
        """Verify that the local Ollama daemon is reachable."""
        try:
            self._client.list()
            return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return a list of locally available model names."""
        try:
            response = self._client.list()
            # ollama-python response may contain models as objects with 'model' attribute or 'name'
            models = []
            for item in response.get("models", []):
                if hasattr(item, "model"):
                    models.append(item.model)
                elif isinstance(item, dict) and "model" in item:
                    models.append(item["model"])
                elif hasattr(item, "name"):
                    models.append(item.name)
                elif isinstance(item, dict) and "name" in item:
                    models.append(item["name"])
            return models
        except Exception:
            return []

    def is_model_available(self, model_name: Optional[str] = None) -> bool:
        """Check if the configured model is installed locally in Ollama."""
        target = model_name or self.model
        available = self.list_models()
        # Check for exact match or prefix match (e.g. qwen3.5:0.8b vs qwen3.5:0.8b-latest)
        return any(target in m or m.startswith(target) for m in available)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Foundational generation call to the local Ollama model.
        
        Args:
            prompt: User or prompt text to pass to Ollama.
            system: Optional system instruction prompt.
            options: Optional inference parameters (temperature, num_predict, etc.).
            
        Returns:
            The generated string response.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat(
            model=self.model,
            messages=messages,
            options=options or {},
        )
        return response["message"]["content"]
