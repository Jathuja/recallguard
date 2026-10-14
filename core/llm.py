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
        timeout: float = 60.0,
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
            models = []
            model_items = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
            for item in model_items:
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
        return any(target in m or m.startswith(target) for m in available)

    def chat_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        format_type: str = "json",
        think: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Execute a structured chat completion using local Ollama.
        
        Args:
            prompt: User message content.
            system_prompt: System instruction prompt.
            format_type: Format constraint (default 'json').
            think: Whether to allow reasoning thinking tokens (default False for fast direct JSON).
            options: Additional inference options like temperature or num_predict.
            
        Returns:
            Extracted text response from the model.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        opts = {"temperature": 0.0, "num_predict": 1024}
        if options:
            opts.update(options)

        # Call ollama chat, passing think=False if supported
        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                format=format_type,
                think=think,
                options=opts,
            )
        except TypeError:
            # Fallback for client versions that don't take think parameter
            response = self._client.chat(
                model=self.model,
                messages=messages,
                format=format_type,
                options=opts,
            )

        # Extract content supporting both Pydantic ChatResponse and dict responses
        content = ""
        if hasattr(response, "message"):
            msg = response.message
            content = getattr(msg, "content", "") or ""
            if not content and hasattr(msg, "thinking"):
                content = getattr(msg, "thinking", "") or ""
        elif isinstance(response, dict) and "message" in response:
            msg = response["message"]
            content = msg.get("content", "") or msg.get("thinking", "")

        return content

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Foundational generation call to the local Ollama model."""
        return self.chat_structured(
            prompt=prompt,
            system_prompt=system,
            format_type="",
            options=options,
        )
