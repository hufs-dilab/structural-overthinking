"""
Model configuration and factory.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from openai import Timeout
from smolagents import OpenAIModel

from .nowait_tokens import get_nowait_logit_bias


DEFAULT_BASE_URL = "http://localhost:8000/v1"


@dataclass
class ModelConfig:
    """Configuration for model initialization."""

    model_id: str
    api_base: str
    api_key: str
    nowait: bool = False
    max_tokens: int = 4096
    request_timeout: int = 300  # Timeout for each API request in seconds
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Create config from environment variables."""
        return cls(
            model_id=os.getenv("OPENAI_MODEL_ID") or os.getenv("VLLM_MODEL_ID", ""),
            api_base=os.getenv("OPENAI_BASE_URL")
            or os.getenv("VLLM_BASE_URL")
            or DEFAULT_BASE_URL,
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY") or "EMPTY",
        )

    @classmethod
    def from_args(
        cls,
        model_id: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        nowait: bool = False,
        max_tokens: int = 4096,
        request_timeout: int = 300,
    ) -> "ModelConfig":
        """
        Create config from arguments with env fallback.

        Args are used if provided, otherwise falls back to environment variables.
        """
        return cls(
            model_id=model_id or os.getenv("OPENAI_MODEL_ID") or os.getenv("VLLM_MODEL_ID", ""),
            api_base=api_base
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("VLLM_BASE_URL")
            or DEFAULT_BASE_URL,
            api_key=api_key or os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY") or "EMPTY",
            nowait=nowait,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )

    def _create_timeout(self) -> Timeout:
        """Create httpx Timeout with separate connect/read timeouts."""
        return Timeout(
            connect=60.0,  # 60s to establish connection
            read=float(self.request_timeout),  # read timeout for streaming response
            write=60.0,  # 60s to send request
            pool=30.0,  # 30s to acquire connection from pool
        )

    def create_model(self) -> OpenAIModel:
        """Create OpenAI model instance."""
        kwargs = self.extra_kwargs.copy()
        kwargs["max_tokens"] = self.max_tokens

        if self.nowait:
            print(f"🚫 NoWait mode enabled for model: {self.model_id}")
            logit_bias = get_nowait_logit_bias(self.model_id)
            if logit_bias:
                kwargs["logit_bias"] = logit_bias
                print(f"   Suppressing {len(logit_bias)} tokens")
            else:
                print("   ⚠️  Skipping nowait (no token IDs found)")

        print(f"⚙️  Max tokens: {self.max_tokens}")
        print(f"⏱️  Request timeout: {self.request_timeout}s (read)")

        return OpenAIModel(
            model_id=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            client_kwargs={"timeout": self._create_timeout()},
            **kwargs,
        )
