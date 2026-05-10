from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional
from urllib import error, request

from pydantic import BaseModel, Field


class ProviderClientError(RuntimeError):
    pass


class ProviderClientConfig(BaseModel):
    model: str = "default"
    timeout_seconds: int = 60
    max_retries: int = 2
    api_base_env: str = "OLM_API_BASE"
    api_key_env: str = "OLM_API_KEY"
    extra_headers: Dict[str, str] = Field(default_factory=dict)


class StructuredResponsesClient:
    def __init__(self, config: Optional[ProviderClientConfig] = None) -> None:
        self.config = config or ProviderClientConfig()

    def generate_json(
        self,
        schema_name: str,
        schema: Dict[str, Any],
        system_prompt: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        api_key = os.environ.get(self.config.api_key_env)
        api_base = os.environ.get(self.config.api_base_env)
        if not api_key:
            raise ProviderClientError(
                "Missing provider API key. Set OLM_API_KEY before using the provider backend."
            )
        if not api_base:
            raise ProviderClientError(
                "Missing provider API base URL. Set OLM_API_BASE before using the provider backend."
            )

        body = {
            "model": self.config.model,
            "input": json.dumps(
                {
                    "instruction": system_prompt,
                    "payload": payload,
                },
                ensure_ascii=True,
            ),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        raw = self._post_json(body, api_key, api_base)
        parsed = self._extract_json_object(raw)
        if not isinstance(parsed, dict):
            raise ProviderClientError(
                f"Expected object output for schema '{schema_name}', got: {type(parsed).__name__}"
            )
        return parsed

    def _post_json(self, body: Dict[str, Any], api_key: str, api_base: str) -> Dict[str, Any]:
        encoded = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.config.extra_headers)
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            req = request.Request(api_base, data=encoded, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                body_text = exc.read().decode("utf-8", errors="replace")
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.config.max_retries:
                    time.sleep(1.2 * (attempt + 1))
                    last_error = exc
                    continue
                raise ProviderClientError(f"Provider API error {exc.code}: {body_text}") from exc
            except error.URLError as exc:
                if attempt < self.config.max_retries:
                    time.sleep(1.2 * (attempt + 1))
                    last_error = exc
                    continue
                raise ProviderClientError(f"Network error while calling provider API: {exc}") from exc
        raise ProviderClientError(f"Provider API request failed after retries: {last_error}")

    def _extract_json_object(self, raw: Dict[str, Any]) -> Any:
        text = raw.get("output_text")
        if isinstance(text, str) and text.strip():
            return json.loads(text)
        for item in raw.get("output", []):
            if isinstance(item, dict) and item.get("type") == "message":
                for content in item.get("content", []):
                    candidate = content.get("text") or content.get("output_text")
                    if isinstance(candidate, str) and candidate.strip():
                        return json.loads(candidate)
        raise ProviderClientError(f"Could not extract structured JSON output from response: {raw}")
