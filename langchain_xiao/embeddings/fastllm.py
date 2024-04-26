from typing import List, Optional, Dict, Any

import httpx
from httpx._types import AuthTypes, HeaderTypes, CookieTypes, VerifyTypes, CertTypes
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator, Field


class FastllmEmbeddings(BaseModel, Embeddings):

    sync_client: Any
    async_client: Any

    url: str

    timeout: Optional[float] = None
    auth: Optional[AuthTypes] = None
    headers: Optional[HeaderTypes] = None
    cookies: Optional[CookieTypes] = None
    verify: VerifyTypes = True
    cert: Optional[CertTypes] = None
    client_kwargs: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        url = values["url"]
        values["url"] = url if url.endswith("/") else url + "/"

        params = {
            "base_url": url,
            "timeout": values["timeout"],
            "auth": values["auth"],
            "headers": values["headers"],
            "cookies": values["cookies"],
            "verify": values["verify"],
            "cert": values["cert"]
        }
        params.update(values["client_kwargs"])

        values["sync_client"] = httpx.Client(**params)
        values["async_client"] = httpx.AsyncClient(**params)
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        response = self.sync_client.post("embed_documents", json=texts)
        result = response.json()
        return result

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        response = self.sync_client.post("embed_query", json=text)
        result = response.json()
        return result

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        response = await self.async_client.post("embed_documents", json=texts)
        result = response.json()
        return result

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        response = await self.async_client.post("embed_query", json=text)
        result = response.json()
        return result
