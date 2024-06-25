from typing import List, Optional, Dict, Any, Union

import httpx
from httpx._types import AuthTypes, HeaderTypes, CookieTypes, VerifyTypes, CertTypes
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator, Field


def _get_error_string(response: httpx.Response) -> str:
    try:
        if response.content:
            error_detail = response.json().get("detail")
            if error_detail:
                return error_detail
    except ValueError:
        pass

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        return str(e)

    return "Unknown error"


class XinferenceEmbeddings(BaseModel, Embeddings):

    sync_client: Any
    async_client: Any

    url: str
    model: str

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

    def create_embedding(self, input: Union[str, List[str]], **kwargs):
        """Create an Embedding from user input via RESTful APIs."""

        request_body = {
            "model": self.model,
            "input": input,
        }
        request_body.update(kwargs)
        response = self.sync_client.post("v1/embeddings", json=request_body)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to create the embeddings, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data

    async def acreate_embedding(self, input: Union[str, List[str]], **kwargs):
        """Asynchronous Create an Embedding from user input via RESTful APIs."""

        request_body = {
            "model": self.model,
            "input": input,
        }
        request_body.update(kwargs)
        response = await self.async_client.post("v1/embeddings", json=request_body)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to create the embeddings, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""

        response_data = self.create_embedding(texts)
        embeddings = response_data["data"]
        return [list(map(float, e["embedding"])) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query of documents using Xinference.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """

        response_data = self.create_embedding(text)
        embedding = response_data["data"][0]["embedding"]
        return list(map(float, embedding))

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""

        response_data = await self.acreate_embedding(texts)
        embeddings = response_data["data"]
        return [list(map(float, e["embedding"])) for e in embeddings]

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""

        response_data = await self.acreate_embedding(text)
        embedding = response_data["data"][0]["embedding"]
        return list(map(float, embedding))
