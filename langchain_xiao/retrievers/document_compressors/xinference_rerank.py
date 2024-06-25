import operator
from typing import Optional, Sequence, Dict, Any, List

import httpx
from httpx._types import AuthTypes, HeaderTypes, CookieTypes, VerifyTypes, CertTypes
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.pydantic_v1 import root_validator, Field


class XinferenceReranker(BaseDocumentCompressor):

    sync_client: Any
    async_client: Any

    url: str
    model: str
    top_n: int = 3
    """Number of documents to return."""
    score_threshold: float = Field(0.0, ge=0.0, le=1.0)
    """Score threshold of documents to return."""

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

    def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = None,
        **kwargs,
    ):
        """Returns an ordered list of documents ordered by their relevance to the provided query."""

        request_body = {
            "model": self.model,
            "documents": documents,
            "query": query,
            "top_n": top_n,
            "return_documents": return_documents,
        }
        request_body.update(kwargs)
        response = self.sync_client.post("v1/rerank", json=request_body)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to rerank documents, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        for r in response_data["results"]:
            r["document"] = documents[r["index"]]
        return response_data

    async def arerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = None,
        **kwargs,
    ):
        """Returns an ordered list of documents ordered by their relevance to the provided query."""

        request_body = {
            "model": self.model,
            "documents": documents,
            "query": query,
            "top_n": top_n,
            "return_documents": return_documents,
        }
        request_body.update(kwargs)
        response = await self.async_client.post("v1/rerank", json=request_body)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to rerank documents, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        for r in response_data["results"]:
            r["document"] = documents[r["index"]]
        return response_data

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if len(documents) == 0:  # to avoid empty api call
            return []

        texts = [d.page_content for d in documents]
        response_data = self.rerank(texts, query)
        return self.handle_response(response_data["results"], documents)

    async def acompress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if len(documents) == 0:  # to avoid empty api call
            return []

        texts = [d.page_content for d in documents]
        response_data = await self.arerank(texts, query)
        return self.handle_response(response_data["results"], documents)

    def handle_response(self, results: List[Dict], documents: Sequence[Document]) -> Sequence[Document]:
        topn = results[: self.top_n]
        ret = [documents[r["index"]] for r in topn if r["relevance_score"] > self.score_threshold]
        return ret
