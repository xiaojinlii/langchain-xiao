import operator
from typing import Optional, Sequence, Dict, Any, List

import httpx
from httpx._types import AuthTypes, HeaderTypes, CookieTypes, VerifyTypes, CertTypes
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.pydantic_v1 import root_validator, Field


class FastllmReranker(BaseDocumentCompressor):

    sync_client: Any
    async_client: Any

    url: str
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

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if len(documents) == 0:  # to avoid empty api call
            return []

        texts = [d.page_content for d in documents]
        data = {
            "query": query,
            "texts": texts
        }

        response = self.sync_client.post("compute_score_by_query", json=data)
        return self.handle_response(response.json(), documents)

    async def acompress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if len(documents) == 0:  # to avoid empty api call
            return []

        texts = [d.page_content for d in documents]
        data = {
            "query": query,
            "texts": texts
        }

        response = await self.async_client.post("compute_score_by_query", json=data)
        return self.handle_response(response.json(), documents)

    def handle_response(self, scores: List[float], documents: Sequence[Document]) -> Sequence[Document]:
        docs_with_scores = list(zip(documents, scores))
        result_sorted = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        result = [doc for doc, score in result_sorted if score > self.score_threshold]
        return result[: self.top_n]
