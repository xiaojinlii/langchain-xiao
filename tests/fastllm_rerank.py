from langchain_core.documents import Document

from langchain_xiao.retrievers.document_compressors.fastllm_rerank import FastllmReranker


async def main():
    reranker_model = FastllmReranker(url="http://127.0.0.1:9000/reranker", top_n=3, score_threshold=0.7)
    docs = [
        Document(page_content="早上好"),
        Document(page_content="哈喽"),
        Document(page_content="嗨"),
        Document(page_content="你好"),
    ]
    result = reranker_model.compress_documents(docs, "哈喽")
    print(result)
    result = await reranker_model.acompress_documents(docs, "哈喽")
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
