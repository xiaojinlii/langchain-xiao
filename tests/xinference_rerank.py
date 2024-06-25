from langchain_core.documents import Document

from langchain_xiao.retrievers.document_compressors.xinference_rerank import XinferenceReranker


async def main():
    reranker_model = XinferenceReranker(url="http://10.12.25.5:9997", model="bge-reranker-base", top_n=3, score_threshold=0.7)
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
