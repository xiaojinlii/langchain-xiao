from langchain_xiao.embeddings.fastllm import FastllmEmbeddings


async def main():
    embeddings = FastllmEmbeddings(url="http://127.0.0.1:9000/embeddings")
    print(embeddings.embed_query("哈喽"))
    print(embeddings.embed_documents(["哈喽", "嗨"]))
    print(await embeddings.aembed_query("哈喽"))
    print(await embeddings.aembed_documents(["哈喽", "嗨"]))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
