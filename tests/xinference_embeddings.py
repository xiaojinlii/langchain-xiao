from langchain_xiao.embeddings.xinference import XinferenceEmbeddings


async def main():
    embeddings = XinferenceEmbeddings(url="http://10.12.25.5:9997", model="bge-large-zh-v1.5")
    print(embeddings.embed_query("哈喽"))
    print(await embeddings.aembed_query("哈喽"))
    print(embeddings.embed_documents(["哈喽", "嗨"]))
    print(await embeddings.aembed_documents(["哈喽", "嗨"]))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
