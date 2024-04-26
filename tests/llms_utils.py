from langchain_xiao.llms.utils import get_llm


async def main():
    instance_type = "MyGPT4All"
    model_kwargs = {
        "model": r"E:\WorkSpace\LLMWorkSpace\Models\LLM\qwen\Qwen1.5-0.5B-Chat-GGUF\qwen1_5-0_5b-chat-q5_k_m.gguf",
    }

    llm = get_llm(instance_type, **model_kwargs)
    # print(await llm.ainvoke("你是谁"))
    async for chunk in llm.astream("你是谁"):
        print(chunk)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
