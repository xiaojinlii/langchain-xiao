from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_xiao.chat_models.utils import get_chat_model


async def main():
    # instance_type = "ChatLlamaCpp"
    # model_kwargs = {
    #     "model_path": r"E:\WorkSpace\LLMWorkSpace\Models\LLM\qwen\Qwen1.5-0.5B-Chat-GGUF\qwen1_5-0_5b-chat-q5_k_m.gguf",
    #     "verbose": True,
    # }
    instance_type = "ChatOpenAI"
    model_kwargs = {
        "model_name": "moonshot-v1-8k",
        "openai_api_key": "sk-xxx",
        "openai_api_base": "https://api.moonshot.cn/v1",
    }

    chat_model = get_chat_model(instance_type, **model_kwargs)
    messages = [
        SystemMessage(content="你是一个聪明的助手，请根据用户的提示来完成任务"),
        HumanMessage(content="介绍一下你自己"),
    ]
    print(await chat_model.ainvoke(messages))
    # async for chunk in chat_model.astream(messages):
    #     print(chunk)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
