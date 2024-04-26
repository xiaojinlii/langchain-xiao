import importlib
from typing import Any

from langchain_core.language_models import BaseChatModel


XIAO_CHAT_MODELS = [
    "ChatLlamaCpp",
    "MyChatBaichuan",
]


def get_chat_model(instance_type: str, **model_kwargs: Any) -> BaseChatModel:
    if instance_type == "ChatOpenAI":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "Could not import langchain_openai python package. "
                "Please install it with `pip install langchain_openai`."
            )
        model = ChatOpenAI(**model_kwargs)

    elif instance_type == "ChatTongyi":
        from langchain_community.chat_models import ChatTongyi
        model = ChatTongyi(**model_kwargs)
        model.model_name = model_kwargs["model_name"]

    else:
        if instance_type in XIAO_CHAT_MODELS:
            chat_models_module = importlib.import_module("langchain_xiao.chat_models")
        else:
            chat_models_module = importlib.import_module("langchain_community.chat_models")

        ChatModel = getattr(chat_models_module, instance_type)
        model = ChatModel(**model_kwargs)

    return model
