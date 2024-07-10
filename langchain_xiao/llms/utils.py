import importlib
from typing import Any

from langchain_core.language_models import BaseLLM

XIAO_CHAT_MODELS = [
    "MyGPT4All",
]


def get_llm(instance_type: str, **model_kwargs: Any) -> BaseLLM:
    if instance_type == "OpenAI":
        try:
            from langchain_openai import OpenAI
        except ImportError:
            raise ImportError(
                "Could not import langchain_openai python package. "
                "Please install it with `pip install langchain_openai`."
            )
        model = OpenAI(**model_kwargs)

    elif instance_type == "Tongyi":
        from langchain_community.llms import Tongyi
        model = Tongyi(**model_kwargs)
        model.model_name = model_kwargs["model"]

    else:
        if instance_type in XIAO_CHAT_MODELS:
            chat_models_module = importlib.import_module("langchain_xiao.llms")
        else:
            chat_models_module = importlib.import_module("langchain_community.llms")

        LLM = getattr(chat_models_module, instance_type)
        model = LLM(**model_kwargs)

    return model
