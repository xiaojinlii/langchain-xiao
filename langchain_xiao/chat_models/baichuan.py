from typing import List, Optional, Any, Iterator, Dict, Mapping, Type

import requests
from langchain_community.chat_models import baichuan
from langchain_community.chat_models.baichuan import ChatBaichuan
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGenerationChunk

from langchain_core.messages import (
    SystemMessage,
    SystemMessageChunk,
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")

    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        return AIMessage(content=_dict.get("content", "") or "")
    elif role == "system":
        return SystemMessage(content=_dict.get("content", "") or "")
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_delta_to_message_chunk(
        _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)


# 添加 system message
baichuan._convert_message_to_dict = _convert_message_to_dict
baichuan._convert_dict_to_message = _convert_dict_to_message
baichuan._convert_delta_to_message_chunk = _convert_delta_to_message_chunk


class MyChatBaichuan(ChatBaichuan):
    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # 调用stream和astream时，必须手动在kwargs里添加上stream=True才行
        kwargs.update({"stream": True})
        return super()._stream(messages, **kwargs)

    def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> requests.Response:
        """
        增加max_tokens参数
        """

        parameters = {**self._default_params, **kwargs}

        model = parameters.pop("model")
        headers = parameters.pop("headers", {})
        temperature = parameters.pop("temperature", 0.3)
        top_k = parameters.pop("top_k", 5)
        top_p = parameters.pop("top_p", 0.85)
        with_search_enhance = parameters.pop("with_search_enhance", False)
        stream = parameters.pop("stream", False)
        max_tokens = parameters.pop("max_tokens", 2048)

        payload = {
            "model": model,
            "messages": [_convert_message_to_dict(m) for m in messages],
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "with_search_enhance": with_search_enhance,
            "stream": stream,
            "max_tokens": max_tokens
        }

        url = self.baichuan_api_base
        api_key = ""
        if self.baichuan_api_key:
            api_key = self.baichuan_api_key.get_secret_value()

        res = requests.post(
            url=url,
            timeout=self.request_timeout,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                **headers,
            },
            json=payload,
            stream=self.streaming,
        )
        return res
