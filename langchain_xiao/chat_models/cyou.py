import hashlib
import json
import random
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type

import aiohttp
import requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage
)

from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


DEFAULT_API_BASE = "http://10.1.9.87:8100"
DEFAULT_API_URL = "/cyouNeiOpenAi/api/chatGpt"


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


def calculate_md5(input_string):
    md5 = hashlib.md5()
    md5.update(input_string.encode('utf-8'))
    encrypted = md5.hexdigest()
    return encrypted


class ChatCyou(BaseChatModel):

    api_base: str = Field(default=DEFAULT_API_BASE)
    api_url: str = Field(default=DEFAULT_API_URL)
    client_id: str = Field()
    private_key: str = Field()

    temperature: float = 1
    request_timeout: int = 60

    def _handle_request_params(self, messages: List[BaseMessage]):
        payload = {
            "bodyArray": [_convert_message_to_dict(m) for m in messages],
            "temperature": self.temperature,
        }

        json_data = json.dumps(payload)
        timestamp = int(time.time() * 1000)
        signature = calculate_md5(
            self.client_id + self.private_key + self.api_url + str(timestamp) + json_data
        )

        params = {
            'clientId': self.client_id,
            'timestamp': timestamp,
            'random': random.random(),
            'algorithm': "MD5",
            'sign': signature,
        }

        url = self.api_base + self.api_url
        return url, params, payload

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        url, params, body = self._handle_request_params(messages)
        print(f"request params:{params}")
        print(f"request body:{body}")
        start_time = time.time()
        response = requests.post(
            url=url,
            timeout=self.request_timeout,
            params=params,
            json=body
        )
        if response.status_code != 200:
            raise ValueError(f"Error from Cyou api response: {response.text}")
        ret = response.json()
        end_time = time.time()
        print(f"response:{ret}, time:{end_time-start_time}")
        return self._create_chat_result(ret)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        url, params, body = self._handle_request_params(messages)
        print(f"request params:{params}")
        print(f"request body:{body}")
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params, json=body) as response:
                if response.status != 200:
                    raise ValueError(f"Error from Cyou api response: {await response.text()}")
                ret = await response.json()
                end_time = time.time()
                print(f"response:{ret}, time:{end_time-start_time}")

        return self._create_chat_result(ret)

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []

        if response["msg"] is None:
            message = AIMessage(content=response["data"]["content"])
            total_tokens = response["data"]["totalTokens"]
        else:
            error_msg = response['msg'].split(', ', 1)[1]
            error_msg = error_msg.strip('"')
            msg_content = json.loads(error_msg)

            msg = msg_content['error']['message']
            status = msg_content['error']['status']

            message = AIMessage(content=f"error_code:{status} error_message:{msg}")
            total_tokens = 0

        gen = ChatGeneration(message=message)
        generations.append(gen)

        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": total_tokens
        }
        llm_output = {"token_usage": token_usage, "model": self._llm_type}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "cyou-chat"
