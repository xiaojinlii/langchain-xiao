import json
import logging

from typing import Any, Dict, Iterator, List, Optional, Type, Mapping

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
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
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import (
    get_from_dict_or_env,
    get_pydantic_field_names,
)

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "hunyuan.tencentcloudapi.com"


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"Role": message.role, "Content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"Role": "user", "Content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"Role": "assistant", "Content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"Role": "system", "Content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")

    return message_dict


def _convert_dict_to_message(message) -> BaseMessage:
    role = message.Role
    content = message.Content
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    else:
        return ChatMessage(content=content, role=role)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("Role")
    content = _dict.get("Content") or ""

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


def _create_chat_result(response) -> ChatResult:
    generations = []
    for choice in response.Choices:
        message = _convert_dict_to_message(choice.Message)
        generations.append(ChatGeneration(message=message))

    token_usage = response.Usage
    llm_output = {
        "token_usage": {
            "prompt_tokens": token_usage.PromptTokens,
            "completion_tokens": token_usage.CompletionTokens,
            "total_tokens": token_usage.TotalTokens
        }
    }
    return ChatResult(generations=generations, llm_output=llm_output)


class MyChatHunyuan(BaseChatModel):
    """Tencent Hunyuan chat models API by Tencent.

    For more information, see https://cloud.tencent.com/document/product/1729
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "hunyuan_secret_id": "HUNYUAN_SECRET_ID",
            "hunyuan_secret_key": "HUNYUAN_SECRET_KEY",
        }

    @property
    def lc_serializable(self) -> bool:
        return True

    client: Any  #: :meta private:
    model_name: str = Field(default="hunyuan-lite", alias="model")
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for API call not explicitly specified."""

    hunyuan_api_base: str = Field(default=DEFAULT_API_BASE)
    """Hunyuan custom endpoints"""
    hunyuan_secret_id: Optional[str] = None
    """Hunyuan Secret ID"""
    hunyuan_secret_key: Optional[str] = None
    """Hunyuan Secret Key"""
    streaming: bool = False
    """Whether to stream the results or not."""
    temperature: float = 1.0
    """What sampling temperature to use."""
    top_p: float = 1.0
    """What probability mass to use."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["hunyuan_api_base"] = get_from_dict_or_env(
            values,
            "hunyuan_api_base",
            "HUNYUAN_API_BASE",
            DEFAULT_API_BASE,
        )
        values["hunyuan_secret_id"] = get_from_dict_or_env(
            values,
            "hunyuan_secret_id",
            "HUNYUAN_SECRET_ID",
        )
        values["hunyuan_secret_key"] = get_from_dict_or_env(
            values,
            "hunyuan_secret_key",
            "HUNYUAN_SECRET_KEY",
        )

        try:
            from tencentcloud.common import credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.common.profile.http_profile import HttpProfile
            from tencentcloud.hunyuan.v20230901 import hunyuan_client
        except ImportError:
            raise ImportError(
                "Could not import tencentcloud-sdk-python-hunyuan package. "
                "Please install it with `pip install tencentcloud-sdk-python-hunyuan`."
            )

        cred = credential.Credential(values["hunyuan_secret_id"], values["hunyuan_secret_key"])
        # 实例化一个http选项，可选的，没有特殊需求可以跳过
        httpProfile = HttpProfile()
        httpProfile.endpoint = values["hunyuan_api_base"]
        # 实例化一个client选项，可选的，没有特殊需求可以跳过
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        values["client"] = hunyuan_client.HunyuanClient(cred, "", clientProfile)

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Hunyuan API."""
        normal_params = {
            "Model": self.model_name,
            "Temperature": self.temperature,
            "TopP": self.top_p,
        }

        return {**normal_params, **self.model_kwargs}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        res = self._chat(messages, **kwargs)
        return _create_chat_result(res)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        kwargs.update({"Stream": True})
        res = self._chat(messages, **kwargs)

        default_chunk_class = AIMessageChunk
        for event in res:
            response = json.loads(event["data"])
            for choice in response["Choices"]:
                chunk = _convert_delta_to_message_chunk(
                    choice["Delta"], default_chunk_class
                )
                default_chunk_class = chunk.__class__
                cg_chunk = ChatGenerationChunk(message=chunk)
                if run_manager:
                    run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk)
                yield cg_chunk

    def _chat(self, messages: List[BaseMessage], **kwargs: Any):
        if self.hunyuan_secret_key is None:
            raise ValueError("Hunyuan secret key is not set.")

        from tencentcloud.hunyuan.v20230901 import models

        parameters = {**self._default_params, **kwargs}
        payload = {
            "Messages": [_convert_message_to_dict(m) for m in messages],
            **parameters,
        }

        req = models.ChatCompletionsRequest()
        req.from_json_string(json.dumps(payload))
        res = self.client.ChatCompletions(req)
        return res

    @property
    def _llm_type(self) -> str:
        return "hunyuan-chat"
