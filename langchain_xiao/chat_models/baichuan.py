from typing import List, Optional, Any, Iterator

from langchain_community.chat_models.baichuan import ChatBaichuan
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk


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
