from functools import partial
from typing import Optional, List, Any

from langchain_community.llms import GPT4All
from langchain_core.callbacks import CallbackManagerForLLMRun

from langchain_community.llms.utils import enforce_stop_tokens


class MyGPT4All(GPT4All):
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)
        text = ""
        params = {**self._default_params(), **kwargs}
        with self.client.chat_session():    # 新增行，解决输出异常
            for token in self.client.generate(prompt, **params):
                if text_callback:
                    text_callback(token)
                text += token
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text
