from typing import Any, Dict

from langchain_community.chat_models import ChatTongyi
from langchain_community.chat_models.tongyi import convert_dict_to_message


class MyChatTongyi(ChatTongyi):
    """将token_usage数据结构改为openai的形式"""

    @staticmethod
    def _chat_generation_from_qwen_resp(
        resp: Any, is_chunk: bool = False, is_last_chunk: bool = True
    ) -> Dict[str, Any]:
        choice = resp["output"]["choices"][0]
        message = convert_dict_to_message(choice["message"], is_chunk=is_chunk)
        if is_last_chunk:
            token_usage = {
                "prompt_tokens": resp["usage"]["input_tokens"],
                "completion_tokens": resp["usage"]["output_tokens"],
                "total_tokens": resp["usage"]["total_tokens"]
            }
            return dict(
                message=message,
                generation_info=dict(
                    finish_reason=choice["finish_reason"],
                    request_id=resp["request_id"],
                    token_usage=token_usage,
                ),
            )
        else:
            return dict(message=message)
