"""Use Cosmos3-Edge's native non-thinking template consistently for JSON tasks.

Approved answer text is unchanged. Empty thinking delimiters are template
control tokens, excluded by the upstream assistant loss mask.
"""
import copy

CHAT_MODE = 'edge-non-thinking-empty-prefix-v1'


def with_empty_thinking(conversation):
    messages = copy.deepcopy(conversation)
    for message in messages:
        if message.get('role') != 'assistant': continue
        content = message['content']
        if isinstance(content, str):
            if not content.startswith('<think></think>'): message['content']='<think></think>'+content
        else:
            first = next((v for v in content if v.get('type')=='text'), None)
            if first is not None and not first['text'].startswith('<think></think>'):
                first['text']='<think></think>'+first['text']
    return messages


def install():
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from cosmos_framework.data.generator.processors.cosmos3_edge_processing import NemotronNanoV3BridgeProcessor
    if getattr(NemotronNanoV3BridgeProcessor, '_tancho_chat_mode', None) == CHAT_MODE: return
    original_processor = NemotronNanoV3BridgeProcessor.apply_chat_template
    original_tokenizer = PreTrainedTokenizerBase.apply_chat_template
    def processor_chat(self, conversation, **kwargs):
        kwargs['enable_thinking'] = False
        return original_processor(self, with_empty_thinking(conversation), **kwargs)
    def tokenizer_chat(self, conversation, *args, **kwargs):
        kwargs['enable_thinking'] = False
        return original_tokenizer(self, conversation, *args, **kwargs)
    NemotronNanoV3BridgeProcessor.apply_chat_template = processor_chat
    PreTrainedTokenizerBase.apply_chat_template = tokenizer_chat
    NemotronNanoV3BridgeProcessor._tancho_chat_mode = CHAT_MODE
