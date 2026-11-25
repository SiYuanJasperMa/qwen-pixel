"""Fresh Qwen hybrid caches that retain the recurrent kernel's state precision.

The installed Transformers layer initializes recurrent state using the
convolution cache dtype. That rounds float32 GDN state to bfloat16 at every
decode step even when the recurrent kernel and training use float32 state.
This local cache preserves the returned state dtype without changing weights,
the model architecture, or the third-party Transformers installation.
"""
import torch
from transformers.cache_utils import DynamicCache, LinearAttentionLayer


class RecurrentPrecisionLayer(LinearAttentionLayer):
    def lazy_initialization(self, conv_states=None, recurrent_states=None):
        if conv_states is not None:
            super().lazy_initialization(conv_states=conv_states)
        if recurrent_states is not None:
            self.recurrent_states = torch.zeros_like(recurrent_states)
            if not torch._dynamo.is_compiling():
                torch._dynamo.mark_static_address(self.recurrent_states)
            self.is_recurrent_states_initialized = True


def fresh_precision_cache(config):
    cache = DynamicCache(config=config.get_text_config(decoder=True))
    replaced = 0
    for index, layer in enumerate(cache.layers):
        if type(layer) is LinearAttentionLayer:
            cache.layers[index] = RecurrentPrecisionLayer()
            replaced += 1
        elif isinstance(layer, LinearAttentionLayer):
            raise ValueError('Unsupported composite linear cache; do not silently replace its attention state.')
    if not replaced:
        raise ValueError('No linear-attention cache layers found for the Qwen precision probe.')
    return cache
