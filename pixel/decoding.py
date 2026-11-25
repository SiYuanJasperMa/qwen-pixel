"""Explicit numerical policies for native Qwen action generation."""
import os

import torch

from pixel.generation_cache import fresh_precision_cache, RecurrentPrecisionLayer

POLICIES = ('legacy', 'deterministic', 'fp32_state')


def configure_decoding(policy):
    if policy not in POLICIES:
        raise ValueError('Unknown decoding policy: ' + policy)
    if policy == 'legacy':
        return
    workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if not workspace:
        if torch.cuda.is_initialized():
            raise ValueError('Configure deterministic decoding before loading CUDA weights.')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    elif workspace not in (':4096:8', ':16:8'):
        raise ValueError('Unsupported deterministic cuBLAS workspace configuration.')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def decoding_identity(policy):
    return {'policy': policy,
            'torch_version': str(torch.__version__),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cublas_workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'recurrent_state': 'returned_kernel_dtype' if policy == 'fp32_state' else 'installed_cache_dtype'}


def decoding_implementation_identity():
    """Identify the loaded decoder, independently of mutable source files."""
    from pixel.eval_manifest import runtime_digest
    return {'loaded_policy_sha256':runtime_digest(DecodePolicyModel.generate),
            'loaded_cache_factory_sha256':runtime_digest(fresh_precision_cache),
            'loaded_cache_initialization_sha256':runtime_digest(RecurrentPrecisionLayer.lazy_initialization)}


class DecodePolicyModel:
    """Delegate weights and tool generation to the real actor with a fresh cache."""
    def __init__(self, actor, policy):
        if policy not in POLICIES:
            raise ValueError('Unknown decoding policy: ' + policy)
        self.actor = actor
        self.decode_policy = policy

    def __getattr__(self, name):
        return getattr(self.actor, name)

    def eval(self):
        self.actor.eval()
        return self

    def generate(self, **kwargs):
        if self.decode_policy != 'legacy' and not torch.are_deterministic_algorithms_enabled():
            raise ValueError('The requested deterministic actor policy is not configured.')
        if self.decode_policy == 'fp32_state' and kwargs.get('use_cache', True):
            if kwargs.get('past_key_values') is not None:
                raise ValueError('Each native action prefix requires a fresh recurrent cache.')
            kwargs = dict(kwargs)
            kwargs['past_key_values'] = fresh_precision_cache(self.actor.config)
        return self.actor.generate(**kwargs)
