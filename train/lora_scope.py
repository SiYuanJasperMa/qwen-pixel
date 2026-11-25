"""Explicit Qwen text-layer adapter scopes, keeping the visual encoder fixed."""


def target_pattern(scope):
    modules={
        'hybrid':'q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj|gate_proj|up_proj|down_proj',
        'standard':'q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj',
        # Adapt context routing in all32 attention layers while preserving
        # recurrent decay/output gates and the pretrained MLP transformations.
        'attention':'q_proj|k_proj|v_proj|o_proj|in_proj_qkv|out_proj',
    }
    if scope not in modules:raise ValueError('Unknown LoRA scope: '+scope)
    return r'.*language_model.*\.('+modules[scope]+')'
