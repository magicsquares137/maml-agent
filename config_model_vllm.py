MODEL_INFOS = [
    {
        "model_name": "qwen-ppo-iter3-merged",
        "client_name": "openai",
        "model_id": "models/qwen-ppo-iter3-merged",  # Path to your merged model
        "model_kwargs": {
            "api_type": "chat_completions",
            "temperature": 0.7,  # Match your PPO training config
            "top_p": 0.8,        # Match your PPO training config
            "top_k": 20,         # Match your PPO training config
            "seed": 100,
            "api_key_env_name": "NO_API_KEY",
            "base_url": "http://localhost:8000/v1",  # Local vLLM server
            "max_completion_tokens": 3000,
            "tool_parser_name": None,
            "parallel_tool_calls": False,
            "cost_per_token": {
                "input_cache_miss": 0.0,
                "input_cache_hit": 0.0,
                "input_cache_write": 0.0,
                "output": 0.0,
            },
        },
        "function_calling": False,
        "tool_choice": "none",
        "function_calling_demos": False,
        "remove_function_property_keys": [
            "exclusiveMinimum",
            "exclusiveMaximum",
            "minimum",
            "maximum",
        ],
        "model_server_config": {
            "enabled": False,
        },
        "part_of": ["ppo", "qwen", "vllm"],
        "provider": "vllm",
    },
]