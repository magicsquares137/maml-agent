MODEL_INFOS = [
    {
        "model_name": "deepseek-coder-33b-instruct-vllm",
        "client_name": "openai",
        "model_id": "deepseek-ai/deepseek-coder-33b-instruct",
        "model_kwargs": {
            "api_type": "chat_completions",
            # Paper greedy decoding:
            "temperature": 0,
            "top_p": 1.0,
            "seed": 100,
            "api_key_env_name": "NO_API_KEY",
            "base_url": "https://get7p2dg7934vi-8000.proxy.runpod.net/v1",

            # keep reasonable so runs don't hang forever
            "max_completion_tokens": 3000,

            # For DeepSeek, unless you *know* it supports tool calling well,
            # it's usually safer to disable tool parsing first.
            "tool_parser_name": None,
            "parallel_tool_calls": False,

            "cost_per_token": {
                "input_cache_miss": 0.0,
                "input_cache_hit": 0.0,
                "input_cache_write": 0.0,
                "output": 0.0,
            },
        },

        # Start conservative: no function-calling / tool-choice automation
        # until you confirm the baseline runs end-to-end.
        "function_calling": False,
        "tool_choice": "none",
        "function_calling_demos": False,

        # If you later enable function calling and hit schema issues with vLLM,
        # you can copy their remove_function_property_keys list here.
        "remove_function_property_keys": [
            "exclusiveMinimum",
            "exclusiveMaximum",
            "minimum",
            "maximum",
        ],

        "model_server_config": {
            "enabled": True,
            "command": (
                "vllm serve deepseek-ai/deepseek-coder-33b-instruct "
                "--port {port} "
                "--max-model-len 8192 "
                "--gpu-memory-utilization 0.90 "
                "--max-num-seqs 4"
            ),
            "timeout": 600,
            "show_logs": False,
        },

        "part_of": ["vn", "vllm"],
        "provider": "vllm",
    },
]
