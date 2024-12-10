class ModelConfig:
    """Configuration constants for different AI models"""
    
    MODEL_SPECS = {
        # Azure OpenAI models
        "gpt-4o-mini": {
            "context_window": 128000,
            "pricing": {
                "input": 0.15,
                "output": 0.60
            },
            "rate_limit": {
                "tokens_per_minute": 60000,
                "requests_per_minute": 500
            },
            "family": "azure"
        },
        
        # Anthropic Claude models  
        # "claude-3-opus-20240229": {
        #     "context_window": 200000,
        #     "pricing": {
        #         "input": 15.0,
        #         "output": 75.0
        #     },
        #     "rate_limit": {
        #         "tokens_per_minute": 40000
        #     },
        #     "family": "claude-3"
        # },
        # "claude-3-5-sonnet-20241022": {
        #     "context_window": 200000,
        #     "pricing": {
        #         "input": 3.0,
        #         "output": 15.0
        #     },
        #     "rate_limit": {
        #         "tokens_per_minute": 80000
        #     },
        #     "family": "claude"
        # },
        # "claude-3-5-haiku-20241022": {
        #     "context_window": 200000,
        #     "pricing": {
        #         "input": 0.8,
        #         "output": 4.0
        #     },
        #     "rate_limit": {
        #         "tokens_per_minute": 100000
        #     },
        #     "family": "claude"
        # },
        
        
        # Google models
        "gemini-1.5-pro": {
            "context_window": 128000,
            "pricing": {
                "input": 1.25,
                "output": 5.0,
                "cache": 0.3125
            },
            "rate_limit": {
                "tokens_per_minute": 128000,
                "requests_per_minute": 60
            },
            "family": "gemini"
        },
        
        "gemini-1.5-flash-002": {
            "context_window": 1000000,
            "pricing": {
                "input": 0.0375,
                "output": 0.15
            },
            "rate_limit": {
                "tokens_per_minute": 100000,
                "requests_per_minute": 60
            },
            "family": "gemini"
        },
        
        # Amazon Bedrock models
        "amazon.nova-micro-v1:0": {
            "context_window": 128000,
            "pricing": {
                "input": 0.035,
                "output": 0.14
            },
            "rate_limit": {
                "tokens_per_minute": 50000,
                "requests_per_minute": 100
            },
            "family": "aws"
        },
        "amazon.nova-lite-v1:0": {
            "context_window": 300000,
            "pricing": {
                "input": 0.06,
                "output": 0.24
            },
            "rate_limit": {
                "tokens_per_minute": 50000,
                "requests_per_minute": 100
            },
            "family": "aws"
        },
        "amazon.nova-pro-v1:0": {
            "context_window": 300000,
            "pricing": {
                "input": 0.8,
                "output": 3.2
            },
            "rate_limit": {
                "tokens_per_minute": 50000,
                "requests_per_minute": 100
            },
            "family": "aws"
        },
        
        # Adding Anthropic models through AWS Bedrock
        # "arn:aws:bedrock:us-east-1:502675305491:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0": {
        #     "context_window": 200000,
        #     "pricing": {
        #         "input": 0.25,  # $0.00025 * 1000
        #         "output": 1.25  # $0.00125 * 1000
        #     },
        #     "rate_limit": {
        #         "tokens_per_minute": 100000,
        #         "requests_per_minute": 100
        #     },
        #     "family": "claude"
        # },
        "arn:aws:bedrock:us-east-1:502675305491:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0": {
            "context_window": 200000,
            "pricing": {
                "input": 3.0,  # $0.003 * 1000
                "output": 15.0  # $0.015 * 1000
            },
            "rate_limit": {
                "tokens_per_minute": 40000,
                "requests_per_minute": 100
            },
            "family": "claude"
        },

        # Groq models
        "llama-3.1-8b-instant": {
            "context_window": 128000,
            "pricing": {
                "input": 0.05,
                "output": 0.08
            },
            "rate_limit": {
                "requests_per_minute": 30,
                "tokens_per_minute": 20000
            },
            "family": "groq"
        },

        # Cohere
        # "command-nightly": {
        #     "context_window": 128000,
        #     "pricing": {
        #         "input": 0.001,
        #         "output": 0.002
        #     },
        #     "rate_limit": {
        #         "requests_per_minute": 10,
        #         "tokens_per_minute": 128000
        #     },
        #     "family": "cohere"
        # },
        # "command-r": {
        #     "context_window": 128000,
        #     "pricing": {
        #         "input": 0.15,
        #         "output": 0.60
        #     },
        #     "rate_limit": {
        #         "requests_per_minute": 10,
        #         "tokens_per_minute": 128000
        #     },
        #     "family": "cohere"
        # },
        "command-r-plus": {
            "context_window": 128000,
            "pricing": {
                "input": 2.50,
                "output": 10.00
            },
            "rate_limit": {
                "requests_per_minute": 10,
                "tokens_per_minute": 128000
            },
            "family": "cohere"
        }
    }

    @classmethod
    def get_models_by_family(cls, family: str) -> list[str]:
        """Get all model names belonging to a specific family"""
        return [
            model_name for model_name, specs in cls.MODEL_SPECS.items()
            if specs.get("family") == family
        ]

    @classmethod
    def get_all_families(cls) -> list[str]:
        """Get list of all unique model families"""
        return list(set(
            specs.get("family") for specs in cls.MODEL_SPECS.values()
            if specs.get("family")
        )) 