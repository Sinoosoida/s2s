from dataclasses import dataclass, field


@dataclass
class ServerHandlerArguments:
    server_llm_model_name: str = field(
        default="gpt-4o-mini-2024-07-18",
        metadata={
            "help": "The pretrained language model to use. Default is 'gpt-4o-mini-2024-07-18'."
        },
    )
    server_user_role: str = field(
        default="user",
        metadata={
            "help": "Role assigned to the user in the chat context. Default is 'user'."
        },
    )
    server_init_chat_role: str = field(
        default="system",
        metadata={
            "help": "Initial role for setting up the chat context. Default is 'system'."
        },
    )
    server_init_chat_prompt: str = field(
        default="You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words.",
        metadata={
            "help": "The initial chat prompt to establish context for the language model. Default is 'You are a helpful AI assistant."
        },
    )

    server_uri: str = field(
        default="ws://205.172.57.158:8765",
        metadata={
            "help": "uri of USA server"
        },
    )

    server_chat_size: int = field(
        default=2,
        metadata={
            "help": "Number of interactions assitant-user to keep for the chat. None for no limitations."
        },
    )
    server_tts_voice: str = field(
        default=None,
        metadata={
            "help": "eleven labs tts voice"
        },
    )
    server_tts_model: str = field(
        default=None,
        metadata={
            "help": "eleven labs tts model"
        },
    )
    server_tts_optimize_streaming_latency: str = field(
        default=None,
        metadata={
            "help": "optimize_streaming_latency of tts"
        },
    )
    server_tts_output_format: bool = field(
        default="pcm_16000",
        metadata={
            "help": "Output audio format"
        },
    )