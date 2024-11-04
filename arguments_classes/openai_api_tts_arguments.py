from dataclasses import dataclass, field

@dataclass
class OpenAITTSHandlerArguments:
    openai_tts_api_key: str = field(
        default=None,
        metadata={
            "help": "OpenAI API ключ. Если не указан, будет использован переменная окружения 'OPENAI_API_KEY'."
        },
    )
    openai_tts_proxy_url: str = field(
        default=None,
        metadata={
            "help": "URL прокси-сервера для запросов к OpenAI API. Если не указан, будет использован переменная окружения 'PROXY_URL'."
        },
    )
    openai_tts_voice: str = field(
        default="alloy",
        metadata={
            "help": "Голосовая модель для TTS. По умолчанию 'alloy'."
        },
    )