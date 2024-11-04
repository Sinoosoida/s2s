from dataclasses import dataclass, field

@dataclass
class ElevenLabsTTSHandlerArguments:
    elevenlabs_tts_api_key: str = field(
        default=None,
        metadata={
            "help": "API ключ для ElevenLabs TTS. Если не указан, будет использован переменная окружения 'ELEVENLABS_API_KEY'."
        },
    )
    elevenlabs_tts_proxy_url: str = field(
        default=None,
        metadata={
            "help": "URL прокси-сервера для запросов к ElevenLabs API. Если не указан, будет использован переменная окружения 'PROXY_URL'."
        },
    )
    elevenlabs_tts_voice: str = field(
        default="Rachel",
        metadata={
            "help": "Голосовая модель для TTS."
        },
    )
    elevenlabs_tts_model: str = field(
        default="eleven_turbo_v2_5",
        metadata={
            "help": "Модель TTS для генерации речи. По умолчанию 'eleven_turbo_v2_5'."
        },
    )
    elevenlabs_tts_use_proxy: str = field(
        default=True,
        metadata={
            "help": "Нужно ли использовать proxy при запуске"
        },
    )
