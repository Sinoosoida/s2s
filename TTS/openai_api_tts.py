import logging
from baseHandler import BaseHandler
import numpy as np
from rich.console import Console
import torch
import time
import httpx
from openai import OpenAI
import os
import librosa

logger = logging.getLogger(__name__)
console = Console()


class OpenAITTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        api_key=None,
        proxy_url=None,
        voice="alloy",
        gen_kwargs={},  # Не используется
    ):
        self.should_listen = should_listen
        self.voice = voice

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OpenAI API key must be provided or set in the OPENAI_API_KEY environment variable.")
        self.api_key = api_key

        if proxy_url is None:
            proxy_url = os.getenv("PROXY_URL")
        self.proxy_url = proxy_url

        assert proxy_url

        # Настройка клиента OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            http_client=httpx.Client(proxy=self.proxy_url)
        )

        self.warmup()

    def warmup(self):
        logger.info(f"Разогрев {self.__class__.__name__}")
        # Разогрев не требуется для OpenAI API

    def process(self, llm_sentence):
        # Обработка возможного кода языка
        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        try:
            start_time = time.time()
            with self.client.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice=self.voice,
                    response_format="pcm",  # PCM формат без заголовка
                    input=llm_sentence,
            ) as response:
                for chunk in response.iter_bytes(2400):
                    # Преобразование байтов в numpy массив
                    audio_chunk = np.frombuffer(chunk, dtype='<i2')
                    print(audio_chunk.max())

                    # Ресемплирование с учетом исходной частоты дискретизации 48000 Гц
                    original_sr = 24000  # Исходная частота дискретизации
                    target_sr = 16000    # Целевая частота дискретизации для вашего пайплайна

                    audio_chunk = librosa.resample(
                        audio_chunk.astype(np.float32),
                        orig_sr=original_sr,
                        target_sr=target_sr
                    )

                    audio_chunk = audio_chunk
                    audio_chunk = audio_chunk.astype(np.int16)
                    yield audio_chunk
                    start_time = time.time()
        except Exception as e:
            logger.error(f"Ошибка в OpenAITTSHandler: {e}")
            self.should_listen.set()
            return

        self.should_listen.set()
