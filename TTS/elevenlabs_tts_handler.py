import logging
from baseHandler import BaseHandler
import numpy as np
from rich.console import Console
import time
import httpx
from elevenlabs.client import ElevenLabs
import os

logger = logging.getLogger(__name__)
console = Console()

class ElevenLabsTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        api_key=None,
        proxy_url=None,
        voice=None,
        model="eleven_turbo_v2_5",
        gen_kwargs={},  # Not used
    ):
        self.should_listen = should_listen
        self.voice = voice
        self.model = model

        if api_key is None:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if api_key is None:
                raise ValueError("ElevenLabs API key must be provided or set in the ELEVENLABS_API_KEY environment variable.")
        self.api_key = api_key

        if proxy_url is None:
            proxy_url = os.getenv("PROXY_URL")
        self.proxy_url = proxy_url

        assert proxy_url


        self.client = ElevenLabs(
            api_key=self.api_key,
            httpx_client=httpx.Client(proxy=self.proxy_url),
        )

        self.warmup()

    def warmup(self):
        logger.info(f"Warmup {self.__class__.__name__}")
        try:
            self.client.models.get_all()
            logger.debug(f"Warmup {self.__class__.__name__} done")
        except Exception as e:
            logger.error(f"Warmup {self.__class__.__name__} failed, {e}")

    def process(self, llm_sentence):

        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        try:
            audio = self.client.generate(
                voice=self.voice,
                text=llm_sentence,
                model=self.model,
                stream=True,
                optimize_streaming_latency=3,
                output_format="pcm_16000",
            )
            buffer = b""
            first_chunk = True
            for chunk in audio:
                if chunk:

                    if first_chunk:  # Добавлено: вывод информации о первом чанке
                        logger.debug(f"First chunk received")
                        first_chunk = False

                    buffer += chunk
                    even_chunk = buffer[:(len(buffer) // 2) * 2]
                    audio_chunk = np.frombuffer(even_chunk, dtype='<i2')  # 16-битные целые числа, little-endian
                    yield audio_chunk
                    buffer = buffer[(len(buffer) // 2) * 2:]

            logger.debug(f"All chunck recived")
        except Exception as e:
            logger.error(f"Error in ElevenLabsTTSHandler: {e}")
            if self.should_listen is not None:
                self.should_listen.set()
            return

        if self.should_listen is not None:
            self.should_listen.set()
