import logging
from baseHandler import BaseHandler
from iteratorsHandler import IteratorHandler
import numpy as np
from rich.console import Console
import time
import httpx
from elevenlabs.client import ElevenLabs
from utils.process_iterator import ProcessIterator
from utils.data import ImmutableDataChain
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
        use_proxy=True,
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


        if use_proxy:
            assert self.proxy_url is not None, "Proxy URL must be provided."
            # Создаем один экземпляр httpx.Client и сохраняем его в self.http_client
            limits = httpx.Limits(keepalive_expiry=60 * 60)
            self.http_client = httpx.Client(proxies=self.proxy_url, limits=limits, timeout=60*60)

            # Переиспользуем http_client для клиента ElevenLabs
            self.client = ElevenLabs(
                api_key=self.api_key,
                httpx_client=self.http_client,
            )
        else:
            self.client = ElevenLabs(
                api_key=self.api_key
            )

        self.warmup()

    def warmup(self):
        logger.info(f"Warmup {self.__class__.__name__}")
        try:
            self.client.models.get_all()
            logger.debug(f"Warmup {self.__class__.__name__} done")
        except Exception as e:
            logger.error(f"Warmup {self.__class__.__name__} failed, {e}")

    def process(self, input_data: ImmutableDataChain):

        llm_sentence = input_data.get("llm_sentence")
        language_code = input_data.get("language_code")

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        iterator = ProcessIterator()
        try:
            yield input_data.add_data(iterator, "output_audio_iterator")

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

                    if first_chunk:
                        logger.debug(f"First chunk received")
                        first_chunk = False

                    buffer += chunk
                    even_chunk = buffer[:(len(buffer) // 2) * 2]
                    audio_chunk = np.frombuffer(even_chunk, dtype='<i2')
                    iterator.put(audio_chunk)
                    buffer = buffer[(len(buffer) // 2) * 2:]

            logger.debug(f"All chunks received")
            iterator.close()
        except Exception as e:
            logger.error(f"Error in ElevenLabsTTSHandler: {e}")
            iterator.close()
            if self.should_listen is not None:
                self.should_listen.set()
            return

        if self.should_listen is not None:
            self.should_listen.set()

    def close(self):
        # Закрываем http_client при завершении работы
        self.http_client.close()
