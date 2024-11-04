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
from utils.constants import end_of_data

logger = logging.getLogger(__name__)
console = Console()

class ElevenLabsTTSServerHandler(BaseHandler):
    def setup(
        self,
    ):
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if api_key is None:
            raise ValueError("ElevenLabs API key must be provided or set in the ELEVENLABS_API_KEY environment variable.")
        self.api_key = api_key

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

        if input_data == end_of_data or input_data.get_data() == end_of_data:
            yield input_data
            return

        llm_sentence = input_data.get("llm_sentence")
        language_code = input_data.get("language_code")
        tts_voice = input_data.get("tts_voice")
        tts_model = input_data.get("tts_model")
        tts_optimize_streaming_latency = input_data.get("tts_optimize_streaming_latency")
        tts_output_format = input_data.get("tts_output_format")

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        try:
            audio = self.client.generate(
                voice=tts_voice,
                text=llm_sentence,
                model=tts_model,
                stream=True,
                optimize_streaming_latency=tts_optimize_streaming_latency,
                output_format=tts_output_format,
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
                    yield input_data.add_data(audio_chunk, "llm_sentence")
                    buffer = buffer[(len(buffer) // 2) * 2:]

            logger.debug(f"All chunks received")
        except Exception as e:
            logger.error(f"Error in ElevenLabsTTSHandler: {e}")

    def close(self):
        # Закрываем http_client при завершении работы
        self.http_client.close()
