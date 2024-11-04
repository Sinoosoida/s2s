import logging
import time
import httpx

from nltk import sent_tokenize
from rich.console import Console
from openai import OpenAI

from baseHandler import BaseHandler
from utils.constants import end_of_data
import os
from utils.data import ImmutableDataChain
logger = logging.getLogger(__name__)

console = Console()

class OpenApiModelServerHandler(BaseHandler):
    """
    Handles the language model part.
    """
    def setup(
        self,
    ):

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key must be provided or set in the OPENAI_API_KEY environment variable.")



        # self.http_client = httpx.Client(proxies=proxy_url)
        self.http_client = httpx.Client()

        self.client = OpenAI(api_key=api_key, http_client=self.http_client)
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            stream=self.stream
        )
        end = time.time()
        logger.info(
            f"{self.__class__.__name__}: warmed up! time: {(end - start):.3f} s"
        )

    def process(self, data: ImmutableDataChain):
        logger.debug("call api language model...")

        model_name = data.get("llm_model_name")
        messages = data.get("messages")

        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True
        )

        first_chunk = True
        first_sentence = True
        if self.stream:
            generated_text, printable_text = "", ""
            for chunk in response:
                if first_chunk:
                    logger.debug(f"First chunk received")
                    first_chunk = False
                new_text = chunk.choices[0].delta.content or ""
                generated_text += new_text
                printable_text += new_text
                sentences = sent_tokenize(printable_text)
                if len(sentences) > 1:
                    if first_sentence:
                        logger.debug(f"First sentence received")
                        first_sentence = False
                    yield data.add_data(sentences[0], "llm_sentence")
                    printable_text = new_text

            logger.debug(f"All chunks received")
            self.chat.append({"role": "assistant", "content": generated_text})
            # don't forget last sentence
            yield data.add_data(printable_text, "llm_sentence")
            yield data.add_data(end_of_data, "llm_sentence")
        else:
            generated_text = response.choices[0].message.content
            self.chat.append({"role": "assistant", "content": generated_text})
            yield data.add_data(generated_text, "llm_sentence")
