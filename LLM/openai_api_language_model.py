import logging
import time
import httpx

from nltk import sent_tokenize
from rich.console import Console
from openai import OpenAI

from baseHandler import BaseHandler
from LLM.chat import Chat
import os
from utils.data import ImmutableDataChain
logger = logging.getLogger(__name__)

console = Console()

class OpenApiModelHandler(BaseHandler):
    """
    Handles the language model part.
    """
    def setup(
        self,
        model_name="deepseek-chat",
        device="cuda",
        gen_kwargs={},
        base_url=None,
        api_key=None,
        stream=False,
        user_role="user",
        chat_size=1,
        init_chat_role="system",
        init_chat_prompt="You are a helpful AI assistant.",
        proxy_url=None
    ):
        self.model_name = model_name
        self.stream = stream
        self.chat = Chat(chat_size)

        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial prompt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
            logger.debug(f"Prompt: {init_chat_prompt}")

        self.user_role = user_role

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OpenAI API key must be provided or set in the OPENAI_API_KEY environment variable.")

        if proxy_url is None:
            proxy_url = os.getenv("PROXY_URL")

        # Создаем один экземпляр httpx.Client и переиспользуем его
        limits = httpx.Limits(keepalive_expiry = 60*60)
        if proxy_url is not None:
            self.http_client = httpx.Client(proxies=proxy_url, limits = limits, timeout=60*60)
        else:
            raise ConnectionError("No proxy")
            self.http_client = httpx.Client(limits = limits, timeout=60*60)

        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=self.http_client)
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

        prompt = data.get("text")
        language_code = data.get("language_code")
        start_phrase = data.get("start_phrase")

        logger.debug(f"prompt is {prompt}")
        logger.debug(f"language_code is {language_code}")
        logger.debug(f"start_phrase is {start_phrase}")

        self.chat.append({"role": self.user_role, "content": prompt})

        # Add the start_phrase to the assistant's role to guide the model
        if start_phrase:
            self.chat.append({"role": "assistant", "content": start_phrase})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.chat.to_list(),
            stream=self.stream
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
        else:
            generated_text = response.choices[0].message.content
            self.chat.append({"role": "assistant", "content": generated_text})
            yield data.add_data(generated_text, "llm_sentence")
