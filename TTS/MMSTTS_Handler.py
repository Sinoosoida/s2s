from transformers import VitsModel, AutoTokenizer
import logging
from baseHandler import BaseHandler
import numpy as np
from rich.console import Console
import torch
import re
from num2words import num2words
from transliterate import translit

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


class MMSTTSHandler(BaseHandler):
    def setup(
            self,
            should_listen,
            device="cuda",
            gen_kwargs={},
            model_name="facebook/mms-tts-rus",
    ):
        self.should_listen = should_listen
        self.device = device
        self.model_name = model_name
        self.model = VitsModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        text = "Тест"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model(**inputs).waveform

    def process(self, llm_sentence):

        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        text = llm_sentence
        text = re.sub(r'\d+', lambda x: num2words(int(x.group()), lang='ru'), text)
        text = translit(text, 'ru')

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).waveform

        audio = output.cpu().numpy()[0]

        if len(audio) == 0:
            self.should_listen.set()
            return

        # Масштабируем аудио и преобразуем в int16
        audio = (audio * 32768).astype(np.int16)

        # Выдаем аудио целиком
        yield audio

        self.should_listen.set()
