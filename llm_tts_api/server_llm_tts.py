import logging
import time
import websocket
import json
import numpy as np
from rich.console import Console

from baseHandler import BaseHandler
from LLM.chat import Chat
from utils.data import ImmutableDataChain
from utils.constants import end_of_data, end_of_data_bytes

logger = logging.getLogger(__name__)

console = Console()


class LLMTTSAPI(BaseHandler):
    def setup(
            self,
            llm_model_name="gpt-4o-mini-2024-07-18",
            user_role="user",
            chat_size=10,
            init_chat_role="system",
            init_chat_prompt="You are a helpful AI assistant.",
            uri="ws://205.172.57.158:8765",
            tts_voice="alloy",
            tts_model="eleven_turbo_v2_5",
            tts_optimize_streaming_latency=3,
            tts_output_format="pcm_16000"
    ):
        self.llm_model_name = llm_model_name
        self.user_role = user_role
        self.uri = uri
        self.tts_voice = tts_voice
        self.tts_model = tts_model
        self.tts_optimize_streaming_latency = tts_optimize_streaming_latency
        self.tts_output_format = tts_output_format

        self.chat = Chat(chat_size)

        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial prompt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
            logger.debug(f"Prompt: {init_chat_prompt}")

        # Установка соединения с сервером
        self.ws = websocket.create_connection(self.uri)
        logger.info(f"Подключено к WebSocket серверу по адресу {self.uri}")

        self.warmup()

    def warmup(self):
        start = time.time()
        for _ in self.get_data(
                llm_model_name=self.llm_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Напиши историю из 3 коротких предложений."},
                ],
                language_code="ru",
                tts_voice=self.tts_voice,
                tts_model=self.tts_model,
                tts_optimize_streaming_latency=self.tts_optimize_streaming_latency,
                tts_output_format=self.tts_output_format,
                add_llm_messages_to_chat=False,
        ):
            pass
        end = time.time()
        logger.info(f"Разогрев завершен за {end - start:.2f} секунд.")

    def get_data(
            self,
            llm_model_name,
            messages,
            language_code,
            tts_voice,
            tts_model,
            tts_optimize_streaming_latency,
            tts_output_format,
            add_llm_messages_to_chat=True
    ):
        data_to_send = {
            "llm_model_name": llm_model_name,
            "messages": messages,
            "language_code": language_code,
            "tts_voice": tts_voice,
            "tts_model": tts_model,
            "tts_optimize_streaming_latency": tts_optimize_streaming_latency,
            "tts_output_format": tts_output_format,
        }

        message = json.dumps(data_to_send)

        # Отправка сообщения серверу
        self.ws.send(message)
        logger.info(f"Отправлено сообщение на сервер: {message}")

        concatenated_text = ""
        previous_text = ""

        while True:
            response = self.ws.recv()
            logger.info(f"Получено сообщение от сервера: {response}")

            # Проверяем, является ли сообщение сигналом окончания обработки
            if response == end_of_data_bytes:
                logger.info("Получен сигнал конца обработки от сервера.")
                break

            # Десериализация сообщения
            try:
                response_data = json.loads(response)
                # Если данные были сериализованы дважды, десериализуем еще раз
                if isinstance(response_data, str):
                    response_data = json.loads(response_data)
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка декодирования JSON: {e}")
                continue

            # Обработка текстовых данных
            llm_sentence = response_data.get("llm_sentence", "")
            if llm_sentence != previous_text:
                concatenated_text += llm_sentence
                previous_text = llm_sentence

            # Обработка аудиоданных
            output_audio_chunk = response_data.get("output_audio_chunk", [])
            output_audio_chunk = np.array(output_audio_chunk, dtype=np.int16)

            # Возвращаем аудиочанк
            yield output_audio_chunk

        # Добавляем накопленный текст в чат
        if add_llm_messages_to_chat and concatenated_text:
            self.chat.append({"role": "assistant", "content": concatenated_text})

    def process(self, data: ImmutableDataChain):
        logger.debug("Вызов языковой модели...")

        prompt = data.get("text")
        language_code = data.get("language_code")
        start_phrase = data.get("start_phrase")

        self.chat.append({"role": self.user_role, "content": prompt})

        if start_phrase:
            self.chat.append({"role": "assistant", "content": start_phrase})

        messages = self.chat.to_list()

        # Вызов get_data и возвращение аудиочанков
        for output_audio_chunk in self.get_data(
                llm_model_name=self.llm_model_name,
                messages=messages,
                language_code=language_code,
                tts_voice=self.tts_voice,
                tts_model=self.tts_model,
                tts_optimize_streaming_latency=self.tts_optimize_streaming_latency,
                tts_output_format=self.tts_output_format,
                add_llm_messages_to_chat=True,
        ):
            yield data.add_data(output_audio_chunk, "output_audio_chunk")

    def close(self):
        if self.ws:
            self.ws.close()
            logger.info("WebSocket соединение закрыто.")
