import logging
logger = logging.getLogger(__name__)

from utils.constants import end_of_data, end_of_data_bytes  # Импортируем специальное значение
from utils.data import ImmutableDataChain
import traceback
import asyncio
import websockets
import logging
import json

logger = logging.getLogger(__name__)  # Определяем логгер

class WebSocketHandler:
    def __init__(self, stop_event, queue_in, queue_out, host='0.0.0.0', port=8765):
        self.stop_event = stop_event  # threading.Event()
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.host = host
        self.port = port
        self.server = None
        self.loop = None
        self.websocket = None  # Единственный клиент
        self.client_connected_event = asyncio.Event()  # Событие для отслеживания подключения клиента

    def run(self):
        # Создаем новый event loop для этого потока
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Запускаем вебсокет-сервер
        start_server = websockets.serve(self.handler, self.host, self.port)
        self.server = self.loop.run_until_complete(start_server)
        logger.info(f"WebSocket сервер запущен на ws://{self.host}:{self.port}")

        # Запускаем задачу для отправки данных из выходной очереди клиенту
        send_task = asyncio.ensure_future(self.send_output_to_client(), loop=self.loop)

        # Запускаем задачу для мониторинга stop_event
        stop_task = asyncio.ensure_future(self.check_stop_event(), loop=self.loop)

        try:
            # Запускаем event loop
            self.loop.run_forever()
        finally:
            # Останавливаем сервер и закрываем соединение
            send_task.cancel()
            stop_task.cancel()
            self.server.close()
            self.loop.run_until_complete(self.server.wait_closed())
            if self.websocket is not None:
                self.loop.run_until_complete(self.websocket.close())
            self.loop.close()
            logger.info("WebSocket сервер остановлен")

    async def check_stop_event(self):
        # Ожидаем, пока threading.Event будет установлен, используя run_in_executor
        await self.loop.run_in_executor(None, self.stop_event.wait)
        # Останавливаем event loop
        self.loop.stop()

    async def handler(self, websocket, path):
        # Сохраняем websocket клиента
        self.websocket = websocket
        logger.info(f"Клиент подключился: {websocket.remote_address}")
        # Устанавливаем событие подключения клиента
        self.client_connected_event.set()

        try:
            while not self.stop_event.is_set():
                # Ожидаем сообщения от клиента
                message = await websocket.recv()
                logger.debug(f"Получено сообщение от клиента {websocket.remote_address}: {message}")

                try:
                    # Десериализация сообщения из JSON в dict
                    message_dict = json.loads(message)
                    assert isinstance(message_dict, dict)
                    logger.debug(f"Десериализованное сообщение: {message_dict}")
                    message_dict = ImmutableDataChain.from_dict(message_dict)
                    # Помещаем сообщение во входную очередь
                    self.queue_in.put(message_dict)
                except Exception as e:
                    logger.error(f"Ошибка при обработке сообщения: {e}")
                    traceback_str = ''.join(traceback.format_tb(e.__traceback__))
                    logger.error(f"Трассировка исключения:\n{traceback_str}")
                    # Отправляем сообщение об ошибке клиенту
                    error_message = json.dumps({'error': str(e)})
                    await websocket.send(error_message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Клиент отключился: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Неожиданная ошибка в обработчике: {e}")
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            logger.error(f"Трассировка исключения:\n{traceback_str}")
        finally:
            # Очищаем информацию о клиенте
            if self.websocket == websocket:
                self.websocket = None
                # Сбрасываем событие подключения клиента
                self.client_connected_event.clear()

    async def send_output_to_client(self):
        buffer = []  # Буфер для хранения сообщений, когда клиент не подключен
        while not self.stop_event.is_set():
            try:
                # Получаем данные из выходной очереди
                output_item = await self.loop.run_in_executor(None, self.queue_out.get)
                sent = output_item.get("llm_sentence")
                logger.debug(f"sent {sent}")
                if output_item.get("llm_sentence") == end_of_data:
                    logger.debug(f"Получен последний элемент")
                    buffer.append(end_of_data_bytes)
                else:
                    assert isinstance(output_item, ImmutableDataChain)
                    llm_sentence = output_item.get("llm_sentence")
                    output_audio_chunk = output_item.get("output_audio_chunk").tolist()
                    logger.debug(f"Получен элемент из выходной очереди: {llm_sentence}")
                    # logger.debug(f"Получено аудио: {output_audio_chunk}")
                    output_item = json.dumps({"llm_sentence": llm_sentence, "output_audio_chunk": output_audio_chunk})
                    output_data = json.dumps(output_item)
                    buffer.append(output_data)

                # Пытаемся отправить сообщения, если клиент подключен
                while buffer:
                    if self.websocket and self.websocket.open:
                        msg = buffer.pop(0)
                        try:
                            await self.websocket.send(msg)
                            logger.debug(f"Отправлено сообщение клиенту: {msg}")
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("Соединение с клиентом потеряно при отправке сообщения")
                            self.websocket = None
                            self.client_connected_event.clear()
                            buffer.insert(0, msg)  # Возвращаем сообщение в буфер
                            break
                    else:
                        # Ожидаем подключения клиента
                        logger.debug("Клиент не подключен, ожидаем подключения")
                        await self.client_connected_event.wait()
                        await asyncio.sleep(0.1)  # Небольшая пауза, чтобы избежать быстрого цикла
            except Exception as e:
                logger.error(f"Ошибка при отправке данных клиенту: {e}")
                traceback_str = ''.join(traceback.format_tb(e.__traceback__))
                logger.error(f"Трассировка исключения:\n{traceback_str}")
                continue
