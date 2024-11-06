import socket
from rich.console import Console
import logging
import time
from queue import Empty
from utils.data import ImmutableDataChain

logger = logging.getLogger(__name__)

console = Console()


class SocketSender:
    """
    Handles sending generated audio packets to the clients.
    """

    def __init__(self, stop_event, queue_in, host="0.0.0.0", port=12346, sample_rate=16000, bytes_per_sample=2, buffer_time=0.2):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self.bytes_per_sample = bytes_per_sample
        self.buffer_time = buffer_time

    # def run(self):
    #     self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #     self.socket.bind((self.host, self.port))
    #     self.socket.listen(1)
    #     logger.info("Sender waiting to be connected...")
    #     self.conn, _ = self.socket.accept()
    #     logger.info("sender connected")
    #
    #     start_time = time.time()
    #     seconds_of_users_audio = 0
    #
    #     while not self.stop_event.is_set():
    #         audio_chunk = self.queue_in.get().get_data()
    #         chunk_duration = len(audio_chunk) / self.sample_rate # * self.bytes_per_sample) #idk why it is not
    #
    #         #(time.time() - start_time) - время, которое прошло с начала передачи непрерывного аудио
    #         #seconds_of_users_audio - длительность аудио, которое суммарно было передано с момента start_time
    #         #self.buffer_time - временная фора, с которой чанки аудио присылаются пользователю
    #         if (time.time() - start_time) < (seconds_of_users_audio - self.buffer_time):
    #             self.stop_event.wait(timeout=(seconds_of_users_audio - self.buffer_time)-(time.time() - start_time))
    #
    #         #выполняется в случае, если новый чанк для отправки получен слишком поздно
    #         if time.time() - start_time > seconds_of_users_audio:
    #             seconds_of_users_audio = 0
    #             start_time = time.time()
    #
    #         if isinstance(audio_chunk, bytes) and audio_chunk == b"END":
    #             break
    #
    #         seconds_of_users_audio += chunk_duration
    #         self.conn.sendall(audio_chunk)
    #
    #     self.conn.close()
    #     logger.info("Sender closed")
    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info("Sender waiting to be connected...")

        while not self.stop_event.is_set():
            self.conn, _ = self.socket.accept()
            logger.info("Sender connected")

            start_time = time.time()
            seconds_of_users_audio = 0

            try:
                while not self.stop_event.is_set():
                    audio_chunk = self.queue_in.get().get_data()
                    chunk_duration = len(audio_chunk) / self.sample_rate

                    if (time.time() - start_time) < (seconds_of_users_audio - self.buffer_time):
                        self.stop_event.wait(
                            timeout=(seconds_of_users_audio - self.buffer_time) - (time.time() - start_time))

                    if time.time() - start_time > seconds_of_users_audio:
                        seconds_of_users_audio = 0
                        start_time = time.time()

                    if isinstance(audio_chunk, bytes) and audio_chunk == b"END":
                        break

                    seconds_of_users_audio += chunk_duration
                    self.conn.sendall(audio_chunk)

            except (ConnectionResetError, BrokenPipeError):
                logger.info("Client disconnected unexpectedly, waiting for reconnection...")

            finally:
                self.conn.close()

        self.socket.close()
        logger.info("Sender closed")
