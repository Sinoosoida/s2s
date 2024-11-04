from os import write
from time import perf_counter
import logging
import threading
from queue import Queue
from collections import deque  # Added for efficient buffer management
import concurrent.futures

logger = logging.getLogger(__name__)

class BaseHandler:
    """
    Base class for pipeline parts. Each part of the pipeline has an input and an output queue.
    The `setup` method along with `setup_args` and `setup_kwargs` can be used to address the specific requirements of the implemented pipeline part.
    To stop a handler properly, set the stop_event and, to avoid queue deadlocks, place b"END" in the input queue.
    Objects placed in the input queue will be processed by the `process` method, and the yielded results will be placed in the output queue.
    The cleanup method handles stopping the handler, and b"END" is placed in the output queue.
    """

    def __init__(self, stop_event, queue_in, queue_out, threads=1, setup_args=(), setup_kwargs={}):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.setup(*setup_args, **setup_kwargs)
        self._times = []
        self.threads = threads

        self.writer_id_counter = 0                                  # Counter for assigning sequence numbers to requests
        self.next_write_sequence = 0                                # Next sequence number that should write to the output queue

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads)
        self.condition = threading.Condition()

    def setup(self, *args, **kwargs):
        pass

    def process(self, input_data):
        raise NotImplementedError

    def run(self):
        while not self.stop_event.is_set():
            input_data = self.queue_in.get()

            if isinstance(input_data, bytes) and input_data == b"END":
                logger.debug("Stopping thread")
                break

            writer_id = self.writer_id_counter #Writer
            self.writer_id_counter += 1
            self.executor.submit(self.process_and_write, input_data, writer_id)

        self.executor.shutdown(wait=True)
        self.cleanup()
        self.queue_out.put(b"END")

    def process_and_write(self, input_data, writer_id):

        logger.debug(
            f"{self.__class__.__name__} [{writer_id}]: Started")

        start_time = perf_counter()                 # Время начала обработки
        first_chunk_time = None                     # Время получения первого чанка
        last_chunk_time = None                      # Время получения последнего чанка
        buffer = deque()  # Internal buffer for storing chunks

        try:

            #Пока получаем данные
            for chunk in self.process(input_data):

                current_time = perf_counter()

                if first_chunk_time is None:
                    first_chunk_time = current_time
                    total_time_first_chunk = first_chunk_time - start_time
                    if total_time_first_chunk > self.min_time_to_debug:
                        logger.debug(
                            f"{self.__class__.__name__} [{writer_id}]: First chunk after {total_time_first_chunk:.3f} s")

                with self.condition:
                    if writer_id == self.next_write_sequence:
                        while buffer:
                            self.queue_out.put(buffer.popleft())
                        self.queue_out.put(chunk)
                    else:
                        buffer.append(chunk)

            last_chunk_time = perf_counter()
            total_time_all_chunks = last_chunk_time - start_time
            if total_time_all_chunks > self.min_time_to_debug:
                logger.debug(f"{self.__class__.__name__} [{writer_id}]: All chunks after {total_time_all_chunks:.3f} s")
            #Получили все данные, но ждём очереди записи.
            if buffer:
                with self.condition:
                    self.condition.wait_for(lambda: self.next_write_sequence == writer_id)
                while buffer:
                    self.queue_out.put(buffer.popleft())


        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}")
            self.stop_event.set()
            self.queue_out.put(b"END")

        with self.condition:
            self.next_write_sequence += 1
            self.condition.notify_all()  # Уведомляем все потоки о том, что переменная изменилась

    @property
    def last_time(self):
        return self._times[-1] if self._times else 0.0

    @property
    def min_time_to_debug(self):
        return 0.001

    def cleanup(self):
        pass
