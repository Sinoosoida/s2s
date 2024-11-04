from time import perf_counter
import logging
import threading
from queue import Queue
from collections import deque  # Added for efficient buffer management
import concurrent.futures
from utils.process_iterator import ProcessIterator

logger = logging.getLogger(__name__)

class IteratorHandler:
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
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads)

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

            iterator = ProcessIterator()
            self.queue_out.put(iterator)
            self.executor.submit(self.process_and_write, input_data, iterator)

        self.executor.shutdown(wait=True)
        self.cleanup()
        self.queue_out.put(b"END")

    def process_and_write(self, input_data, iterator):
        start_time = perf_counter()
        first_chunk = True

        try:
            for chunk in self.process(input_data):
                if first_chunk:
                    logger.debug(f"{self.__class__.__name__} started output after: {perf_counter() - start_time:.3f} s")
                    first_chunk = False
                iterator.put(chunk)
            iterator.close()
            logger.debug(f"{self.__class__.__name__} finished output after: {perf_counter() - start_time:.3f} s")
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}")
            self.stop_event.set()
            self.queue_out.put(b"END")
            return

    @property
    def last_time(self):
        return self._times[-1] if self._times else 0.0

    @property
    def min_time_to_debug(self):
        return 0.001

    def cleanup(self):
        pass
