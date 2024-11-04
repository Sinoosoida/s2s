import logging
from queue import Queue, Empty
from typing import List
from utils.data import FilteredQueue

logger = logging.getLogger(__name__)

class InterruptionManagerHandler:
    def __init__(self, stop_event, interruption_request_queue : Queue, filtered_queues : List[FilteredQueue]):
        self.stop_event = stop_event
        self.interruption_request_queue = interruption_request_queue
        self.filtered_queues = filtered_queues

    def run(self):
        logger.debug("Interruption Manager started.")
        while not self.stop_event.is_set():
            try:
                interruption_request = self.interruption_request_queue.get(timeout=0.05)
                if interruption_request is not None:
                    logger.debug(f"Processing interruption request: {interruption_request}")
                    for filtered_queue in self.filtered_queues:
                        filtered_queue.filter(interruption_request)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error while handling interruption: {e}")

        logger.debug("Interruption Manager stopped.")
