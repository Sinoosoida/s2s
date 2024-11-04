from utils.process_iterator import ProcessIterator
from time import perf_counter
import logging
import json
import os
import random
from utils.data import ImmutableDataChain

logger = logging.getLogger(__name__)


class FillerHandler:
    """
    Combines the functionality of BaseHandler and TextProcessingHandler into a single class.

    Processes text data by adding a greeting at the beginning. Each part of the pipeline has an input and an output queue.
    The `setup` method along with `setup_args` and `setup_kwargs` can be used to address the specific requirements of the implemented pipeline part.
    To stop a handler properly, set the stop_event and, to avoid queue deadlocks, place b"END" in the input queue.
    Objects placed in the input queue will be processed by the `process` method, and the yielded results will be placed in the output queue.
    The cleanup method handles stopping the handler, and b"END" is placed in the output queue.
    """

    def __init__(self, stop_event, queue_in, queue_out_mess, queue_out_audio, setup_args=(), setup_kwargs={}):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.queue_out_mess = queue_out_mess
        self.queue_out_audio = queue_out_audio
        self.setup(*setup_args, **setup_kwargs)
        self._times = []

    def setup(
            self,
            audio_data_dir="data/filler_data",
            audio_description_json_path="data/filler_data/description.json",
            activated=True,
            device="cuda",
            gen_kwargs = {},
    ):
        self.audio_data_dir = audio_data_dir
        self.audio_description_json_path = audio_description_json_path
        self.activated = activated
        self.device = device
        self.audio_descriptions = []  # Will hold the loaded JSON data
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        # Load the JSON file into memory
        try:
            with open(self.audio_description_json_path, 'r', encoding='utf-8') as f:
                self.audio_descriptions = json.load(f)
            logger.debug(f"Loaded audio descriptions from {self.audio_description_json_path}")
        except Exception as e:
            logger.error(f"Error loading audio descriptions: {e}")
            self.stop_event.set()
            return

        # Verify that all audio files listed in the JSON are present
        for item in self.audio_descriptions:
            audio_filename = item.get('filename', None)
            if not audio_filename:
                logger.warning(f"No 'filename' in item: {item}")
                continue
            audio_path = os.path.join(self.audio_data_dir, audio_filename)
            if not os.path.isfile(audio_path):
                logger.error(f"Missing audio file: {audio_path}")
                self.stop_event.set()
                return

    def process(self, data: ImmutableDataChain):
        if not self.activated:
            # If the handler is deactivated, pass the data unchanged
            self.queue_out_mess.put(data)
            return
        else:
            # input_data = data.get_data()
            if not self.audio_descriptions:
                logger.error("Audio descriptions not loaded. Cannot process data.")
                self.stop_event.set()
                return

            # Select a random audio file description
            random_item = random.choice(self.audio_descriptions)
            text_content = random_item.get('text', '')
            if text_content:
                self.queue_out_mess.put(data.add_data(text_content, "start_phrase"))
                logger.debug(f"Added start_phrase: {text_content}")
            else:
                logger.warning(f"No 'text' field in random item: {random_item}")

            # Read the audio file and put it into queue_out_audio
            audio_filename = random_item.get('filename', None)
            if audio_filename:
                audio_path = os.path.join(self.audio_data_dir, audio_filename)
                if os.path.isfile(audio_path):
                    try:
                        with open(audio_path, 'rb') as f:
                            audio_data = f.read()

                        # to return iterator
                        iterator = ProcessIterator()
                        iterator.put(audio_data)
                        iterator.close()
                        logger.debug(f"Added audio data from file: {audio_filename}")
                        self.queue_out_audio.put(data.add_data(iterator, "output_audio_iterator"))

                    except Exception as e:
                        logger.error(f"Error reading audio file {audio_filename}: {e}")
                        self.stop_event.set()
                        return
                else:
                    logger.error(f"Audio file not found: {audio_path}")
                    self.stop_event.set()
                    return
            else:
                logger.warning(f"No 'filename' in random item: {random_item}")
                self.stop_event.set()
                return

    def run(self):
        while not self.stop_event.is_set():
            input_data_chain = self.queue_in.get()
            if isinstance(input_data_chain, bytes) and input_data_chain == b"END":
                # Sentinel signal to avoid queue deadlock
                logger.debug("Stopping thread")
                break
            start_time = perf_counter()
            self.process(input_data_chain)
            self._times.append(perf_counter() - start_time)
            if self.last_time > self.min_time_to_debug:
                logger.debug(f"{self.__class__.__name__}: {self.last_time:.3f} s")
            start_time = perf_counter()

        self.cleanup()
        self.queue_out_mess.put(b"END")
        self.queue_out_audio.put(b"END")

    @property
    def last_time(self):
        return self._times[-1] if self._times else 0

    @property
    def min_time_to_debug(self):
        return 0.001

    def cleanup(self):
        # Any necessary cleanup actions can be added here
        pass
