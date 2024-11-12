import logging
from queue import Queue
from threading import Event
import torch
from utils.thread_manager import ThreadManager
from utils.deiterator import DeiteratorHandler, logger
from INTERRUPTION.interruption_manager_handler import InterruptionManagerHandler
from LLM.openai_api_language_model_server import OpenApiModelServerHandler
#from LLM.openai_api_l_m_server import OpenApiModelServerHandler
from TTS.elevenlabs_tts_handler_server import ElevenLabsTTSServerHandler
from connections.websocket_server import WebSocketHandler
import nltk
# Ensure that the necessary NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    nltk.download("punkt_tab")
try:
    nltk.data.find("tokenizers/averaged_perceptron_tagger_eng")
except (LookupError, OSError):
    nltk.download("averaged_perceptron_tagger_eng")


def setup_logger(log_level):
    global logger
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # torch compile logs
    if log_level == "debug":
        torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)




def main():
    queues_and_events = {
        "stop_event": Event(),
        "input_queue": Queue(),
        "sentences_queue": Queue(),
        "output_queue": Queue(),

    }
    stop_event = queues_and_events["stop_event"]
    input_queue = queues_and_events["input_queue"]
    sentences_queue = queues_and_events["sentences_queue"]
    output_queue = queues_and_events["output_queue"]

    setup_logger("debug")

    websocket = WebSocketHandler(stop_event = stop_event, queue_in = input_queue, queue_out=output_queue)
    llm = OpenApiModelServerHandler(
        stop_event,
        queue_in=input_queue,
        queue_out=sentences_queue,
    )
    tts = ElevenLabsTTSServerHandler(
        stop_event,
        queue_in=sentences_queue,
        queue_out=output_queue,
        threads=2
    )

    try:
        pipeline_manager = ThreadManager([websocket, llm, tts])
        pipeline_manager.start()
        input()
    except KeyboardInterrupt:
        pipeline_manager.stop()


if __name__ == "__main__":
    main()

