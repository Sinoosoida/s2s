import logging
from baseHandler import BaseHandler
from utils.data import ImmutableDataChain

logger = logging.getLogger(__name__)


class DeiteratorHandler(BaseHandler):

    def setup(self):
        pass

    def warmup(self):
        pass

    def process(self, data:ImmutableDataChain):
        iterator = data.get_data("output_audio_iterator")
        for chunk in iterator:
            yield data.add_data(chunk, "output_audio_chunk")
