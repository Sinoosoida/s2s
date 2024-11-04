import torch
from utils.data import ImmutableDataChain
from connections.socket_sender import logger
from queue import Queue


class VADIterator:
    def __init__(
        self,
        model,
        start_data: ImmutableDataChain,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        interruption_request_queue: Queue= None,
        min_speech_ms: int = 1000,
    ):
        """
        Mainly taken from https://github.com/snakers4/silero-vad
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit/.onnx silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = model
        self.start_data = start_data
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.is_speaking = False
        self.speech_start_sample = 0
        self.buffer = []
        self.interruption_request_queue = interruption_request_queue
        self.min_speech_ms = min_speech_ms

        if sampling_rate not in [8000, 16000]:
            raise ValueError(
                "VADIterator does not support sampling rates other than [8000, 16000]"
            )
        if interruption_request_queue is None:
            logger.debug("No interruption in VAD")

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        self.samples_in_buffer = 0
        self.event_set = False

    @torch.no_grad()
    def __call__(self, x):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except Exception:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:#чел говорит, сдвигаем таймер времени с момента окончания речи
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:#чел начал говорить
            self.triggered = True
            self.event_set = False
            return None

        if self.triggered and (self.samples_in_buffer / self.sampling_rate * 1000)  >= self.min_speech_ms and not self.event_set:
            self.interruption_request_queue.put(self.start_data.get_counter())
            self.event_set = True

        if (speech_prob < self.threshold - 0.15) and self.triggered:#чел вроде закончил говорить
            if not self.temp_end:
                self.temp_end = self.current_sample #запоминаем время, когда он закончил говорить
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None #если он не так много молчал, то ничего не возвращаем пока что
            else:
                # end of speak тут он уже долго молчит
                self.temp_end = 0
                self.triggered = False
                spoken_utterance = self.buffer
                self.buffer = []
                self.samples_in_buffer = 0
                return spoken_utterance

        if self.triggered:
            self.buffer.append(x)
            self.samples_in_buffer+=window_size_samples

        return None
