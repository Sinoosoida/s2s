from dataclasses import dataclass, field

@dataclass
class MMSTTSHandlerArguments:
    mms_tts_device: str = field(
        default="cuda",
        metadata={
            "help": "The device to be used for speech synthesis. Default is 'cuda'."
        },
    )
    mms_tts_model_name: str = field(
        default="facebook/mms-tts-rus",
        metadata={
            "help": "The name of the TTS model to be used. Default is 'facebook/mms-tts-rus'."
        },
    )
