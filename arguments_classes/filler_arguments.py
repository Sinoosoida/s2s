from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FillerHandlerArguments:
    filler_audio_data_dir: str = field(
        default="data/filler_data",
        metadata={
            "help": "Directory for audio"
        },
    )
    filler_audio_description_json_path: str = field(
        default="data/filler_data/description.json",
        metadata={
            "help": "Directory for json"
        },
    )
    filler_activated: str = field(
        default=True,
        metadata={
            "help": "If activated, fillers are adding"
        },
    )
    filler_device: str = field(
        default="cuda",
        metadata={
            "help": "Is not using just now"
        },
    )