from dataclasses import dataclass


@dataclass
class Plate:
    filename: str
    label: str
    prediction: str
