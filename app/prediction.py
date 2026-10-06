'''The Prediction type shared by the classifier and the smoother.

Kept in its own dependency-free module so pure logic (the smoother) can use
it without importing torch.
'''

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    '''The model's top label and its softmax confidence for one frame.'''

    label: str
    confidence: float
