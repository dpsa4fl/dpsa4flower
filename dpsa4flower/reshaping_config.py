
import ast
from typing import Dict, Optional, Tuple, List
from flwr.common.typing import Scalar

Shapes = List[Tuple[int, ...]]
SplitIndices = List[int]

class ReshapingConfig:
    def __init__(self, shapes: Shapes, split_indices: SplitIndices) -> None:
        self.shapes = shapes
        self.split_indices = split_indices

def readReshapingConfig(d: Dict[str, Scalar]) -> Optional[ReshapingConfig]:
    reshaping_config_encoded = d.get('reshaping_config')

    if isinstance(reshaping_config_encoded, str):
        return ast.literal_eval(reshaping_config_encoded)
    else:
        return None

def writeReshapingConfig(d: Dict[str, Scalar], c: ReshapingConfig) -> Dict[str, Scalar]:
    d['reshaping_config'] = repr(c)
    return d

