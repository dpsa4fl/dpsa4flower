
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from flwr.common.typing import Scalar
from dataclass_wizard import JSONWizard

Shapes = List[Tuple[int, ...]]
SplitIndices = List[int]

@dataclass
class ReshapingConfig(JSONWizard):
    shapes: Shapes
    split_indices: SplitIndices

def readReshapingConfig(d: Dict[str, Scalar]) -> Optional[ReshapingConfig]:
    reshaping_config_encoded = d.get('reshaping_config')

    if isinstance(reshaping_config_encoded, str):
        c = ReshapingConfig.from_json(reshaping_config_encoded)
        if isinstance(c, ReshapingConfig):
            return c
        else:
            print("WARNING: Got a list instead of a single object! (when deserializing a ReshapingConfig)")
            return None
    else:
        return None

def writeReshapingConfig(d: Dict[str, Scalar], c: ReshapingConfig) -> Dict[str, Scalar]:
    d['reshaping_config'] = c.to_json()
    return d

