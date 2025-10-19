from typing import List, Tuple, Any
from dataclasses import dataclass, field
from configs.constants.constants import DEFINITIONS


@dataclass
class OcclusionDetectionConfig:
    USE_DEPTH: bool = True
    USE_FLOW: bool = True
    USE_DEPTH_GRADS: bool = False
    AREA_THRESH: float = 500.0
    DEPTH_THRESH: float = 0.0
    DEPTH_GRADS_THRESH: float = 0.1
    FINAL_DEPTH_THRESH: float = 3.0     # TODO this may not be used
    OUT_OF_BOUND_THRESH_H: float = 0.0#0.05
    OUT_OF_BOUND_THRESH_W: float = 0.0#0.1
    POSTPROCESSING_OPS_DEPTH: List[Any] = field(default_factory=lambda: [        
        (DEFINITIONS.OPENING, 3),
        (DEFINITIONS.DILATION, 15),
    ])
    POSTPROCESSING_OPS_FLOW: List[Any] = field(default_factory=lambda: [
        (DEFINITIONS.OPENING, 3),
        (DEFINITIONS.CLOSING, 15),
    ])
    POSTPROCESSING_OPS_DEPTH_GRADS: List[Any] = field(default_factory=lambda: [
        (DEFINITIONS.CLOSING, 5),
    ])
    MAX_OCCLUSION_DETECTION_DEPTH: float = 50.0