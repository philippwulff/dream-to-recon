# This file defines naming conventions.
from dataclasses import dataclass

@dataclass
class NamingAndValueDefinitions:
    # Values in occlusion maps
    IS_VISIBLE = 1
    IS_OCCLUDED = 0
    IS_INVALID = -1
    # Morphological ops from CV2
    DILATION = "dilation"
    EROSION = "erosion"
    OPENING = "opening"
    CLOSING = "closing"
    
DEFINITIONS = NamingAndValueDefinitions()

DEPTH_NORM_CONST_IN_PNG = 255.0