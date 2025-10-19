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
    
    
@dataclass
class OcclusionConfig:
    USE_DEPTH = True
    USE_FLOW = True
    AREA_THRESH = 50
    DEPTH_THRESH = 4.0
    FINAL_DEPTH_THRESH = 3.0
    OUT_OF_BOUND_THRESH_H = 0.05
    OUT_OF_BOUND_THRESH_W = 0.1
    POSTPROCESSING_OPS_DEPTH = [        
        (DEFINITIONS.OPENING, 5),
        (DEFINITIONS.CLOSING, 5),
    ]
    POSTPROCESSING_OPS_FLOW = [
        (DEFINITIONS.OPENING, 3),
        (DEFINITIONS.CLOSING, 9),
    ]

