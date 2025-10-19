from typing import Tuple
from dataclasses import dataclass, field


@dataclass
class OccupancyEvalConfig:
    MODE: str = "cascade"

    QUERY_BATCH_SIZE: int = 50000
    OCC_THRESHOLD: float = 0.5
    # Default values according to Table 1 in
    # https://arxiv.org/pdf/2301.07668
    X_RANGE: Tuple[float, float] = field(default_factory=lambda: (-4, 4))
    Y_RANGE: Tuple[float, float] = field(default_factory=lambda: (0, 0.75))
    # Z_RANGE: Tuple[float, float] = field(default_factory=lambda: [20, 3])
    Z_RANGE: Tuple[float, float] = field(default_factory=lambda: (20, 4))
    # GT_AGGREGATE_TIMESTEPS: int = 20
    GT_AGGREGATE_TIMESTEPS: int = 300#20
    Y_RES: int = 1
    
    CUT_FAR_INVISIBLE_AREA: bool = True

    # If set, the GT occupancy maps are loaded from here 
    # instead of being recomputed.
    READ_GT_OCC_PATH: str = ""
    # If set, the GT occupancy maps are saved here.
    SAVE_GT_OCC_MAP_PATH: str = ""