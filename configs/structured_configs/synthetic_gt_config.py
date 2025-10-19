import re
from typing import List, Tuple, Any, Optional, Dict
from omegaconf import MISSING
from dataclasses import dataclass, field

import matplotlib.pyplot as plt

from configs.structured_configs.occlusions_config import OcclusionDetectionConfig
from configs.structured_configs.bts_config import RendererConfig
from configs.constants.constants import DEFINITIONS


# --- BEGIN CAMERA SAMPLER CONFIGS --- 


@dataclass
class CameraSamplerConfig:
    TYPE: str = MISSING
    # Optional unique name used for registering the sympler in the Hydra config.
    # If not given, the TYPE is used instead.
    UNIQUE_NAME: Optional[str] = None 
    # Defult arguments after non-default arguments.
    NUM_NOVEL_VIEWS: int = 1
    EDGE_DIST_FOR_PROJ_SAMPLE: float = 0.0
    # One of: ["none", "center_crop", "crop_to_visible"]
    MAKE_PROJS_POLICY: str = "center_crop"
    APPLY_CAM_INCL_ADJUST: bool = False
    
    # We need to define this attribute here, such that we can implement it 
    # as a property lateron. Has type: List[CameraSamplerConfig].
    CAMERA_SAMPLER_CONFIGS: Any = field(init=False, default=MISSING)
    # Field used in the RandomCoiceCameraSsampler
    CHOSEN_SAMPLER_TYPE: Any = field(init=False, default=MISSING)
    
    
@dataclass
class OrbitCameraSamplerConfig(CameraSamplerConfig):
    TYPE: str = "OrbitCameraSampler"     # Rename to orbit
    X_LIMS: Tuple[float, float] = field(default_factory=lambda: (0., 0.))
    Y_LIMS: Tuple[float, float] = field(default_factory=lambda: (-10., 10.))
    Z_DIST_LIMS: Tuple[float, float] = field(default_factory=lambda: (5., 5.))
    ORBIT_POLICY: str = "cam_frame"


@dataclass
class ShiftCameraSamplerConfig(CameraSamplerConfig):
    TYPE: str = "ShiftCameraSampler"
    X_LIMS: Tuple[float, float] = field(default_factory=lambda: (-3., 3.))
    Y_LIMS: Tuple[float, float] = field(default_factory=lambda: (0., 0.))
    Z_LIMS: Tuple[float, float] = field(default_factory=lambda: (0., 0.))


@dataclass
class ShiftRotCameraSamplerConfig(CameraSamplerConfig):
    TYPE: str = "ShiftRotCameraSampler"
    X_LIMS: Tuple[float, float] = field(default_factory=lambda: (0., 0.))
    Y_LIMS: Tuple[float, float] = field(default_factory=lambda: (0., 0.))
    Z_LIMS: Tuple[float, float] = field(default_factory=lambda: (0., 0.))
    X_ROT_LIMS: Tuple[float, float] = field(default_factory=lambda: (0., 0.))
    Y_ROT_LIMS: Tuple[float, float] = field(default_factory=lambda: (0., 0.))
    Z_ROT_LIMS: Tuple[float, float] = field(default_factory=lambda: (0., 0.))
    

@dataclass
class RigCameraSamplerConfig(CameraSamplerConfig):
    TYPE: str = "RigCameraSampler"
    CAMS_XYZ: list[Any] = field(default_factory=lambda: [[0., 0., 0.]])
    CAMS_ALPHA_BETA_GAMMA: list[Any] = field(default_factory=lambda: [[0., 0., 0.]])


@dataclass
class Rig12CameraSamplerConfig(RigCameraSamplerConfig):
    """
    Contains some pre-defined cameras for KITTI-360 scenes.
    Poses are shifted along the global coordinate axes (by adjusting camera inclination), 
    then orientated back to -5° inclination.
    """
    TYPE: str = "RigCameraSampler"
    UNIQUE_NAME: str = "Rig12CameraSampler"
    NUM_NOVEL_VIEWS = 12
    MAKE_PROJS_POLICY: str = "center_crop"
    APPLY_CAM_INCL_ADJUST: bool = True
    CAMS_XYZ: list[Any] = field(default_factory=lambda: [
        # x=±4 rot(y)=±20°
        [-4., 0., 0.],
        [-4., 0., 0.],
        [+4., 0., 0.],
        [+4., 0., 0.],
        # x=±2 rot(y)=±20°
        [-2., 0., 0.],
        [-2., 0., 0.],
        [+2., 0., 0.],
        [+2., 0., 0.],
        # z=+2 rot(y)=±45°
        [0., 0., 2.],
        [0., 0., 2.],
        # z=+5 rot(y)=±45°
        [0., 0., 5.],
        [0., 0., 5.],
    ])
    CAMS_ALPHA_BETA_GAMMA: list[Any] = field(default_factory=lambda: [
        # x=±4 rot(y)=±20°
        [-5., 0., 0.],
        [-5., 20., 0.],
        [-5., 0., 0.],
        [-5., -20., 0.],
        # x=±2 rot(y)=±20°
        [-5., 0., 0.],
        [-5., 20., 0.],
        [-5., 0., 0.],
        [-5., -20., 0.],
        # z=+2 rot(y)=±30°
        [-5., 30., 0.],
        [-5., -30., 0.],
        # z=+5 rot(y)=±45°
        [-5., 45., 0.],
        [-5., -45., 0.],
    ])


@dataclass
class Rig8CameraSamplerConfig(Rig12CameraSamplerConfig):
    UNIQUE_NAME: str = "Rig8CameraSampler"
    NUM_NOVEL_VIEWS = 8
    CAMS_XYZ: list[Any] = field(default_factory=lambda: [
        # x=±4 rot(y)=±20°
        [-4., 0., 0.],
        [-4., 0., 0.],
        [+4., 0., 0.],
        [+4., 0., 0.],
        # x=±2 rot(y)=±20°
        [-2., 0., 0.],
        [-2., 0., 0.],
        [+2., 0., 0.],
        [+2., 0., 0.],
    ])
    CAMS_ALPHA_BETA_GAMMA: list[Any] = field(default_factory=lambda: [
        # x=±4 rot(y)=±20°
        [-5., 0., 0.],
        [-5., 20., 0.],
        [-5., 0., 0.],
        [-5., -20., 0.],
        # x=±2 rot(y)=±20°
        [-5., 0., 0.],
        [-5., 20., 0.],
        [-5., 0., 0.],
        [-5., -20., 0.],
    ])


@dataclass
class Rig8CameraSamplerWaymoConfig(Rig8CameraSamplerConfig):
    UNIQUE_NAME: str = "Rig8CameraSamplerWaymo"
    CAMS_ALPHA_BETA_GAMMA: list[Any] = field(default_factory=lambda: [
        # x=±4 rot(y)=±20°
        [0., 0., 0.],
        [0., 20., 0.],
        [0., 0., 0.],
        [0., -20., 0.],
        # x=±2 rot(y)=±20°
        [0., 0., 0.],
        [0., 20., 0.],
        [0., 0., 0.],
        [0., -20., 0.],
    ])


@dataclass
class Rig4CameraSamplerConfig(Rig12CameraSamplerConfig):
    UNIQUE_NAME: str = "Rig4CameraSampler"
    NUM_NOVEL_VIEWS = 4
    CAMS_XYZ: list[Any] = field(default_factory=lambda: [
        [-4., 0., 0.],
        [+4., 0., 0.],
        [-2., 0., 0.],
        [+2., 0., 0.],
    ])
    CAMS_ALPHA_BETA_GAMMA: list[Any] = field(default_factory=lambda: [
        [-5., 0., 0.],
        [-5., 0., 0.],
        [-5., 0., 0.],
        [-5., 0., 0.],
    ])


@dataclass
class Rig4V2CameraSamplerConfig(Rig12CameraSamplerConfig):
    UNIQUE_NAME: str = "Rig4V2CameraSampler"
    NUM_NOVEL_VIEWS = 4
    CAMS_XYZ: list[Any] = field(default_factory=lambda: [
        [-3., 0., 0.],
        [+3., 0., 0.],
        [-1., 0., 0.],
        [+1., 0., 0.],
    ])
    CAMS_ALPHA_BETA_GAMMA: list[Any] = field(default_factory=lambda: [
        [-5., 0., 0.],
        [-5., 0., 0.],
        [-5., 0., 0.],
        [-5., 0., 0.],
    ])


@dataclass
class MovementCameraSamplerConfig(CameraSamplerConfig):
    TYPE: str = "MovementCameraSampler"
    TRAJECTORY_FILE_NAME: str = "simple_movement.np"
    SCALE: float = 1.0
    SKIP: int = 1

    
@dataclass
class TrajectoryCameraSamplerConfig(CameraSamplerConfig):
    TYPE: str = "TrajectoryCameraSampler"
    POLICY: str = "left_right"
    NUM_STEPS: int = 10
    SAMPLER_INIT_KWARGS: Dict = field(default_factory=lambda: {})


@dataclass
class RandomChoiceCameraSamplerConfig(CameraSamplerConfig):
    TYPE: str = "RandomChoiceCameraSampler"
    MAKE_PROJS_POLICY: str = "crop_to_visible"
    EDGE_DIST_FOR_PROJ_SAMPLE: float = 0.0
    CHOSEN_SAMPLER_TYPE: str = "shift_m3to3"

    @property
    def CAMERA_SAMPLER_CONFIGS(self) -> list[Any]:
        match self.CHOSEN_SAMPLER_TYPE:
            case "shift_m3to3":
                return [
                    ShiftCameraSamplerConfig(X_LIMS=[-3., -1.], Z_LIMS=[0., 0.]),
                    ShiftCameraSamplerConfig(X_LIMS=[1., 3.], Z_LIMS=[0., 0.]),
                ]
            case "shift_m4to4":
                return [
                    ShiftCameraSamplerConfig(X_LIMS=[-4., -1.], Z_LIMS=[0., 0.]),
                    ShiftCameraSamplerConfig(X_LIMS=[1., 4.], Z_LIMS=[0., 0.]),
                ]
            case "orbit":  # Strategy for CO3D
                return [
                    OrbitCameraSamplerConfig(Y_LIMS=(-10., -2.), Z_DIST_LIMS=(2., 3.)),
                    OrbitCameraSamplerConfig(Y_LIMS=(2., 10.), Z_DIST_LIMS=(2., 3.)),
                ]
            case _:
                raise ValueError(f"CHOSEN_SAMPLER_TYPE {self.CHOSEN_SAMPLER_TYPE} is not available.")

@dataclass
class ExplorationCameraSamplerConfig(CameraSamplerConfig):
    TYPE: str = "ExplorationCameraSampler"
    VISUALIZE: bool = False
    VISUALIZATION_PATH: str = "media/exploration_sampler_vis"
    VISUALIZATION_NO_TEXT: bool = True
    VISUALIZATION_STEPS: List[int] | None = None
    VISUALIZE_GIF: bool = False
    NUM_PROPOSALS: int = 9
    NUM_STEPS: int = 100
    INIT_POLICY: str = "stratified"
    # Optional plt.Axes object to use for plotting instead of creating a new figure.
    LOSS_PLOT_AX: Any = None
    ZLIMS: Tuple[float, float] = field(default_factory=lambda: [0, 10])
    
    LAMBDA_XZ_BOUNDS: float = 1.0
    LAMBDA_BETA_BOUNDS: float = 1.0
    LAMBDA_OCCLUSION: float = 1.0
    LAMBDA_OCCLUSION_CENTER: float = 1.0
    LAMBDA_OCCLUSION_EDGES: float = 1.0
    LAMBDA_DEPTH: float = 0.1
    LAMBDA_SIGMAS: float = 0.0
    LAMBDA_POSE_SIM: float = 0.0
    LAMBDA_WEIGHTS: float = 1.0

    
    
ALL_CAMERA_SAMPLER_CONFIGS = [
    OrbitCameraSamplerConfig,
    ShiftCameraSamplerConfig,
    ShiftRotCameraSamplerConfig,
    RigCameraSamplerConfig,
    Rig12CameraSamplerConfig,
    Rig8CameraSamplerConfig,
    Rig4CameraSamplerConfig,
    MovementCameraSamplerConfig,
    TrajectoryCameraSamplerConfig,
    RandomChoiceCameraSamplerConfig,
    ExplorationCameraSamplerConfig,
]
    

# --- END CAMERA SAMPLER CONFIGS --- 

@dataclass
class PseudoVolumeConfig:
    COLOR_SAMPLING_MODE: str = "mean"
    SURFACE_THRESH: float = 1.0
    SET_ONLY_OCCLUSIONS_EMPTY: bool = False
    DENSITY_VALUE: float = 1000.0
    DENSITY_DOWNSCALE_FACTOR: float = 100.0
    

@dataclass
class CascadeConfig:
    ENCODING_POLICY: str = "input+anchor"
    SHOW_PROGRESS_BAR: bool = False
    NON_OCCLUSION_MASKS: bool = False
    ANCHOR_SEED: int | None = None
    CASCADE_SEED: int | None = None
    TRAJECTORY_SAMPLER_CONFIG: TrajectoryCameraSamplerConfig = TrajectoryCameraSamplerConfig(
        # Use a single step by default, such that we have minimal 
        # overhead when we just want to encode the previous anchors.
        NUM_STEPS=1,
    )
    # ANCHOR_SAMPLER_CONFIGS: List[CameraSamplerConfig] | None = None
    CHOSEN_ANCHOR_TYPE: str = "none"
    
    _anchor_sampler_configs: Optional[List[CameraSamplerConfig]] = field(default=MISSING, init=False)

    @property
    def ANCHOR_SAMPLER_CONFIGS(self) -> Optional[List[CameraSamplerConfig]]:
        if self._anchor_sampler_configs is MISSING:
            self._anchor_sampler_configs = self._generate_default_anchor_configs()
        return self._anchor_sampler_configs

    # Adding a setter allows changing sampler values (e.g. visualization settings) 
    # after dataclass creation.
    @ANCHOR_SAMPLER_CONFIGS.setter
    def ANCHOR_SAMPLER_CONFIGS(self, value: Optional[List[CameraSamplerConfig]]):
        self._anchor_sampler_configs = value

    def _generate_default_anchor_configs(self) -> List[Any] | None:
        match self.CHOSEN_ANCHOR_TYPE:
            case "empty":
                return []
            case "explore":
                return [
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=8, NUM_NOVEL_VIEWS=2, VISUALIZE=False, ZLIMS=[0,15])
                        for _ in range(6)
                    ]
            case "rig12_explore_short":
                return [
                    Rig12CameraSamplerConfig(NUM_NOVEL_VIEWS=4),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,5]),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,5]),
                ]
            case "rig12_explore_mid":
                return [
                    Rig12CameraSamplerConfig(NUM_NOVEL_VIEWS=4),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,10]),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,10]),
                ]
            case "rig12_8_explore3x4_ranged":
                return [
                    Rig12CameraSamplerConfig(NUM_NOVEL_VIEWS=8),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,5]),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,10]),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,20]),
                ]
            case "rig12_explore_far":
                return [
                    Rig12CameraSamplerConfig(NUM_NOVEL_VIEWS=4),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,20]),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,20]),
                ]
            case "rig12_8_explore3x":
                return [
                    Rig12CameraSamplerConfig(NUM_NOVEL_VIEWS=8),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=2, VISUALIZE=False, ZLIMS=[0,10]),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=2, VISUALIZE=False, ZLIMS=[0,10]),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=2, VISUALIZE=False, ZLIMS=[0,10]),
                ]
            case "rig12_8_explore3x4":
                return [
                    Rig12CameraSamplerConfig(NUM_NOVEL_VIEWS=12),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,10]),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,10]),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,10]),
                ]
            case "rig12_8_explore4x4":
                return [
                    Rig12CameraSamplerConfig(NUM_NOVEL_VIEWS=8),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,10]),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,10]),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,10]),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=4, VISUALIZE=False, ZLIMS=[0,10]),
                ]
            case "rig12_8_explore3x_far":
                return [
                    Rig12CameraSamplerConfig(NUM_NOVEL_VIEWS=8),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=2, VISUALIZE=False, ZLIMS=[0,20]),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=2, VISUALIZE=False, ZLIMS=[0,20]),
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=12, NUM_NOVEL_VIEWS=2, VISUALIZE=False, ZLIMS=[0,20]),
                ]
            case "eval_controlnet_rig":
                return [
                    RigCameraSamplerConfig(CAMS_XYZ=[[3, 0, 0]], CAMS_ALPHA_BETA_GAMMA=[[0, 0, 0]]),
                ]
            case "eval_controlnet_rig12":
                return [
                    Rig12CameraSamplerConfig(NUM_NOVEL_VIEWS=4),
                ]
            case "rig":
                return [
                    RigCameraSamplerConfig(CAMS_XYZ=[[3, 0, 0]], CAMS_ALPHA_BETA_GAMMA=[[0, 0, 0]]),
                    RigCameraSamplerConfig(CAMS_XYZ=[[3, 0, 0]], CAMS_ALPHA_BETA_GAMMA=[[0, -20, 0]]),
                    RigCameraSamplerConfig(CAMS_XYZ=[[0, 0, 5]], CAMS_ALPHA_BETA_GAMMA=[[0, 20, 0]]),
                    RigCameraSamplerConfig(CAMS_XYZ=[[-3, 0, 0]], CAMS_ALPHA_BETA_GAMMA=[[0, 20, 0]]),
                ]
            case "rig4of4":
                return [
                    Rig4CameraSamplerConfig(NUM_NOVEL_VIEWS=4),
                ]
            case "rig4of4plus4":
                return [
                    Rig4CameraSamplerConfig(NUM_NOVEL_VIEWS=4),
                    RigCameraSamplerConfig(
                        NUM_NOVEL_VIEWS=4, 
                        CAMS_XYZ=[
                            [0., 0., 2.],
                            [0., 0., 2.],
                            [0., 0., 5.],
                            [0., 0., 5.],
                            ],
                        CAMS_ALPHA_BETA_GAMMA=[
                            [-5., 30., 0.],
                            [-5., -30., 0.],
                            [-5., -30., 0.],
                            [-5., 30., 0.],
                            ]
                    ),
                ]
            case "rig8of8":
                return [
                    Rig8CameraSamplerConfig(NUM_NOVEL_VIEWS=8),
                ]
            case "rig8of8_waymo":
                return [
                    Rig8CameraSamplerWaymoConfig(NUM_NOVEL_VIEWS=8),
                ]
            case "rig8of8plus8":
                return [
                    Rig8CameraSamplerConfig(NUM_NOVEL_VIEWS=8),
                    RigCameraSamplerConfig(
                        NUM_NOVEL_VIEWS=8, 
                        CAMS_XYZ=[
                            [0., 0., 2.],
                            [0., 0., 2.],
                            [2., 0., 2.],
                            [-2., 0., 2.],
                            [0., 0., 5.],
                            [0., 0., 5.],
                            [2., 0., 5.],
                            [-2., 0., 5.]
                            ],
                        CAMS_ALPHA_BETA_GAMMA=[
                            [-5., 30., 0.],
                            [-5., -30., 0.],
                            [-5., -30., 0.],
                            [-5., 30., 0.],
                            [-5., 45., 0.],
                            [-5., -45., 0.],
                            [-5., -45., 0.],
                            [-5., 45., 0.]
                            ]
                    ),
                ]
            case "rig12of12":
                return [
                    Rig12CameraSamplerConfig(NUM_NOVEL_VIEWS=12),
                ]
            case "rig_davis":
                return [
                    Rig4CameraSamplerConfig(
                        CAMS_XYZ=[[-2, 0, 0], [-1, 0, 0], [1, 0, 0], [2, 0, 0]],
                        CAMS_ALPHA_BETA_GAMMA=[[0, 15, 0], [0, 5, 0], [0, -5, 0], [0, -15, 0]]
                    ),  
                ]
            case "orbit8":
                return [
                    OrbitCameraSamplerConfig(NUM_NOVEL_VIEWS=8),
                ]
            case "shift8":
                return [
                    ShiftCameraSamplerConfig(NUM_NOVEL_VIEWS=8),
                ]
            case "orbit4":
                return [
                    OrbitCameraSamplerConfig(NUM_NOVEL_VIEWS=4),
                ]
            case "shift4":
                return [
                    ShiftCameraSamplerConfig(NUM_NOVEL_VIEWS=4),
                ]
            case "plot_shift":
                return [
                    ShiftCameraSamplerConfig(X_LIMS=(4,4)),
                ]
            case "plot_rig1":
                return [
                    RigCameraSamplerConfig(CAMS_XYZ=[[4,0,3]], CAMS_ALPHA_BETA_GAMMA=[[0,-25,0]]),
                ]
            case "plot_rig_final":
                return [
                    RigCameraSamplerConfig(CAMS_XYZ=[[4,0,3]], CAMS_ALPHA_BETA_GAMMA=[[0,-25,0]]),
                    Rig8CameraSamplerConfig(NUM_NOVEL_VIEWS=6),
                ]
            case "empty":
                return []
            case "none":
                return None
            case _ if re.match(r"^explore_\d+_\d+_\d+_[A-Za-z]+_\d+$", self.CHOSEN_ANCHOR_TYPE):
                # Case for grid search purposes.
                print(f"Parsing self.CHOSEN_ANCHOR_TYPE={self.CHOSEN_ANCHOR_TYPE}")
                # explore_${numprop}_${numnv}_${numcasc}_${initstrat}_${zlimsfar}
                _, num_proposals, num_novel_views, num_cascades, init_policy, zlim_far = self.CHOSEN_ANCHOR_TYPE.split("_")
                return [
                    ExplorationCameraSamplerConfig(NUM_PROPOSALS=int(num_proposals), NUM_NOVEL_VIEWS=int(num_novel_views), INIT_POLICY=init_policy, VISUALIZE=False, ZLIMS=(0, float(zlim_far)))
                        for _ in range(int(num_cascades))
                    ]
            case _:
                raise ValueError(f"Anchor type {self.CHOSEN_ANCHOR_TYPE} is not available.")

    # @property
    # def ANCHOR_SAMPLER_CONFIGS(self) -> List[CameraSamplerConfig] | None:
    #     match self.CHOSEN_ANCHOR_TYPE:
    #         case "explore":
    #             return [
    #                 ExplorationCameraSamplerConfig(
    #                     NUM_PROPOSALS=4, NUM_NOVEL_VIEWS=1, VISUALIZE=False, 
    #                     # VISUALIZATION_PATH=os.path.join(out_path, "exploration_sampler", str(i))
    #                 ) 
    #                 for i in range(5)
    #             ]

@dataclass
class SyntheticGTConfig:
    # The resolution of the input image used for encoding the pseudo volume.
    ENCODING_RESOLUTION: Optional[Tuple[int, int]] = None
    # The resolution of the rendered novel view.
    RERENDERING_RESOLUTION: Optional[Tuple[int, int]] = None
    
    # Novel view camera sampler
    NV_CAM_SAMPLER: CameraSamplerConfig = CameraSamplerConfig()
    
    # Config for occlusion detection.
    OCCLUSIONS: OcclusionDetectionConfig = OcclusionDetectionConfig(
        USE_DEPTH=True,
        USE_DEPTH_GRADS=False,
        USE_FLOW=False,
        AREA_THRESH=500.,
        DEPTH_THRESH=0.1,
        POSTPROCESSING_OPS_DEPTH = [
            [DEFINITIONS.CLOSING, 1], 
            [DEFINITIONS.OPENING, 3], 
            [DEFINITIONS.DILATION, 7]
        ],
    )
    INFERENCE_OCCLUSIONS: OcclusionDetectionConfig = OcclusionDetectionConfig(
        USE_DEPTH=False,
        USE_DEPTH_GRADS=True,
        USE_FLOW=False,
        AREA_THRESH=300.,
        POSTPROCESSING_OPS_FLOW=[
            (DEFINITIONS.OPENING, 3),
            (DEFINITIONS.CLOSING, 15),
            (DEFINITIONS.DILATION, 15),
        ],
        POSTPROCESSING_OPS_DEPTH_GRADS=[
            (DEFINITIONS.OPENING, 3),
            (DEFINITIONS.CLOSING, 15),
            (DEFINITIONS.DILATION, 9),
        ]
    )
    # The resolution for input images, all the images in the train/validation dataset will be resized to this resolution.
    # RESOLUTION: Tuple[int, int] = (512, 512)
    RENDERER: RendererConfig = RendererConfig()
    
    # Near- and far-planes used during rendering of the pseudo-volume. 
    # Z_FAR should be larger than the input image depth for accurate occlusions maps.
    Z_NEAR: float = 3.0
    Z_FAR: float = 100.0
    
    # Depth alignment
    ALIGN_DEPTH_POLICY: str = "direct"
    ALIGN_DEPTH_MODE: str = "median"
    ALIGN_DEPTH_SCALE_ONLY: bool = True
    ALIGN_DEPTH_MIN_VALID_FRACTION: float = 0.2
    MAX_REL_DEPTH_ERROR_TO_ALIGN: float = 0.0
    
    MIN_MEAN_VALID_PIXELS: float = 0.4
    MIN_MEAN_OCCLUDED_PIXELS: float = 0.01
    MAX_MEAN_OCCLUDED_PIXELS: float = 0.3
    # Minimum and maximum valid depths. If `None`, the near and far planes are used in-place of these.  
    Z_NEAR_GT: Optional[float] = None
    Z_FAR_GT: Optional[float] = None
    OUT_INVALID_LRTB_EDGE_FRACTION: Tuple[float, float, float, float] = field(default_factory=lambda: (0, 0, 0, 0))
    
    ONLY_OCCLUSIONS_VALID: bool = False
        
    # EDGE_DIST_FOR_PROJ_SAMPLE: float = 0.0
    # Name of the depth predictor wrapper class. 
    # Options: ["Metric3D", "UniDepth"]
    DEPTH_PREDICTOR_NAME: str = "UniDepth"
    
    COMPILE_DEPTH_PREDICTOR: bool = False
    COMPILE_REFINER: bool = False
    
    # Setting to `0` disables resampling. 
    # Otherwise invalid invalid views are replaced with resampled ones.
    N_RETRIES_TO_SAMPLE_VALID: int = 0
    TOP_K_NOVEL_VIEWS_TO_KEEP: Optional[int] = None
    
    PSEUDO_VOLUME: PseudoVolumeConfig = PseudoVolumeConfig()
    
    INPUT_CROP_POLICY: str = "center"
    
    NUM_SYNTHETIC_VERSIONS: int = 1
    
    # Arguments for how to build the conditioning image
    DEPTHS_CONDITIONING_INVERSE: bool = False
    SET_OCCLUSIONS_TO_RANDOM_NOISE: bool = False
    CLOSE_CONDITIONING_INVALID: bool = False
    CLOSE_CONDITIONING_INVALID_KERNEL_SIZE: int = 5
    
    CASCADE: CascadeConfig = CascadeConfig()

    DEPTH_GRADS_THRESH: float = 0.5