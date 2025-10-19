import sys
sys.path.append(".")

from scripts.inference_setup import *
import torch
from utils.array_ops import to_tensor_unsqueeze, to

from utils.plotting import color_tensor
from utils.plotting import render_profile
import sys
sys.path.append(".")

import torch
from utils.array_ops import to_tensor_unsqueeze, to
from torchvision.utils import save_image
from utils.utils import invert_depth
from bts.gt_synthesis.gt_synthesis import Outputs, Data
from bts.ignite_evaluation.evaluator import initialize, InferenceWrapper

def main():
    dry_run = False

    indices = [
        54, 84, 119
        # 0, 100, 200, 300,
        # 28, 84, 119, 287, 374, 385,
        # 42, 54, 112, 374, 398,
    ]

    config = load_and_setup_config(
        config_name="exp_recon_full", 
        # config_name="eval_bts_lidar_occ", 
    )
    dataset, out_path = setup_task(config, "figures/bev")
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP201_backend-nccl-4_run_0//training_checkpoint_29000.pt"
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP203_l2_backend-nccl-4_run_0/training_checkpoint_29000.pt"
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP204_l2_var_backend-nccl-4_run_0/training_checkpoint_29000.pt"
    # config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP205_weight_guided_occl_backend-nccl-4_run_0/training_checkpoint_29000.pt"
    config.BTS.RESUME_FROM = "/home/stud/wph/storage/user/BTS/out/recon/EXP334_backend-gloo-4_run_0/best_model.pt"
    model_name = "full"

    suffix = config.NAME
    
    config.BTS.DATA.VIS_VOLUME.X_RANGE = [-9, 9]
    config.BTS.DATA.VIS_VOLUME.Z_RANGE = [3, 21]
    
    # model = InferenceWrapper.from_conf(config, refine_output=True)
    # model.to(device)
    
    config.EVAL_OCCUPANCY.MODE = "recon"
    # config.SYNTHETIC_GT.DEPTH_PREDICTOR_NAME = "UniDepth"
    match config.EVAL_OCCUPANCY.MODE:
        case "depth_pred":
            IS_CONTROLNET = True
            wrapper: InferenceWrapper = initialize(config, refine_output=False)
            net = wrapper.gt_synthesizer.renderer.net
            forward_fn = lambda d: wrapper(d, forward_type="controlnet_novel_view")
            suffix = config.SYNTHETIC_GT.DEPTH_PREDICTOR_NAME
            net = wrapper.gt_synthesizer.renderer.net
        case "cascade":
            IS_CONTROLNET = True
            wrapper: InferenceWrapper = initialize(config, refine_output=True)
            net = wrapper.gt_synthesizer.renderer.net
            forward_fn = lambda d: wrapper(d, forward_type="controlnet_cascade")
            suffix = "cascade_" + config.NAME
        case "recon":
            wrapper: InferenceWrapper = initialize(config, refine_output=False)
            net = wrapper.renderer.net
            forward_fn = lambda d: wrapper(d, forward_type="recon")
            suffix = model_name if model_name else "recon_" + config.NAME
            net = wrapper.renderer.net
    wrapper.to(device)
    
    with torch.no_grad():
        for idx in indices:
            data = dataset[idx]
            data = to_tensor_unsqueeze(data)
            data = to(data, device)
            data = forward_fn(data)
            
            if "out_synth" in data:
                out: Outputs = data["out_synth"]
                data_synth: Data = data["data_synth"]
            elif "out_synth_l" in data:
                out: Outputs = data["out_synth_l"][-1]
                data_synth: Data = data["data_synth_l"][-1]
            # else:
            #     depth = data["coarse"][0]["depth"][0, 0]
            #     depth = invert_depth(depth, config.BTS.MODEL_CONF.z_near, config.BTS.MODEL_CONF.z_far)
            #     depth = color_tensor(depth, "magma", norm=False).permute(2, 0, 1)
            
            images = torch.stack(data["imgs"], dim=1).to(device)[:, :1]
            image = images[0, 0] * .5 + .5
            
            # image = torch.concat([image, depth], dim=1)

            profile = render_profile(net, config.BTS.DATA.VIS_VOLUME, config.BTS.DATA.CAM_INCL_ADJUST.to(device))[0].permute(2, 0, 1)

            if config.BTS.MODEL_CONF.OUTPUT_UNCERTAINTY:
                    profile_uncert = render_profile(
                        net, 
                        config.BTS.DATA.VIS_VOLUME, 
                        config.BTS.DATA.CAM_INCL_ADJUST.to(device),
                        mode="uncert",
                        color_profile="viridis",
                    )[0].permute(2, 0, 1)
                    profile = torch.cat([profile, profile_uncert], dim=2)

            if not dry_run:
                filepath = os.path.join(out_path, f"profile_{idx:03d}_{suffix}.png")
                img_path = os.path.join(out_path, f"input_{idx:03d}.png")
                print(f"Saving to {img_path} and {filepath}")
                save_image(image, img_path)
                # save_image(depth, f"{filepath}_in_d.png")
                save_image(profile, filepath)

    print("Completed.")


if __name__ == '__main__':
    main()
