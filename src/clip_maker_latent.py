from flowsrepo import example_registry
from tqdm import tqdm
import torch
import os
import time
import argparse
import numpy as np
from PIL import Image
from DiffusersUtils import (
    StableDiffusionManager,
    get_attention_processor,
    get_attention_processor_from_pattern,
)

args = argparse.ArgumentParser()
args.add_argument("--tag", type=str, default="MotionCraft")
args.add_argument(
    "--example",
    type=str,
    required=True,
)
args.add_argument(
    "--interpolationmode",
    type=str,
    choices=["bilinear", "nearest", "bicubic"],
    default="nearest",
)
args.add_argument("--device", type=str, default="cuda:0")
args.add_argument("--tau", type=int, default=400)
args.add_argument("--num_inference_steps", type=int, default=200)
args.add_argument("--guidance_scale", type=float, default=7.5)
args.add_argument(
    "--crossframeattention_pattern", type=str, default="[[0,0],[1,1],[0,1]]"
)
args.add_argument("--invert", action=argparse.BooleanOptionalAction, default=True)
args.add_argument("--spatialeta", action=argparse.BooleanOptionalAction, default=True)
args.add_argument("--SDXL", action=argparse.BooleanOptionalAction, default=False)

args = args.parse_args()

##### Parameters
torch.manual_seed(11)
device = args.device
N = 128 if args.SDXL else 64
image_warper = example_registry[args.example](N=N)
prompt, negative_prompt = image_warper.get_default_prompt()

# prompt = ""
# clean |-------------------|> noise
# 0     |----tau------------|> 1000
# 0     |---*-----*------*--|> * num inference steps
num_inference_steps = args.num_inference_steps
guidance_scale = args.guidance_scale
folder_path = f"output/{args.tag}/{time.time()}_{args.example}"
os.makedirs(folder_path, exist_ok=True)
SDM = StableDiffusionManager(device, args.tau, SDXL=args.SDXL)

single_attention_processor = get_attention_processor(
    video_length=1, crossframe_attn="disabled", should_record_history=False
)

cross_attention_processor = get_attention_processor_from_pattern(
    args.crossframeattention_pattern
)

SDM.pipeline.unet.set_attn_processor(single_attention_processor)
z_tau_orig = image_warper.get_ztau_orig(SDM, num_inference_steps)
z_tau = z_tau_orig.clone()

framesteps = image_warper.get_default_framesteps()
for f, (framestep) in enumerate(tqdm(framesteps)):
    warped_latent = image_warper.warp(
        t=framestep,
        previous_frame=z_tau,
        original_frame=z_tau_orig,
        mode=args.interpolationmode,
    )
    if args.spatialeta:
        spatial_eta = image_warper.get_spatial_eta(t=framestep)
        spatial_eta = spatial_eta.repeat(cross_attention_processor.video_length, 1, 1, 1)
        spatial_eta[:-1] = 0.0
        spatial_eta = spatial_eta.to(warped_latent.device, warped_latent.dtype)
        print(spatial_eta.abs().sum())
    else:
        spatial_eta = 0.0

    if cross_attention_processor.video_length == 2:
        z_pair = torch.cat([z_tau_orig, warped_latent], dim=0)
    else:
        z_pair = torch.cat([z_tau_orig, z_tau, warped_latent], dim=0)

    # Generate with cross attention
    SDM.pipeline.unet.set_attn_processor(cross_attention_processor)
    generated_latents = SDM.partial_generation(
        z=z_pair,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        eta=spatial_eta,
        guidance_scale=guidance_scale,
        negative_prompt=[prompt] * (len(z_pair) - 1) + [negative_prompt],
    )

    generated_frames = SDM.latent_to_image(generated_latents)
    
    # ===================== 方案1：图像级混合 (改进版 - 不需要 cv2) =====================
    # 检查是否为 FloodFlow（需要保护非洪水区）
    if hasattr(image_warper, 'flood_seq') and f < len(image_warper.flood_seq):
        # 获取当前帧的洪水掩码
        flooded_hard = image_warper.flood_seq[f].to(device)  # (1,1,N,N)
        
        # 获取原始图像（初始帧）
        original_image_pil = image_warper.get_default_image()
        
        # 遍历所有生成的帧进行混合
        for i, generated_img in enumerate(generated_frames):
            # 调整原始图像尺寸以匹配生成图像
            if original_image_pil.size != generated_img.size:
                original_img_resized = original_image_pil.resize(
                    generated_img.size, Image.LANCZOS
                )
            else:
                original_img_resized = original_image_pil
            
            # 转换为 numpy 数组
            generated_np = np.array(generated_img)
            original_np = np.array(original_img_resized)
            
            # 获取图像尺寸
            H, W = generated_np.shape[:2]
            
            # ✅ 使用 PIL 替代 cv2.resize
            flooded_mask_np = flooded_hard[0, 0].cpu().numpy().astype(np.float32)
            flooded_mask_pil = Image.fromarray(
                (flooded_mask_np * 255).astype(np.uint8), mode='L'
            )
            flooded_mask_resized_pil = flooded_mask_pil.resize(
                (W, H), Image.BILINEAR
            )
            flooded_mask_resized = np.array(flooded_mask_resized_pil).astype(np.float32) / 255.0
            
            # 扩展掩码为 3 通道（RGB）
            flooded_mask_3d = np.stack(
                [flooded_mask_resized] * 3, axis=-1
            )  # (H, W, 3)
            
            # 混合图像：洪水区使用生成图像，非洪水区使用原始图像
            blended_np = (
                generated_np.astype(np.float32) * flooded_mask_3d +
                original_np.astype(np.float32) * (1 - flooded_mask_3d)
            ).astype(np.uint8)
            
            # 转换回 PIL 图像
            blended_img = Image.fromarray(blended_np)
            
            # 保存混合后的图像
            blended_img.save(f"{folder_path}/frame_{f:03d}_{i}.png")
            
            print(f"  Frame {f:03d}_{i}: 洪水区覆盖率 = {flooded_mask_resized.mean():.2%}")
    else:
        # 非 FloodFlow 场景，直接保存
        for i, img in enumerate(generated_frames):
            img.save(f"{folder_path}/frame_{f:03d}_{i}.png")
    # =====================================================================

    # Update the z0 for the next iteration
    if not args.invert:
        print("Warning: not inverting")
        z_tau = warped_latent
    else:
        SDM.pipeline.unet.set_attn_processor(single_attention_processor)
        z_tau = SDM.partial_inversion(
            z=generated_latents[[-1]],
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
        )