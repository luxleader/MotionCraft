import os
import glob
from PIL import Image

# ====== 配置 ======
input_folder = "../output/MotionCraft_bilinear/1773220406.2896607_flood"
output_gif = "flood_preview_1.gif"
duration = 400 # 每帧 200ms = 5fps
# =================

# 寻找所有 _2.png
search_pattern = os.path.join(input_folder, "frame_*_2.png")
image_paths = sorted(glob.glob(search_pattern))

if not image_paths:
    print("没找到图片！")
    exit()

print(f"找到 {len(image_paths)} 张图片，正在生成 GIF...")

# 读取图片
frames = [Image.open(p) for p in image_paths]

# 保存为 GIF
# save_all=True: 保存多帧
# append_images: 后续帧列表
# loop=0: 无限循环
frames[0].save(
    output_gif,
    format='GIF',
    save_all=True,
    append_images=frames[1:],
    duration=duration,
    loop=0
)

print(f"成功！GIF 已保存为: {output_gif}")