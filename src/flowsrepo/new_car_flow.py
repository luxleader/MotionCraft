import torch
import PIL.Image as Image
import numpy as np
from .base_flow import BaseFlow

class CarFlow(BaseFlow):
    def __init__(self, N, mass=1000, inertia=1500, wheelbase=2.5):
        super().__init__(N=N)
        self.mass = mass
        self.inertia = inertia
        self.wheelbase = wheelbase

        u = torch.linspace(-1, 1, N)
        v = torch.linspace(-1, 1, N)
        u, v = torch.meshgrid(u, v, indexing="xy")
        
        # 初始车辆位置
        self.position_at_time_tau = torch.stack([u, v], dim=-1)
        self.orientation = torch.zeros((N, N))  # 初始角度

        self.pos_prompt = "A close-up of a vehicle on a road."

    def get_spatial_eta(self, t):
        flow = self.get_flow(t)
        mask = (flow.abs() > 1).any(dim=-1).reshape(1, 1, self.N, self.N).float()
        return mask

    def get_default_image(self) -> torch.Tensor:
        # 这里假设有车辆的默认图像
        image = Image.open(f"{self.this_path}/data/newcar/newcar.png")
        return image

    def get_default_framesteps(self) -> torch.Tensor:
        img_fraction = 1/64
        return torch.full((64,), img_fraction*2)

    def get_flow(self, tau):
        # 模拟车辆的平移和旋转
        position_at_time_zero = self.position_at_time_tau - torch.tensor([0, tau], device=self.position_at_time_tau.device)

        # 车辆的旋转效果
        rotation_matrix = torch.tensor(
            [
                [torch.cos(tau), -torch.sin(tau)],
                [torch.sin(tau), torch.cos(tau)],
            ],
            device=self.position_at_time_tau.device,
            dtype=self.position_at_time_tau.dtype,
        )
        position_at_time_zero_rotated = torch.einsum(
            "ij,nmj->nmi", rotation_matrix.T, position_at_time_zero
        )
        return position_at_time_zero_rotated

