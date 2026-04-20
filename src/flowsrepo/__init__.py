from typing import Dict, Type
from .base_flow import BaseFlow
from .earth_flow import EarthFlow
from .dragons_flow import DragonsFlow
from .satellite_flow import SatelliteFlow
from .meltingman_flow import MeltingManFlow
from .glass_flow import GlassFlow
from .mycar_flow import MyCarFlow
from .new_car_flow import CarFlow
from .smoke_flow import SmokeFlow
from .fire_flow import FireFlow
from .flood_flow import FloodFlow
from .canal_flood_flow import CanalFloodFlow

example_registry: Dict[str, Type[BaseFlow]] = {
    "earth": EarthFlow,
    "dragons": DragonsFlow,
    "satellite": SatelliteFlow,
    "meltingman": MeltingManFlow,
    "glass": GlassFlow,
    "mycar": MyCarFlow,
    "newcar": CarFlow,
    "smoke": SmokeFlow,
    "fire": FireFlow,
    "flood": FloodFlow,
    "canal_flood": CanalFloodFlow,
}
