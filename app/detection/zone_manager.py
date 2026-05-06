from typing import Dict, List
import numpy as np
import supervision as sv

from app.utils.config_loader import AppConfig, ZoneConfig, ZoneType


class ZoneManager:
    def __init__(self, zone_configs: List[ZoneConfig]):
        self._configs: Dict[str, ZoneConfig] = {z.name: z for z in zone_configs}
        self._zones: Dict[str, sv.PolygonZone] = {
            z.name: sv.PolygonZone(polygon=z.polygon)
            for z in zone_configs
        }

    def zones_of_type(self, zone_type: ZoneType) -> List[str]:
        return [name for name, cfg in self._configs.items() if cfg.type == zone_type]

    def check_zone(self, name: str, detections: sv.Detections) -> np.ndarray:
        """Returns boolean mask of which detections are inside the named zone."""
        if len(detections) == 0:
            return np.array([], dtype=bool)
        return self._zones[name].trigger(detections)

    @property
    def configs(self) -> Dict[str, ZoneConfig]:
        return self._configs
