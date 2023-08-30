from itertools import product

import numpy as np
from torch import nn

from kornia.contrib.face_detection import YuFaceDetectNet
from kornia.geometry.bbox import nms as nms_kornia

__all__ = ["LPD", "YuLPD"]

url: str = "https://github.com/kornia/data/raw/main/yunet_final.pth"


class LPD(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int] = (320, 240),
        confidence_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        keep_top_k: int = 750,
        backend_id: int = 0,
        target_id: int = 0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.backend_id = backend_id
        self.target_id = target_id
        self.config = {
            'name': 'YuFaceDetectNet',
            'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
            'steps': [8, 16, 32, 64],
            'variance': [0.1, 0.2]
            # 'clip': False,
        }
        # self.min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        # self.steps = [8, 16, 32, 64]
        # self.variance = [0.1, 0.2]
        # self.clip = False
        self.model = YuFaceDetectNet('test', pretrained=True)
        self.nms = nms_kornia


class YuLPD(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self._priorGen()

    def _priorGen(self):
        w, h = self.input_size
        feature_map_2th = [int(int((h + 1) / 2) / 2), int(int((w + 1) / 2) / 2)]
        feature_map_3th = [int(feature_map_2th[0] / 2), int(feature_map_2th[1] / 2)]
        feature_map_4th = [int(feature_map_3th[0] / 2), int(feature_map_3th[1] / 2)]
        feature_map_5th = [int(feature_map_4th[0] / 2), int(feature_map_4th[1] / 2)]
        feature_map_6th = [int(feature_map_5th[0] / 2), int(feature_map_5th[1] / 2)]

        feature_maps = [feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th]

        priors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):  # i->h, j->w
                for min_size in min_sizes:
                    s_kx = min_size / w
                    s_ky = min_size / h

                    cx = (j + 0.5) * self.steps[k] / w
                    cy = (i + 0.5) * self.steps[k] / h

                    priors.append([cx, cy, s_kx, s_ky])
        self.priors = np.array(priors, dtype=np.float32)
