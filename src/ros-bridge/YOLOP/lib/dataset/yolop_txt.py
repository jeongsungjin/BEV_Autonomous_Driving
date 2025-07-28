from pathlib import Path
import numpy as np
import os
from tqdm import tqdm

from .AutoDriveDataset import AutoDriveDataset

class YoloTxtDataset(AutoDriveDataset):
    """YOLOP 학습용으로 정리된 (images / labels / da_seg / ll_seg) 폴더 구조를
    그대로 읽어들이는 Dataset.
    - 탐지 라벨: YOLO txt (cls cx cy w h) 0~1 정규화 형식
    - 주행가능영역/차선 마스크: png
    """

    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        print("building YOLOP txt dataset db …")
        gt_db = []
        for mask_path in tqdm(list(self.mask_list)):
            mask_path = str(mask_path)  # da_seg mask (.png)
            # 매칭 경로 계산
            img_path = mask_path.replace(str(self.mask_root), str(self.img_root)).replace(".png", ".jpg")
            lane_path = mask_path.replace(str(self.mask_root), str(self.lane_root))
            label_path = mask_path.replace(str(self.mask_root), str(self.label_root)).replace(".png", ".txt")

            # YOLO txt 라벨 읽기 (없으면 빈 array)
            if os.path.isfile(label_path):
                with open(label_path, "r") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                gt = np.zeros((len(lines), 5), dtype=np.float32)
                for idx, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) != 5:
                        continue  # 잘못된 라인 스킵
                    gt[idx] = np.array(list(map(float, parts)), dtype=np.float32)
            else:
                gt = np.zeros((0, 5), dtype=np.float32)

            rec = [{
                "image": img_path,
                "label": gt,
                "mask": mask_path,
                "lane": lane_path
            }]
            gt_db += rec
        print("db build finish, samples:", len(gt_db))
        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # 필요 시 구현
        pass 