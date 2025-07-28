import os
import shutil
import cv2
import numpy as np
from pycocotools.coco import COCO

# split 목록
splits = ['train', 'valid', 'test']

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DET_ANN_DIR = os.path.join(BASE_DIR, 'det_annotations')
DA_ANN_DIR = os.path.join(BASE_DIR, 'da_seg_annotations')
LL_ANN_DIR = os.path.join(BASE_DIR, 'll_seg_annotations')
IMG_OUT_DIR = os.path.join(BASE_DIR, 'images')
LABEL_OUT_DIR = os.path.join(BASE_DIR, 'labels')
DA_MASK_OUT_DIR = os.path.join(BASE_DIR, 'da_seg')
LL_MASK_OUT_DIR = os.path.join(BASE_DIR, 'll_seg')

for split in splits:
    print(f'\n===== {split.upper()} 변환 시작 =====')
    # 1. Detection: 라벨/이미지 복사
    det_img_dir = os.path.join(DET_ANN_DIR, split)
    label_out_dir = os.path.join(LABEL_OUT_DIR, split)
    img_out_dir = os.path.join(IMG_OUT_DIR, split)
    os.makedirs(label_out_dir, exist_ok=True)
    os.makedirs(img_out_dir, exist_ok=True)
    # .txt (라벨)
    for f in os.listdir(det_img_dir):
        if f.endswith('.txt'):
            shutil.copy(os.path.join(det_img_dir, f), os.path.join(label_out_dir, f))
    # .jpg (이미지)
    for f in os.listdir(det_img_dir):
        if f.endswith('.jpg'):
            shutil.copy(os.path.join(det_img_dir, f), os.path.join(img_out_dir, f))
    print(f'Detection 라벨/이미지 복사 완료')

    # 2. Drivable Area Segmentation: 마스크 변환
    da_ann_path = os.path.join(DA_ANN_DIR, split, '_annotations.coco.json')
    da_src_img_dir = os.path.join(DA_ANN_DIR, split)
    da_mask_out_dir = os.path.join(DA_MASK_OUT_DIR, split)
    os.makedirs(da_mask_out_dir, exist_ok=True)
    if os.path.exists(da_ann_path):
        coco = COCO(da_ann_path)
        for img in coco.imgs.values():
            img_id = img['id']
            file_name = img['file_name']
            base = file_name.split('_png')[0]
            mask_name = f"{base}.png"
            img_name = f"{base}.jpg"
            # 마스크 생성
            mask = np.zeros((img['height'], img['width']), dtype=np.uint8)
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                m = coco.annToMask(ann)
                mask = np.maximum(mask, m * 255)
            cv2.imwrite(os.path.join(da_mask_out_dir, mask_name), mask)
            # 이미지 복사 (이름 변경)
            src_img_path = os.path.join(da_src_img_dir, file_name)
            dst_img_path = os.path.join(img_out_dir, img_name)
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)
        print(f'Drivable Area 마스크/이미지 변환 완료')
    else:
        print(f'Drivable Area annotation 파일 없음: {da_ann_path}')

    # 3. Lane Line Segmentation: 마스크 변환
    ll_ann_path = os.path.join(LL_ANN_DIR, split, '_annotations.coco.json')
    ll_src_img_dir = os.path.join(LL_ANN_DIR, split)
    ll_mask_out_dir = os.path.join(LL_MASK_OUT_DIR, split)
    os.makedirs(ll_mask_out_dir, exist_ok=True)
    if os.path.exists(ll_ann_path):
        coco = COCO(ll_ann_path)
        for img in coco.imgs.values():
            img_id = img['id']
            file_name = img['file_name']
            base = file_name.split('_png')[0]
            mask_name = f"{base}.png"
            img_name = f"{base}.jpg"
            # 마스크 생성
            mask = np.zeros((img['height'], img['width']), dtype=np.uint8)
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                m = coco.annToMask(ann)
                mask = np.maximum(mask, m * 255)
            cv2.imwrite(os.path.join(ll_mask_out_dir, mask_name), mask)
            # 이미지 복사 (이름 변경)
            src_img_path = os.path.join(ll_src_img_dir, file_name)
            dst_img_path = os.path.join(img_out_dir, img_name)
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)
        print(f'Lane Line 마스크/이미지 변환 완료')
    else:
        print(f'Lane Line annotation 파일 없음: {ll_ann_path}')

print('\n✅ 전체 변환 완료! YOLOP 학습에 바로 사용할 수 있습니다.') 