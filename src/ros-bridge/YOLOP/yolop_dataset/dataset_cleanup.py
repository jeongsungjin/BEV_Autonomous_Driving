import os
import cv2

"""dataset_cleanup.py

YOLOP 학습용 데이터셋(images / labels / da_seg / ll_seg)을 대상으로
1. train/valid/test 각 split 폴더에서 네 가지 자료(images, labels, da_seg, ll_seg)의 교집합 prefix만 유지합니다.
2. ll_seg 마스크가 손상된(파일 없음, 읽기 실패, width/height 0) 프레임은 삭제합니다.
   - cv2 읽기 후 None 또는 size 0 체크.

실행 방법:
    python dataset_cleanup.py
필요 패키지: opencv-python
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS = ["train", "valid", "test"]


def _get_prefix(filename: str) -> str:
    """랜덤 postfix(_png.rf...) 및 확장자를 제거하고 frame_XXXX 형태의 prefix 반환"""
    if "_png" in filename:
        return filename.split("_png")[0]
    return os.path.splitext(filename)[0]


def _collect_prefixes(dir_path: str, ext: str):
    """주어진 확장자를 가진 파일들의 prefix 집합 반환"""
    return {
        _get_prefix(f) for f in os.listdir(dir_path) if f.endswith(ext)
    }


def _valid_ll_prefixes(ll_seg_dir: str, prefixes):
    """ll_seg 이미지가 실제로 정상인지 확인하여 유효 prefix 집합 반환"""
    valid = set()
    for p in prefixes:
        path = os.path.join(ll_seg_dir, p + ".png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.size > 0 and img.shape[0] > 0 and img.shape[1] > 0:
            valid.add(p)
    return valid


def _clean_folder(folder: str, ext: str, allowed):
    """folder 내에서 ext 확장자를 가지며 allowed prefix에 속하지 않는 파일 삭제"""
    for f in os.listdir(folder):
        if not f.endswith(ext):
            continue
        if _get_prefix(f) not in allowed:
            os.remove(os.path.join(folder, f))


def cleanup_split(split: str):
    img_dir = os.path.join(BASE_DIR, "images", split)
    label_dir = os.path.join(BASE_DIR, "labels", split)
    da_seg_dir = os.path.join(BASE_DIR, "da_seg", split)
    ll_seg_dir = os.path.join(BASE_DIR, "ll_seg", split)

    # 필요한 네 폴더가 모두 존재하는지 확인
    for d in (img_dir, label_dir, da_seg_dir, ll_seg_dir):
        if not os.path.isdir(d):
            print(f"[WARN] {split}: 폴더가 존재하지 않아 스킵합니다 -> {d}")
            return

    img_p = _collect_prefixes(img_dir, ".jpg")
    label_p = _collect_prefixes(label_dir, ".txt")
    da_p = _collect_prefixes(da_seg_dir, ".png")
    ll_p = _collect_prefixes(ll_seg_dir, ".png")

    intersect = img_p & label_p & da_p & ll_p
    intersect = _valid_ll_prefixes(ll_seg_dir, intersect)

    # 폴더별 정리
    _clean_folder(img_dir, ".jpg", intersect)
    _clean_folder(label_dir, ".txt", intersect)
    _clean_folder(da_seg_dir, ".png", intersect)
    _clean_folder(ll_seg_dir, ".png", intersect)

    print(f"{split}: 최종 남은 샘플 {len(intersect)}개")


def main():
    print("===== YOLOP 데이터셋 정합성 클린업 시작 =====")
    for split in SPLITS:
        cleanup_split(split)
    print("\n✅ 클린업 완료!")


if __name__ == "__main__":
    main() 