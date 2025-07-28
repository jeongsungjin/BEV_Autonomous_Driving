import os
import shutil
import random
import re

# 원본 어노테이션 폴더
DET_ANN_IMG = 'det_annotations/train'
DA_ANN_IMG = 'da_seg_annotations/train'
LL_ANN_IMG = 'll_seg_annotations/train'

# 변환된 폴더
IMG_DIR = 'images'
LABEL_DIR = 'labels'
DA_SEG_DIR = 'da_seg'
LL_SEG_DIR = 'll_seg'

random.seed(42)

def get_real_prefix(filename):
    # frame_001674_png.rf.abcdef123456.jpg → frame_001674
    m = re.match(r'(frame_\d+)', filename)
    return m.group(1) if m else os.path.splitext(filename)[0]

def get_prefixes(folder, ext):
    return set([get_real_prefix(f) for f in os.listdir(folder) if f.endswith(ext)])

def copy_file(src_dir, dst_dir, prefix, exts):
    for ext in exts:
        # src 파일명 패턴: frame_XXXX(_png.rf.XXXXXX).ext
        for f in os.listdir(src_dir):
            if f.endswith(ext) and get_real_prefix(f) == prefix:
                src = os.path.join(src_dir, f)
                dst = os.path.join(dst_dir, prefix + ext)
                if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy(src, dst)
                break

def ensure_dirs(base, splits):
    for split in splits:
        os.makedirs(os.path.join(base, split), exist_ok=True)

# 1. 교집합 prefix 구하기
det_prefix = get_prefixes(DET_ANN_IMG, '.jpg')
da_prefix = get_prefixes(DA_ANN_IMG, '.jpg')
ll_prefix = get_prefixes(LL_ANN_IMG, '.jpg')
common_prefix = sorted(list(det_prefix & da_prefix & ll_prefix))
print(f'3개 어노테이션 교집합 prefix 개수: {len(common_prefix)}')

# 2. 랜덤 셔플 후 8:1:1로 분할
random.shuffle(common_prefix)
n = len(common_prefix)
train_cut = int(n * 0.8)
valid_cut = int(n * 0.9)
train_prefix = common_prefix[:train_cut]
valid_prefix = common_prefix[train_cut:valid_cut]
test_prefix = common_prefix[valid_cut:]
splits = [('train', train_prefix), ('valid', valid_prefix), ('test', test_prefix)]

# 3. 각 split별로 복사
def copy_split(prefixes, split):
    # images
    ensure_dirs(IMG_DIR, [split])
    for p in prefixes:
        copy_file(DET_ANN_IMG, os.path.join(IMG_DIR, split), p, ['.jpg'])
    # labels
    ensure_dirs(LABEL_DIR, [split])
    for p in prefixes:
        copy_file('det_annotations/train', os.path.join(LABEL_DIR, split), p, ['.txt'])
    # da_seg
    ensure_dirs(DA_SEG_DIR, [split])
    for p in prefixes:
        copy_file('da_seg/train', os.path.join(DA_SEG_DIR, split), p, ['.png'])
    # ll_seg
    ensure_dirs(LL_SEG_DIR, [split])
    for p in prefixes:
        copy_file('ll_seg/train', os.path.join(LL_SEG_DIR, split), p, ['.png'])

for split, prefixes in splits:
    print(f'{split}: {len(prefixes)}개')
    copy_split(prefixes, split)

print('교집합 기반 split 및 복사 완료! (프레임 번호 기준 이름 통일)')
