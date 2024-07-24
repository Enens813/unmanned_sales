import os
import shutil
import math

# 경로 설정
src_dir = "/home/aicompetition12/Datasets/1.competition_trainset/1_dataset/"
img_train_dir = "/home/aicompetition12/Datasets/images/train/"
label_train_dir = "/home/aicompetition12/Datasets/labels/train/"
img_val_dir = "/home/aicompetition12/Datasets/images/val/"
label_val_dir = "/home/aicompetition12/Datasets/labels/val/"

# 같은 이름을 가진 파일 목록 추출 (jpg 확장자 제거)
jpg_files = [f[:-4] for f in os.listdir(src_dir) if f.endswith('.jpg')]

# 같은 이름의 txt 파일이 있는지 확인하고, 있다면 리스트에 추가
paired_files = [base_name for base_name in jpg_files if os.path.exists(os.path.join(src_dir, base_name + '.txt'))]

# 총 파일 개수 계산
total_files = len(paired_files)
move_files = math.floor(total_files * 0.8)

# 80% 파일을 이동할 대상 목록
files_to_move = paired_files[:move_files]

# 나머지 20% 파일을 이동할 대상 목록
files_to_move_val = paired_files[move_files:]

# 학습용으로 80% 파일 이동
for base_name in files_to_move:
    shutil.move(os.path.join(src_dir, base_name + '.jpg'), os.path.join(img_train_dir, base_name + '.jpg'))
    shutil.move(os.path.join(src_dir, base_name + '.txt'), os.path.join(label_train_dir, base_name + '.txt'))

# 검증용으로 나머지 20% 파일 이동
for base_name in files_to_move_val:
    shutil.move(os.path.join(src_dir, base_name + '.jpg'), os.path.join(img_val_dir, base_name + '.jpg'))
    shutil.move(os.path.join(src_dir, base_name + '.txt'), os.path.join(label_val_dir, base_name + '.txt'))

