
git clone https://github.com/Enens813/unmanned_sales.git
export PATH="$HOME/.local/bin:$PATH"
pip install -r requirements.txt
source ~/.bashrc

설치된 unmanned_sales/data/에다가
dataset.yaml파일을 새로 만듬
파일 내용:

train: /home/aicompetition12/Datasets/images/train
val: /home/aicompetition12/Datasets/images/val

nc: 60  # number of classes
names: [
  'aunt_jemima_original_syrup',
  'band_aid_clear_strips', 
  'bumblebee_albacore', 
  'cholula_chipotle_hot_sauce', 
  'crayola_24_crayons', 
  'hersheys_cocoa', 
  'honey_bunches_of_oats_honey_roasted', 
  'honey_bunches_of_oats_with_almonds', 
  'hunts_sauce', 
  'listerine_green', 
  'mahatma_rice', 
  'white_rain_body_wash', 
  'pringles_bbq', 
  'cheeze_it', 
  'hersheys_bar', 
  'redbull', 
  'mom_to_mom_sweet_potato_corn_apple', 
  'a1_steak_sauce', 
  'jif_creamy_peanut_butter', 
  'cinnamon_toast_crunch', 
  'arm_hammer_baking_soda', 
  'dr_pepper', 
  'haribo_gold_bears_gummi_candy', 
  'bulls_eye_bbq_sauce_original', 
  'reeses_pieces', 
  'clif_crunch_peanut_butter', 
  'mom_to_mom_butternut_squash_pear',
  'pop_tarts_strawberry', 
  'quaker_big_chewy_chocolate_chip', 
  'spam', 
  'coffee_mate_french_vanilla', 
  'pepperidge_farm_milk_chocolate_macadamia_cookies', 
  'kitkat_king_size', 
  'snickers', 
  'toblerone_milk_chocolate', 
  'clif_z_bar_chocolate_chip', 
  'nature_valley_crunchy_oats_n_honey', 
  'ritz_crackers', 
  'palmolive_orange',
  'crystal_hot_sauce', 
  'tapatio_hot_sauce', 
  'nabisco_nilla_wafers', 
  'pepperidge_farm_milano_cookies_double_chocolate', 
  'campbells_chicken_noodle_soup', 
  'frappuccino_coffee', 
  'chewy_dips_chocolate_chip', 
  'chewy_dips_peanut_butter', 
  'nature_valley_fruit_and_nut', 
  'cheerios', 
  'lindt_excellence_cocoa_dark_chocolate', 
  'hersheys_symphony', 
  'campbells_chunky_classic_chicken_noodle', 
  'martinellis_apple_juice', 
  'dove_pink', 
  'dove_white', 
  'david_sunflower_seeds', 
  'monster_energy', 
  'act_ii_butter_lovers_popcorn', 
  'coca_cola_glass_bottle', 
  'twix' 
]


# 디렉토리 생성
mkdir -p /home/aicompetition12/Datasets/images/train
mkdir -p /home/aicompetition12/Datasets/labels/train

# 이미지 파일 이동
find /home/aicompetition12/Datasets/1.competition_trainset/1_dataset/ -maxdepth 1 -type f -name '*.jpg' | head -n $(find /home/aicompetition12/Datasets/1.competition_trainset/1_dataset/ -maxdepth 1 -type f -name '*.jpg' | wc -l | awk '{print int($1*0.8)}') | xargs -I {} mv {} /home/aicompetition12/Datasets/images/train/

# 라벨 파일 이동
find /home/aicompetition12/Datasets/1.competition_trainset/1_dataset/ -maxdepth 1 -type f -name '*.txt' | head -n $(find /home/aicompetition12/Datasets/1.competition_trainset/1_dataset/ -maxdepth 1 -type f -name '*.txt' | wc -l | awk '{print int($1*0.8)}') | xargs -I {} mv {} /home/aicompetition12/Datasets/labels/train/


# 디렉토리 생성
mkdir -p /home/aicompetition12/Datasets/images/val
mkdir -p /home/aicompetition12/Datasets/labels/val

# 이미지 파일 이동
find /home/aicompetition12/Datasets/1.competition_trainset/1_dataset/ -maxdepth 1 -type f -name '*.jpg' | tail -n $(find /home/aicompetition12/Datasets/1.competition_trainset/1_dataset/ -maxdepth 1 -type f -name '*.jpg' | wc -l | awk '{print int($1*0.2)}') | xargs -I {} mv {} /home/aicompetition12/Datasets/images/val/

# 라벨 파일 이동
find /home/aicompetition12/Datasets/1.competition_trainset/1_dataset/ -maxdepth 1 -type f -name '*.txt' | tail -n $(find /home/aicompetition12/Datasets/1.competition_trainset/1_dataset/ -maxdepth 1 -type f -name '*.txt' | wc -l | awk '{print int($1*0.2)}') | xargs -I {} mv {} /home/aicompetition12/Datasets/labels/val/

cd unmanned_sales
ssai_agpu-g=1

python train.py --img 640 --batch 16 --epochs 100 --data data/dataset.yaml --cfg models/yolov5s.yaml --weights '' --name my_yolov5_model --device 0

########################
#1. 검증 데이터로 모델 평가
#검증 데이터셋을 사용하여 모델의 성능을 평가할 수 있습니다. 다음 명령어를 사용하세요:
python val.py --weights runs/train/my_yolov5_model/weights/best.pt --data data/dataset.yaml --img 640


#2. 모델로 새로운 이미지 추론
#모델을 사용하여 새로운 이미지에 대해 예측을 수행할 수 있습니다. 다음 명령어를 사용하세요:
python detect.py --weights runs/train/my_yolov5_model/weights/best.pt --img 640 --conf 0.25 --source /path/to/your/images


#3. 추론 결과 확인
#추론 결과는 runs/detect/exp 디렉토리에 저장됩니다. 이 디렉토리에서 결과 이미지를 확인할 수 있습니다.



##########################
현재(0708)모델은 precision은 매우 낮고, recall은 epoch 50정도에선 1에 가까움. -> 원인파악 필요


