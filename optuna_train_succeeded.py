import optuna
import os
import time

def objective(trial):
    # Define hyperparameters to tune
    lrf = trial.suggest_loguniform('lrf', 0.01, 1)
    momentum = trial.suggest_float('momentum', 0.6, 0.98)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    box = trial.suggest_float('box', 0.02, 0.2)
    cls = trial.suggest_float('cls', 0.2, 4.0)
    obj = trial.suggest_float('obj', 0.2, 4.0)
    iou_t = trial.suggest_float('iou_t', 0.1, 0.7)
    fl_gamma = trial.suggest_float('fl_gamma', 0.0, 2.0)
    # Define fixed hyperparameters
    mosaic = 1.0
    copy_paste = 0.0
    hsv_h = 0.015
    hsv_s = 0.7
    hsv_v = 0.4
    degrees = 0.0
    translate = 0.1
    scale = 0.9
    shear = 0.0
    perspective = 0.0
    flipud = 0.0
    fliplr = 0.5
    mixup = 0.1

    # Write the hyperparameters to the hyp_optuna.yaml file
    with open('data/hyps/hyp_optuna.yaml', 'w') as f:
        f.write(f"lr0: {lrf}\n")
        f.write(f"lrf: 0.1\n")
        f.write(f"momentum: {momentum}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"warmup_epochs: 3.0\n")
        f.write(f"warmup_momentum: 0.8\n")
        f.write(f"warmup_bias_lr: 0.1\n")
        f.write(f"box: {box}\n")
        f.write(f"cls: {cls}\n")
        f.write(f"cls_pw: 1.0\n")
        f.write(f"obj: {obj}\n")
        f.write(f"obj_pw: 1.0\n")
        f.write(f"iou_t: {iou_t}\n")
        f.write(f"anchor_t: 4.0\n")
        f.write(f"fl_gamma: {fl_gamma}\n")
        f.write(f"mosaic: {mosaic}\n")
        f.write(f"copy_paste: {copy_paste}\n")
        f.write(f"hsv_h: {hsv_h}\n")
        f.write(f"hsv_s: {hsv_s}\n")
        f.write(f"hsv_v: {hsv_v}\n")
        f.write(f"degrees: {degrees}\n")
        f.write(f"translate: {translate}\n")
        f.write(f"scale: {scale}\n")
        f.write(f"shear: {shear}\n")
        f.write(f"perspective: {perspective}\n")
        f.write(f"flipud: {flipud}\n")
        f.write(f"fliplr: {fliplr}\n")
        f.write(f"mixup: {mixup}\n")

    # Construct command to run YOLOv5 training script
    cmd = f"""
    python train.py --img 640 --batch 16 --epochs 50 --data data/dataset.yaml --cfg models/yolov5m.yaml \
    --hyp data/hyps/hyp_optuna.yaml --weights '' --name optuna_model --device 0
    """
        
    # Execute command
    os.system(cmd)

    # Evaluate the model (assumes the best model's results are saved to runs/train/optuna_model/results.txt)
    results_path = 'runs/train/optuna_model/results.txt'
    start_time = time.time()
    timeout = 600  # Timeout after 10 minutes

    while not os.path.exists(results_path):
        time.sleep(10)  # Check every 10 seconds
        if time.time() - start_time > timeout:
            raise Exception(f"Training did not complete within {timeout} seconds")

    with open(results_path) as f:
        results = f.read()
    
    # Parse results (this part depends on how the results are formatted in the file)
    metrics = results.split()
    mAP_0_5 = float(metrics[-1])  # Example: last metric is mAP@0.5

    return mAP_0_5

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Save the best hyperparameters to a YAML file
best_params = study.best_trial.params
with open('data/hyps/hyp_optuna.yaml', 'w') as f:
    f.write(f"lr0: {best_params['lrf']}\n")
    f.write(f"lrf: 0.1\n")
    f.write(f"momentum: {best_params['momentum']}\n")
    f.write(f"weight_decay: {best_params['weight_decay']}\n")
    f.write(f"warmup_epochs: 3.0\n")
    f.write(f"warmup_momentum: 0.8\n")
    f.write(f"warmup_bias_lr: 0.1\n")
    f.write(f"box: {best_params['box']}\n")
    f.write(f"cls: {best_params['cls']}\n")
    f.write(f"cls_pw: 1.0\n")
    f.write(f"obj: {best_params['obj']}\n")
    f.write(f"obj_pw: 1.0\n")
    f.write(f"iou_t: {best_params['iou_t']}\n")
    f.write(f"anchor_t: 4.0\n")
    f.write(f"fl_gamma: {best_params['fl_gamma']}\n")
    f.write(f"mosaic: {mosaic}\n")
    f.write(f"copy_paste: {copy_paste}\n")
    f.write(f"hsv_h: {hsv_h}\n")
    f.write(f"hsv_s: {hsv_s}\n")
    f.write(f"hsv_v: {hsv_v}\n")
    f.write(f"degrees: {degrees}\n")
    f.write(f"translate: {translate}\n")
    f.write(f"scale: {scale}\n")
    f.write(f"shear: {shear}\n")
    f.write(f"perspective: {perspective}\n")
    f.write(f"flipud: {flipud}\n")
    f.write(f"fliplr: {fliplr}\n")
    f.write(f"mixup: {mixup}\n")

print('Best trial:')
trial = study.best_trial
print(trial.values)
print(trial.params)
