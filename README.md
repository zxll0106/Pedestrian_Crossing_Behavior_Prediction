# Pedestrian Crossing Behavior Prediction
This repo contains code of our paper "Social Aware Multi-Modal Pedestrian Crossing Behavior Prediction". 

_Xiaolin Zhai, Zhengxi Hu, Dingye Yang, Lei Zhou and Jingtai Liu_


# Dependencies

- Software Environment: Linux 
- Hardware Environment: NVIDIA RTX 3090
- Python `3.8`
- PyTorch `1.11.0`, Torchvision `0.12.0`


# Data
We have tested our method with [PIE](https://data.nvision2.eecs.yorku.ca/PIE_dataset/) and [JAAD](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/) datasets. Users should follow their original instruction to download and prepare datasets. Users also need to get the extracted features from a pretrained VGG16 following the [PIEPredict repo](https://github.com/aras62/PIEPredict). As another option, users can download the vg166 features [here](https://drive.google.com/file/d/1xQAyvqE2Q4cxvjyWsCEJR09QjB7UYJIV/view?usp=sharing) extracted by [pedestrian_intent_action_detection](https://github.com/umautobots/pedestrian_intent_action_detection) and put it in `DATA_PATH/PIE_dataset/saved_output`.

# Setup
Run setup script
```
python setup.py build develop
```

# Train
Run following command to train model with original PIE data annotation:
```
python tools/train.py --config /extend/zxl/Intent_Estimation/pedestrian_intent_action_detection_wo_pose-main/configs/PIE_intent_action_relation.yaml --gpu 0 STYLE PIE MODEL.TASK action_intent_single SOLVER.MAX_ITERS 15000 TEST.BATCH_SIZE 1
```

Run following command to train model with SF-GRU style data annotation, change `--config_file` to `configs/JAAD_intent_action_relation.yaml` or `configs/PIE_intent_action_relation.yaml` to train on JAAD or PIE datasets. :
```
python tools/train.py --config /extend/zxl/Intent_Estimation/pedestrian_intent_action_detection_wo_pose-main/configs/JAAD_intent_action_relation.yaml --gpu 0 STYLE SF-GRU MODEL.TASK action_intent_single SOLVER.MAX_ITERS 50000 TEST.BATCH_SIZE 1 SOLVER.SCHEDULER none DATASET.BALANCE False
```

# Test 
Change 1) `STYLE` value to `PIE` or `SF-GRU` ; 2)`--config_file` to corresponding datasets. For example: 
 
``` 
python tools/test.py --config /extend/zxl/Intent_Estimation/pedestrian_intent_action_detection_wo_pose-main/configs/PIE_intent_action_relation.yaml --gpu 0 STYLE PIE MODEL.TASK action_intent_single SOLVER.MAX_ITERS 15000 TEST.BATCH_SIZE 1
```

# Acknowledgement

We thank for the part of code of pedestrian_intent_action_detection, whose github repo is [pedestrian_intent_action_detection](https://github.com/umautobots/pedestrian_intent_action_detection). We thank the authors for releasing their code.
