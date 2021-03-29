# Unseen Food Instance Segmentation

This is the implementation of paper submitted as 

> D. Park, J. Lee, J. Lee and K. Lee. **Deep Learning based Unseen Food Instance Segmentation using Synthetic Data.** 2021 Ubiquitous Robots (UR)

![Demo](./resources/demo_unseen_food_segmentation.gif)


# Usage
- Training
```Shell
# Learning from scratch
python train.py --config {PATH/TO/CONFIG}

# Resuming training
python train.py \
 --config {PATH/TO/CONFIG} \
 --resume --resume_ckp PATH/TO/TTRAINED/WEIGHT
```

- Evaluation
```Shell
# Evaluation one weight
python test_single_models.py \
 --config {PATH/TO/CONFIG} \
 --trained_ckp PATH/TO/TTRAINED/WEIGHT

# Evaluation many weight saved in same directory
python test_single_models.py \
 --config {PATH/TO/CONFIG} \
 --ckp_dir PATH/TO/TTRAINED/DIRECTORY
```

- Inference demo
```Shell
# predict and draw food instance
python inference.py \
 --input_dir PATH/TO/IMAGES/FOR/INFERNCE \
 --output_dir PATH/TO/SAVE/VISUALIZED/IMAGES \
 --ckp_path PATH/TO/TRAINED/WEIGHT
```