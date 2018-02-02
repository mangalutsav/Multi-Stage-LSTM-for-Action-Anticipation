# Multi Stage LSTM (MS-LSTM) for Action Anticipation

## Implementation of "Encoraging LSTMs to Anticipate Actions Very Early", ICCV 2017 <br/>
[Encouraging LSTMs To Anticipation Actions Very Early, ICCV 2017](http://openaccess.thecvf.com/content_ICCV_2017/papers/Aliakbarian_Encouraging_LSTMs_to_ICCV_2017_paper.pdf)

---
## Abstract
In contrast to the widely studied problem of recognizing an action given a complete sequence, action anticipation aims to identify the action from only partially available videos. As such, it is therefore key to the success of computer vision applications requiring to react as early as possible, such as autonomous navigation. In this paper, we propose a new action anticipation method that achieves high prediction accuracy even in the presence of a very small percentage of a video sequence. To this end, we develop a multi-stage LSTM architecture that leverages context-aware and action-aware features, and introduce a novel loss function that encourages the model to predict the correct class as early as possible. Our experiments on standard benchmark datasets evidence the benefits of our approach; We outperform the state-of-the-art action anticipation methods
for early prediction by a relative increase in accuracy of 22.0% on JHMDB-21, 14.0% on UT-Interaction and 49.9% on UCF-101.

![Overview of MS-LSTM](MS_LSTM_Overview.png)

---

## Usage
There are a couple of steps involved to run the full model. Later, will will update this and prepare a run.py that covers all of these steps.

##### Step 0: Preparing data/dataset
Please put all video folders of your dataset (code prepared for jhmdb-21) into data/jhmdb_dataset. Please copy all the split annotation files (.txt) into data/splits. We already put all split annotation files into that directory.

##### Step 1: Converting videos to frames in a splitted format
To convert videos into frame, please run
```
python mkframes.py --input-dir path/to/data/jhmdb_dataset/ --output-dir path/to/data/frames/ --format png
```

And, to prepare them for training convNets (put it into train/val splits so that a generator can have access to them), please run
```
python make_split.py --split-dir path/to/data/splits/ --data-dir path/to/data/frames/ --index 1 --output-dir path/to/data/splitted_data/
```

##### Step 2: Train your action-aware/context-aware models.
These models should be pre-trained on ImageNet. If it is the first time you are using this, it automatically download VGG-16 weights, pre-trained on ImageNet. For training context-aware model, please run
```
CUDA_VISIBLE_DEVICES=0 python action_context_train.py --data-dir data/splitted_data/ --classes 21 --model-type context_aware --epochs 128 --save-model data/model_weights/context_best.h5 --save-best-only --fixed-width 224 --learning-rate 0.001 --batch-size 32
```
For action-awre model, similarly, please run
```
CUDA_VISIBLE_DEVICES=0 python action_context_train.py --data-dir data/splitted_data/ --classes 21 --model-type action_aware --epochs 128 --save-model data/model_weights/action_best.h5 --save-best-only --fixed-width 224 --learning-rate 0.001 --batch-size 32
```
The models' weights are going to be saved in data/model_weights. Please note that after training, for each model, you will have a <model>_final.h5 and <model>_best.h5. For the rest of steps, if is recommended to use <model>_best.h5 for each model.


##### Step 3: Feature extraction.
Next step is to extract features from action-aware and context-aware models. To this end, please run
```
CUDA_VISIBLE_DEVICES=0 python context_aware_features.py --data-dir data/jhmdb_dataset/ --split-dir data/splits/ --classes 21 --model data/model_weights/context_best.h5 --temporal-length 50 --split 1 --output data/context_features/ --fixed-width 224
```
Similarly, for action-aware features, please run
```
CUDA_VISIBLE_DEVICES=0 python action_aware_features.py --data-dir data/jhmdb_dataset/ --split-dir data/splits/ --classes 21 --model data/model_weights/action_best.h5 --temporal-length 50 --split 1 --output data/action_features/ --fixed-width 224
```

##### Step 4: Training MS-LSTM
Given all features extracted from action-aware and context-aware model, you can train MS-LSTM model. To this end, please run
```
CUDA_VISIBLE_DEVICES=0 python ms_lstm.py --action-aware data/action_features/ --context-aware data/context_features/ --classes 21 --epochs 128 --save-model data/model_weights/mslstm_best.h5 --save-best-only --learning-rate 0.0001 --batch-size 32 --temporal-length 50 --cell 2048 --loss crossentropy
```
For better performance, if GPU memory lets you, try cell 4096. You can also try other losses from:
'crossentropy', 'hinge', 'totally_linear', 'partially_linear', 'exponential'


##### Step 5: Evaluation
You can evaluate the performance of the model,  with and without using Temporal Average Pooling. To this end, please run
```
CUDA_VISIBLE_DEVICES=0 python ms_lstm.py --action-aware data/action_features/ --context-aware data/context_features/ --classes 21 --temporal-length 50 --cell 2048
```


---
## Citation
If you are using our code, please cite
```
@InProceedings{Aliakbarian_2017_ICCV,
author = {Sadegh Aliakbarian, Mohammad and Sadat Saleh, Fatemeh and Salzmann, Mathieu and Fernando, Basura and Petersson, Lars and Andersson, Lars},
title = {Encouraging LSTMs to Anticipate Actions Very Early},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
} 

@article{aliakbarian2016deep,
  title={Deep action-and context-aware sequence learning for activity recognition and anticipation},
  author={Aliakbarian, Mohammad Sadegh and Saleh, Fatemehsadat and Fernando, Basura and Salzmann, Mathieu and Petersson, Lars and Andersson, Lars},
  journal={arXiv preprint arXiv:1611.05520},
  year={2016}
}
```


---
## Contact
For any question, bug report, and etc., please contact Sadegh Aliakbarian (PhD Student at Australian National Unviersity, Researcher at CSIRO and ACRV), mohammadsadegh.aliakbarian@data61.csiro.au 
