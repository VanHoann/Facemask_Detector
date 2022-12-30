# Facemask Detector

## 1. Papers
- [x] Detector
  - [x] MTCNN: https://arxiv.org/abs/1604.02878
- [x] Classifier
  - [x] EfficcientNet: https://arxiv.org/abs/1905.11946
  - [x] EfficcientNetV2: https://arxiv.org/abs/2104.00298

## 2. Datasets
- [x] Detector
  - [x] Widerface: https://arxiv.org/abs/1511.06523
- [x] Classifier
  - [x] Face Mask Dataset: https://www.kaggle.com/datasets/shiekhburhan/face-mask-dataset

## 3. Project Progress
### 3.1. Data preparation
- [x] Detector
- [x] Classifier
### 3.2. Architecture Implementation
- [x] Detector
- [x] Classifier
### 3.3. Training 
- [x] Detector
- [x] Classifier
### 3.4. Inference and Evaluation
- [x] Detector
- [x] Classifier
### 3.5. Report
- [x] Project report


## 4. Project Layout
```
project
├── MaskDetector_demo.ipynb: project demo
├── MaskDetector_demo_GPU.ipynb: project demo running on GPU instead of CPU
├── Project_report.pdf: final report for the project
├── models
|   ├── detector: scripts for the detector
|   |   └── final_model: configuration and weights for the final detector
|   └── classifier: scripts for the classifier
|       └── weights: weights for the final classifier
├── notebooks
|       ├── MTCNN_Training.ipynb: sample notebook of a training session for the detector
|       └── EfficientNet_Training.ipynb: sample notebook of a training session for the classifier
└── tests: sample images stored on github for demo purposes
