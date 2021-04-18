# P Stage 1: Image Classification

```
$> tree -d
.
├── /dataloader
│     ├── data_preprocess.py: info.csv에 미리 클래스를 만들어주고 이미지 파일을 jpg로 통일 
│     ├── dataset.py: Dataset Class 정의
│     ├── eda.py
├── /model
│     ├── loss.py: 다양한 loss class를 정의
│     ├── model.py: 다양한 model class을 정의
│     └── optimizer.py: 다양한 optimizer 반환 함수 정의
├── /trainer
│     ├── trainer.py: gender_age 모델 학습을 위한 Trainer class 정의
│     └── trainer_mask.py: mask 모델 학습을 위한 Trainer class 정의
├── config.py: Hyperparameter 
├── inference.py
├── train.py
├── train_mask.py
└── utils.py : 그 외 필요한 기능
``` 

## 요약 

- 기간: 2021.03.29 ~ 2021.04.08
- 대회 설명: 인물의 사진에서 마스크 유무(2개 class), 나이대(3개 class), 성별(2개 class) 을 분류
- 검증 전략: Stratified K-fold Cross Validation(n=5)
- 사용한 모델 아키텍처 및 하이퍼 파라미터
    - EfficientNet Noisy Student(b0, b4) [link](https://github.com/rwightman/pytorch-image-models)
    - Optimizer: Adamp
    - Learning rate: 1e-4
    - Scheduler: Consine annealing warm restart
    - Batch size: 64

<br/>

## Scores
|  |Public LB|Private LB|
|--|--|--|
|F1|0.7011|0.7168|
|Accuracy|78.34%|77.44%|

<br/>

## 도움이 되었던 것
1. Oversampling
    - k-fold를 한 후, 학습을 진행할때 모든 클래스의 data 수를 oversampling을 통해 맞췄다. --> 여기서 성능의 한계가 생겼을 수도 있다.
2. Label smoothing
    - 기본적인 Label smoothing 사용(epsilon=0.4)
    - 추가적으로 나이의 경계(+-10살)에 있는 사람의 경우 0.05의 label을 섞어주었다.
3. Cutmix
    - 인물의 사진이 가운데에 있기 때문에 세로로 절반을 잘라서 cutmix를 진행했다.
4. Taylor cross entropy
    - 사람의 나이가 noisy data를 만들어내기 때문에 taylor cross entropy를 적용했다.

<br/>

## 도움이 되지 않은 것
1. F1 loss

<br/>

## Code
### Train
- config.py 에서 hyperparameter를 설정한다.
- 마스크를 구별하는 모델, 나이성별을 구분하는 모델을 각각 학습한다.
```bash
$ python train.py
$ python train_mask.py
```
### Inference
config.py 에서 hyperparameter를 설정한다.
```bash
$ python inference.py
```