# [Dacon] 2023 교원그룹 AI 챌린지
## Dacon Challenge overview
**[주제]**<br/>
손글씨 인식 AI 모델 개발

**[배경]**<br/>
AI가 학습에 적극적으로 활용되는 교육 시장의 흐름을 선도하고자 유아의 손글씨 인식에 최적화된 인공지능 개발

**[설명]**<br/>
예선: 손글씨 폰트 이미지를 바탕으로 Text Recognition을 수행하는 인식 AI 모델을 개발<br/>
본선: 교원 그룹의 실제 유아 손글씨 데이터로 진행

**[평가산식]**<br/>
Accuracy

[대회 링크](https://dacon.io/competitions/official/236042/overview/description)

## Dependencies
- python 3.8
- pytorch 1.10.1
- torchvision 0.11.2
- pandas
- tqdm 
- scikit-learn
- tensorboard
- setuptools
- straug

## Datasets
대회에서 제공하는 [데이터](https://dacon.io/competitions/official/236042/data)는 아래와 같이 구성
```
├── train
│   └── TRAIN_00000.png ~ TRAIN_76887.png
├── test
│   └── TEST_00000.png ~ TEST_74120.png
├── train.csv
├── test.csv
└── sample_submission.csv
```
위 데이터들을 main.py이 있는 경로에 같이 저장

## Baseline Model
Naver Clova에서 발표한 CVPR 2019, "What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis"에서 제공된 코드를 이용하였으며, 모델 구성은 다음과 같이 사용.<br/>
`"TPS - ResNet - BiLSTM - Attn"`<br/>

[paper](https://arxiv.org/abs/1904.01906)<br/>
[github](https://github.com/clovaai/deep-text-recognition-benchmark)<br/>

## Augmentation
[https://github.com/roatienza/straug] 에서 제공된 straug 라이브러리를 이용하여 augmentation 적용<br/>
straug 라이브러리에서 제공하는 여러가지 blur 및 noise를 사용하였고, 그 중 Gaussian Blur를 0.5 비율로 train할 때, 가장 높은 정확도를 보임

## 결과
private 정확도 0.89141, 최종순위 20위로 마감<br/>
(참고: 1위 private 정확도 0.96274)
