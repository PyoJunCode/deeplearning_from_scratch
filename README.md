# Deep Learning Implement from Scratch

해당 repository는 학교수업, 개인프로젝트, 개인공부 등에서 여러가지 Deep learning model 과 MLops technique 들에 대해서 scratch부터 직접 구현한 코드들의 일부를 모았습니다.



## Components

- MLops/

  - [deploy_simple_server](https://github.com/PyoJunCode/deeplearning_from_scratch#deploy_simple_server)
  - [SKLearn_feature_selection](https://github.com/PyoJunCode/deeplearning_from_scratch)
  - [TFDV_diabetes](https://github.com/PyoJunCode/deeplearning_from_scratch#TFDV_diabetes)
  - [TFDV_example](https://github.com/PyoJunCode/deeplearning_from_scratch#TFDV_example)
  - [TFX_pipeline](https://github.com/PyoJunCode/deeplearning_from_scratch#TFX_pipeline)

- NLP/

  - [emojify](https://github.com/PyoJunCode/deeplearning_from_scratch#emojify)
  - [fine_tuning](https://github.com/PyoJunCode/deeplearning_from_scratch#fine_tuning)
  - [Reformer_chatbot](https://github.com/PyoJunCode/deeplearning_from_scratch#Reformer_chatbot)
  - [Summarizer_trax](https://github.com/PyoJunCode/deeplearning_from_scratch#Summarizer_trax)

- Course/

  - [custom_dataloader_cnn](https://github.com/PyoJunCode/deeplearning_from_scratch#custom_dataloader_cnn)

  - [linear_models](https://github.com/PyoJunCode/deeplearning_from_scratch#linear_models)

  - [mnist_fashion_both](https://github.com/PyoJunCode/deeplearning_from_scratch#mnist_fashion_both)

  - [predict_exam_score_regression](https://github.com/PyoJunCode/deeplearning_from_scratch#predict_exam_score_regression)

  - [wine_classifier](https://github.com/PyoJunCode/deeplearning_from_scratch#wine_classifier)

    

---

<br>

# Outline

<br>

# MLOps

자세한 설명은 프로젝트의 notebook에 포함되어있습니다.

## deploy_simple_server

  

fastapi를 사용하여 multi image recognition model을 직접 로컬 서버에 deploy하고 clinet를 통해 consume하는 간단한 Model Serving 예제.



**Server** : pre-trained Yolo v3 model을 FastAPI를 통해 로컬 서버에 deploy. client로 부터의 요청을 받아 사진을 분석하고 label을 box로 표시해서 return해줌.

**Client** : 사진을 업로드하고 Server에 API call

 

  ## SKLearn_feature_selection



[Breast Cancer Dataset ](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29)에 대해 feature selection을 수행



- Pandas를 이용해 dataset을 분석하고 기본적인 전처리

- SKlearn의 random forest, RFE(random feature elimination) 등을 이용해 Feature selection 수행

- Correlation matrix, Accuracy, ROC, Precision, Recall, F1 score 등 여러가지 metric을 통해 각 feature selection 수행 결과에 대한 분석 수행



## TFDV_diabetes



TFDV(Tensorflow Data Validation) 모듈을 사용해 데이터를 분석



Dataset : [Diabetes 130-US hospitals for years 1999-2008 Data Set](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)



당뇨병 환자 Dataset을 이용하여

- Data statistic 생성/visualization 수행
- Data schema 생성/visualization 수행
- Data anomalies 검출/조정
- Training data와 Serving data간의 schema check/validation
- Data Drift/Skew check



## TFDV_Income



TFDV(Tensforflow Data Validation) 모듈을 사용해 데이터를 분석



Dataset : [Census Income Dataset](http://archive.ics.uci.edu/ml/datasets/Census+Income)



- Data statistic 생성/visualization 수행
- Data schema 생성/visualization 수행
- Data anomalies 검출/조정
- Training data와 Serving data간의 schema check/validation
- Data Drift/Skew check





## TFX_Pipeline



TFX(Tensor Flow Extended) 모듈을 사용해 Pipeline을 구축

 기본적인 Interactive Context를 사용하여 notebook내에서의 visualization을 수행

(Kubeflow, Airflow 에 연결하는 예제는 [여기](https://github.com/PyoJunCode/data-centric-pipeline)에서 확인가능)



Dataset :  [Metro Interstate Traffic Volume dataset](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

 

- ExampleGen 생성 (Split Train/Val/Test)
- StatisticsGen 생성/ Visualization (Dataset 통계)
- SchemaGen 생성/ Visualization (Dataset Schema)
- ExampleValidator로 Data 검증 (Anomalies)
- TransformGen 생성 (preprocessing)

---

<br>

# NLP

각 model 에 대한 자세한 설명은  notebook에 포함되어있습니다.

## emojify



Dataset: EMOJISET



EMOJISET data를 이용하여 입력 Text에 대한 알맞은 EMOJI를 예측



- GloVe Word embedding을 수행 후 기본적인 DNN model을 통해 학습/예측하는 Model V1



- keras **LSTM** 구조를 통해 embedding 후 학습/예측하는 Model V2 



Model V1, V2를 구현 후 비교



## fine_tuning





- Tokenizing, Embedding 과정을 거쳐 encoder-decoder를 갖춘 기본적인 **Transformer** Model을 Scratch부터  구현.



- pre-trained BERT, SQuAD, T5 model을 이용하여 원하는 Data로 Fine-tuning 적용하여 QA 예측





## Reformer_chatbot



Reformer model을 scratch부터 구현하고 학습시켜 간단한 Chatbot 구현



## Summarizer_trax

 

Dataset을 Tokenize하고 Transformer model을 scratch부터 구현하여 Text summarizer 구현



---

<br>



# Course

<br>

## custom_dataloader_cnn

Pytroch의 기본 모듈을 Customizing 하여 폴더형식으로 되어있는 Dataset를 Load하고 Label에 matching



Custom CNN을 scratch부터 구현하여 Image data Training.





## linear_models

기본적인 Logistic / Ridge / Polynomial regression 구현

## mnist_fashion_both



mnist와 fashion_mnist dataset을 함께 합쳐 전처리 후 labelling

기본적인 Data augmentation을 적용 후 Custom CNN model을 만들어 Training.



## predict_exam_score_regression



Student data를 통해 feature-engineering을 수행한 뒤 MLP model 으로 training, Q3(exam) score를 예측



## wine_classifier



SKLearn wine dataset을 사용해 기본적인 linear classifier를 만들어 학습/예측

