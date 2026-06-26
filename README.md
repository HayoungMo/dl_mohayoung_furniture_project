# Furniture Image Classification

가구 이미지 기반 상품 카테고리 분류 딥러닝 프로젝트입니다.  
CIFAR-100 데이터셋 중 가구 관련 클래스만 추출하여 CNN 모델을 학습하고, 입력 이미지가 어떤 가구 카테고리에 속하는지 예측합니다.

## 프로젝트 배경

이전 웹 프로젝트에서 상품 검색과 추천 기능을 구현하면서 텍스트 기반 검색만으로는 이미지의 형태, 색상, 질감 같은 시각적 특징을 반영하기 어렵다는 한계를 느꼈습니다.  
본 프로젝트에서는 가구 이미지를 학습하여 상품 카테고리를 자동 분류하는 딥러닝 모델을 구현하고, 향후 쇼핑몰 서비스의 이미지 기반 검색 기능으로 확장할 수 있는 가능성을 확인합니다.

## 사용 데이터

- 데이터셋: CIFAR-100
- 사용 클래스: bed, chair, couch, table, wardrobe
- 학습 데이터: 2,500장
- 테스트 데이터: 500장
- 이미지 크기: 32 x 32 RGB

## 주요 기능

- CIFAR-100 데이터 로드
- 가구 클래스만 필터링
- 이미지 데이터 정규화
- CNN 모델 학습
- 테스트 데이터 성능 평가
- 학습 결과 시각화
- 학습 모델 저장

## 기술 스택

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Streamlit

## 프로젝트 구조

```text
dl_furniture_project/
├─ data/
├─ models/
├─ notebooks/
├─ outputs/
├─ furniture_app.py
├─ train_model.ipynb
├─ requirements.txt
└─ README.md
```

## 실행 방법

가상환경을 활성화한 뒤 필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

모델 학습은 Jupyter Notebook에서 진행합니다.

```text
train_model.ipynb
```

Streamlit 앱 실행은 아래 명령어를 사용합니다.

```bash
streamlit run furniture_app.py
```

## 모델 설명

본 프로젝트에서는 이미지 분류에 적합한 CNN 모델을 사용합니다.  
CNN은 이미지의 선, 모서리, 형태 같은 특징을 단계적으로 학습하여 최종적으로 가구 카테고리를 분류합니다.

## 향후 개선 방향

- 실제 쇼핑몰 가구 이미지 데이터셋으로 확장
- 이미지 업로드 기반 상품 카테고리 예측 기능 구현
- 예측된 카테고리를 활용한 상품 검색 및 추천 기능 연동
- Spring Boot + React 기반 웹 서비스와 딥러닝 모델 API 연결
