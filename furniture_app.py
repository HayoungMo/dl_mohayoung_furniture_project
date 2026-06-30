from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "furniture_cnn.keras"
HISTORY_IMAGE_PATH = BASE_DIR / "outputs" / "training_history.png"

CLASS_NAMES = ["bed", "chair", "couch", "table", "wardrobe"]
CLASS_LABELS = {
    "bed": "침대",
    "chair": "의자",
    "couch": "소파",
    "table": "테이블",
    "wardrobe": "수납장",
}


st.set_page_config(
    page_title="가구 이미지 분류",
    page_icon="🪑",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }
    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        background: #ffffff;
    }
    .result-card {
        border-left: 6px solid #ff4b4b;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        background: #fff7ed;
        border-top: 1px solid #fed7aa;
        border-right: 1px solid #fed7aa;
        border-bottom: 1px solid #fed7aa;
    }
    .small-muted {
        color: #6b7280;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_furniture_model():
    return load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image(model, image: Image.Image) -> tuple[str, float, np.ndarray]:
    input_data = preprocess_image(image)
    probabilities = model.predict(input_data, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])
    return predicted_class, confidence, probabilities


st.title("가구 이미지 기반 카테고리 분류")
st.caption("CIFAR-100 가구 클래스 기반 CNN 딥러닝 개인 프로젝트")

intro_col, stat_col = st.columns([1.7, 1])

with intro_col:
    st.subheader("프로젝트 설명")
    st.write(
        "이미지를 업로드하면 학습된 CNN 모델이 가구 이미지를 "
        "침대, 의자, 소파, 테이블, 수납장 중 하나로 분류합니다."
    )
    st.markdown(
        """
        - 사용 데이터: CIFAR-100 중 가구 관련 5개 클래스
        - 학습 이미지: 2,500장
        - 테스트 이미지: 500장
        - 사용 모델: CNN 이미지 분류 모델
        """
    )

with stat_col:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("분류 클래스", "5개")
    st.metric("이미지 크기", "32 x 32 RGB")
    st.metric("저장 모델", "furniture_cnn.keras")
    st.markdown("</div>", unsafe_allow_html=True)


if not MODEL_PATH.exists():
    st.error("저장된 모델 파일을 찾을 수 없습니다. models/furniture_cnn.keras 파일을 확인하세요.")
    st.stop()

model = load_furniture_model()

tab_predict, tab_training, tab_about = st.tabs(["예측 시연", "학습 결과", "프로젝트 정보"])

with tab_predict:
    st.subheader("이미지 업로드")
    uploaded_file = st.file_uploader(
        "분류할 가구 이미지를 업로드하세요.",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("이미지를 업로드하면 예측 결과가 여기에 표시됩니다.")
    else:
        image = Image.open(uploaded_file)
        image_col, result_col = st.columns([1, 1.35])

        with image_col:
            st.image(image, caption="업로드 이미지", use_container_width=True)
            st.caption("모델 입력을 위해 이미지는 내부적으로 32 x 32 크기로 변환됩니다.")

        with result_col:
            if st.button("카테고리 예측하기", type="primary", use_container_width=True):
                predicted_class, confidence, probabilities = predict_image(model, image)
                predicted_label = CLASS_LABELS[predicted_class]

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="small-muted">현재 이미지 기준 예측 결과</div>
                        <h2>{predicted_label} <span style="font-size:1rem;">({predicted_class})</span></h2>
                        <p>모델이 이 이미지를 <b>{predicted_label}</b> 카테고리로 분류했습니다.</p>
                        <p>예측 확률: <b>{confidence * 100:.2f}%</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                result_df = pd.DataFrame(
                    {
                        "카테고리": [CLASS_LABELS[name] for name in CLASS_NAMES],
                        "영문 라벨": CLASS_NAMES,
                        "예측 확률": probabilities,
                    }
                ).sort_values("예측 확률", ascending=False)

                st.subheader("클래스별 예측 확률")
                st.bar_chart(result_df.set_index("카테고리")["예측 확률"])
                st.dataframe(
                    result_df.assign(예측확률표시=lambda df: (df["예측 확률"] * 100).round(2).astype(str) + "%"),
                    use_container_width=True,
                    hide_index=True,
                )

with tab_training:
    st.subheader("모델 학습 결과")

    if HISTORY_IMAGE_PATH.exists():
        st.image(HISTORY_IMAGE_PATH, caption="학습 정확도 및 손실 그래프", use_container_width=True)
    else:
        st.warning("outputs/training_history.png 파일을 찾을 수 없습니다.")

    st.write(
        "학습된 모델은 `models/furniture_cnn.keras`로 저장되어 있으며, "
        "Streamlit 앱에서는 모델을 다시 학습하지 않고 저장된 모델을 불러와 예측만 수행합니다."
    )

with tab_about:
    st.subheader("프로젝트 정보")
    st.markdown(
        """
        이 프로젝트는 이전 웹 프로젝트에서 텍스트 기반 상품 검색만으로는 이미지의 형태와 시각적 특징을
        반영하기 어렵다는 점에서 출발했습니다. CNN 모델을 활용하여 가구 이미지의 특징을 학습하고,
        향후 이미지 기반 상품 검색 또는 추천 기능으로 확장할 수 있는 가능성을 확인합니다.

        본 프로젝트는 학습용 공개 데이터셋인 CIFAR-100을 사용하였으며, 실제 서비스에 적용하려면
        더 큰 실제 가구 이미지 데이터셋으로 추가 학습이 필요합니다.
        """
    )

    class_info = pd.DataFrame(
        {
            "CIFAR-100 라벨 번호": [5, 20, 25, 84, 94],
            "영문 클래스": CLASS_NAMES,
            "한글 표시": [CLASS_LABELS[name] for name in CLASS_NAMES],
        }
    )
    st.dataframe(class_info, use_container_width=True, hide_index=True)
