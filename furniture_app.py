from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "furniture_cnn.keras"
HISTORY_IMAGE_PATH = BASE_DIR / "outputs" / "training_history.png"
CLASS_DISTRIBUTION_PATH = BASE_DIR / "outputs" / "class_distribution.png"
CONFUSION_MATRIX_PATH = BASE_DIR / "outputs" / "confusion_matrix.png"
REPORT_PATH = BASE_DIR / "outputs" / "classification_report.csv"

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
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }
    .result-card {
        border-left: 6px solid #1e4e8c;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        background: #f8fbff;
        border-top: 1px solid #dbeafe;
        border-right: 1px solid #dbeafe;
        border-bottom: 1px solid #dbeafe;
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


def resize_for_model(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    return image.resize((32, 32))


def preprocess_image(image: Image.Image) -> np.ndarray:
    resized_image = resize_for_model(image)
    image_array = np.array(resized_image) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image(model, image: Image.Image) -> tuple[str, float, np.ndarray]:
    input_data = preprocess_image(image)
    probabilities = model.predict(input_data, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])
    return predicted_class, confidence, probabilities


def make_result_df(probabilities: np.ndarray) -> pd.DataFrame:
    result_df = pd.DataFrame(
        {
            "카테고리": [CLASS_LABELS[name] for name in CLASS_NAMES],
            "영문 라벨": CLASS_NAMES,
            "예측 확률": probabilities,
        }
    ).sort_values("예측 확률", ascending=False)
    result_df["예측 확률(%)"] = (result_df["예측 확률"] * 100).round(2)
    return result_df


st.title("가구 이미지 기반 카테고리 분류")
st.caption("CIFAR-100 가구 클래스 기반 CNN 딥러닝 개인 프로젝트")

if not MODEL_PATH.exists():
    st.error("저장된 모델 파일을 찾을 수 없습니다. models/furniture_cnn.keras 파일을 확인하세요.")
    st.stop()

model = load_furniture_model()

tab_intro, tab_predict, tab_training, tab_about = st.tabs(
    ["프로젝트 설명", "예측 시연", "학습 결과", "프로젝트 정보"]
)

with tab_intro:
    st.subheader("프로젝트 설명")
    st.write(
        "본 프로젝트는 CIFAR-100 데이터셋에서 가구 관련 5개 클래스를 추출하여 "
        "CNN 모델을 학습하고, 업로드된 가구 이미지의 카테고리를 예측하는 딥러닝 프로젝트입니다."
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("분류 클래스", "5개")
    metric_col2.metric("학습 이미지", "2,500장")
    metric_col3.metric("테스트 이미지", "500장")

    st.markdown(
        """
        - 예측 대상: 침대, 의자, 소파, 테이블, 수납장
        - 사용 모델: CNN 이미지 분류 모델
        - 이미지 크기: 32 x 32 RGB
        - 저장 모델: `models/furniture_cnn.keras`
        """
    )

    st.info(
        "본 모델은 CIFAR-100의 32 x 32 저해상도 이미지로 학습했기 때문에 "
        "실제 고해상도 상품 이미지에서는 오분류가 발생할 수 있습니다."
    )

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
            with st.expander("모델이 실제로 보는 32 x 32 이미지"):
                st.image(
                    resize_for_model(image).resize((256, 256), Image.Resampling.NEAREST),
                    caption="32 x 32 변환 이미지",
                    use_container_width=True,
                )

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

                result_df = make_result_df(probabilities)

                st.subheader("Top-3 예측 후보")
                top3_df = result_df.head(3)[["카테고리", "영문 라벨", "예측 확률(%)"]].copy()
                st.dataframe(top3_df, use_container_width=True, hide_index=True)

                st.subheader("클래스별 예측 확률")
                chart = (
                    alt.Chart(result_df)
                    .mark_bar(color="#1E4E8C")
                    .encode(
                        x=alt.X("카테고리:N", title="카테고리"),
                        y=alt.Y(
                            "예측 확률:Q",
                            title="예측 확률",
                            scale=alt.Scale(domain=[0, 1]),
                        ),
                        tooltip=["카테고리", "영문 라벨", "예측 확률(%)"],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart, use_container_width=True)

                full_df = result_df[["카테고리", "영문 라벨", "예측 확률(%)"]].copy()
                st.dataframe(full_df, use_container_width=True, hide_index=True)

                st.caption(
                    "위 표는 읽기 전용 결과표입니다. 마우스 휠을 움직여도 예측 확률 값은 변경되지 않습니다."
                )

with tab_training:
    st.subheader("모델 학습 결과")

    if HISTORY_IMAGE_PATH.exists():
        st.image(HISTORY_IMAGE_PATH, caption="학습 정확도 및 손실 그래프", use_container_width=True)
    else:
        st.warning("outputs/training_history.png 파일을 찾을 수 없습니다.")

    if CLASS_DISTRIBUTION_PATH.exists():
        st.subheader("클래스별 데이터 분포")
        st.image(CLASS_DISTRIBUTION_PATH, use_container_width=True)

    if CONFUSION_MATRIX_PATH.exists():
        st.subheader("혼동행렬")
        st.image(CONFUSION_MATRIX_PATH, use_container_width=True)

    if REPORT_PATH.exists():
        st.subheader("분류 성능 지표")
        report_df = pd.read_csv(REPORT_PATH)
        st.dataframe(report_df, use_container_width=True, hide_index=True)

    st.write(
        "학습된 모델은 `models/furniture_cnn.keras`로 저장되어 있으며, "
        "Streamlit 앱에서는 모델을 다시 학습하지 않고 저장된 모델을 불러와 예측만 수행합니다."
    )

with tab_about:
    st.subheader("프로젝트 정보")
    st.markdown(
        """
        이 프로젝트는 이전 웹 프로젝트에서 텍스트 기반 상품 검색만으로는 이미지의 형태와 시각적 특징을
        충분히 반영하기 어렵다는 점에서 출발했습니다. CNN 모델을 사용하여 가구 이미지의 특징을 학습하고,
        향후 이미지 기반 상품 검색 또는 카테고리 자동 추천 기능으로 확장할 가능성을 확인합니다.

        실제 서비스에 적용하려면 CIFAR-100이 아닌 실제 쇼핑몰 가구 이미지 데이터셋으로 추가 학습이 필요합니다.
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
