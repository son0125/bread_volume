import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="Bread Quality Prediction",
    layout="wide"
)

st.title("Artificial Intelligence Prediction of Bread Qualities")
st.caption("made by Korea Food Research Institute and Sejong Univ.")

st.write(
    "(Hyperspectral imaging 데이터를 이용하여 빵의 volume과 hardness를 예측하는 모델입니다.)"
)


@st.cache_resource
def load_model_files(model_name):
    # Volume model
    volume_model = joblib.load(f"{model_name}_volume_model.pkl")
    volume_x_scaler = joblib.load(f"x_scaler_{model_name}_volume.pkl")
    volume_y_scaler = joblib.load(f"y_scaler_{model_name}_volume.pkl")
    volume_features = joblib.load("feature_columns_volume.pkl")

    # Hardness model
    hardness_model = joblib.load(f"{model_name}_hardness_model.pkl")
    hardness_x_scaler = joblib.load(f"x_scaler_{model_name}_hardness.pkl")
    hardness_y_scaler = joblib.load(f"y_scaler_{model_name}_hardness.pkl")
    hardness_features = joblib.load("feature_columns_hardness.pkl")

    return (
        volume_model,
        volume_x_scaler,
        volume_y_scaler,
        volume_features,
        hardness_model,
        hardness_x_scaler,
        hardness_y_scaler,
        hardness_features
    )


model_name = st.selectbox(
    "Choose a prediction model",
    ["CatBoost", "GBM", "AdaBoost"]
)


(
    volume_model,
    volume_x_scaler,
    volume_y_scaler,
    volume_features,
    hardness_model,
    hardness_x_scaler,
    hardness_y_scaler,
    hardness_features
) = load_model_files(model_name)


uploaded_file = st.file_uploader(
    "Choose an excel file",
    type=["xlsx", "xls"]
)


if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("Uploaded data")
    st.dataframe(df, use_container_width=True)

    # 필요한 컬럼 확인
    missing_volume_cols = [col for col in volume_features if col not in df.columns]
    missing_hardness_cols = [col for col in hardness_features if col not in df.columns]

    if missing_volume_cols or missing_hardness_cols:
        if missing_volume_cols:
            st.error(f"Volume 예측에 필요한 컬럼이 없습니다: {missing_volume_cols}")

        if missing_hardness_cols:
            st.error(f"Hardness 예측에 필요한 컬럼이 없습니다: {missing_hardness_cols}")

        st.write("Volume 모델 입력 컬럼:")
        st.write(volume_features)

        st.write("Hardness 모델 입력 컬럼:")
        st.write(hardness_features)

    else:
        # Volume prediction
        X_volume = df[volume_features].astype(float)
        X_volume_scaled = volume_x_scaler.transform(X_volume)

        pred_volume_scaled = volume_model.predict(X_volume_scaled)
        pred_volume_scaled = np.array(pred_volume_scaled).reshape(-1, 1)

        pred_volume = volume_y_scaler.inverse_transform(pred_volume_scaled).ravel()

        # Hardness prediction
        X_hardness = df[hardness_features].astype(float)
        X_hardness_scaled = hardness_x_scaler.transform(X_hardness)

        pred_hardness_scaled = hardness_model.predict(X_hardness_scaled)
        pred_hardness_scaled = np.array(pred_hardness_scaled).reshape(-1, 1)

        pred_hardness = hardness_y_scaler.inverse_transform(pred_hardness_scaled).ravel()

        # Result
        result_df = df.copy()
        result_df["Predicted_Volume"] = pred_volume
        result_df["Predicted_Hardness"] = pred_hardness

        st.subheader("Prediction result")
        st.dataframe(result_df, use_container_width=True)

        st.success(f"Selected model: {model_name}")
        st.success(f"평균 예측 Volume: {pred_volume.mean():.3f}")
        st.success(f"평균 예측 Hardness: {pred_hardness.mean():.3f}")
