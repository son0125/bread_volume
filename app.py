import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="Bread Volume Prediction",
    layout="wide"
)

st.title("Artificial Intelligence Prediction of Bread Qualities (Volume)")
st.caption("made by Korea Food Research Institute and Sejong Univ.")

st.write(
    "(Hyperspectral imaging 데이터를 이용하여 "
    "빵의 volume을 예측하는 모델입니다.)"
)


@st.cache_resource
def load_model_files(model_name):
    model = joblib.load(f"{model_name}_volume_model.pkl")
    x_scaler = joblib.load(f"x_scaler_{model_name}_volume.pkl")
    y_scaler = joblib.load(f"y_scaler_{model_name}_volume.pkl")
    feature_columns = joblib.load("feature_columns_volume.pkl")
    return model, x_scaler, y_scaler, feature_columns


model_name = st.selectbox(
    "Choose a prediction model",
    ["CatBoost", "GBM", "AdaBoost"]
)

model, x_scaler, y_scaler, feature_columns = load_model_files(model_name)

uploaded_file = st.file_uploader(
    "Choose an excel file",
    type=["xlsx", "xls"]
)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("Uploaded data")
    st.dataframe(df, use_container_width=True)

    missing_cols = [col for col in feature_columns if col not in df.columns]

    if missing_cols:
        st.error(f"엑셀 파일에 다음 컬럼이 없습니다: {missing_cols}")
        st.write("모델 학습에 사용된 입력 컬럼은 아래와 같습니다.")
        st.write(feature_columns)

    else:
        X = df[feature_columns].astype(float)

        X_scaled = x_scaler.transform(X)

        pred_scaled = model.predict(X_scaled)
        pred_scaled = np.array(pred_scaled).reshape(-1, 1)

        pred_volume = y_scaler.inverse_transform(pred_scaled).ravel()

        result_df = df.copy()
        result_df["Predicted_Volume"] = pred_volume

        st.subheader("Prediction result")
        st.dataframe(result_df, use_container_width=True)

        st.success(f"평균 예측 Volume: {pred_volume.mean():.3f}")

        if "Sample" in result_df.columns:
            st.subheader("Sample-level prediction summary")

            summary_df = (
                result_df
                .groupby("Sample", as_index=False)["Predicted_Volume"]
                .mean()
            )

            st.dataframe(summary_df, use_container_width=True)

            summary_csv = summary_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="Download sample-level result",
                data=summary_csv,
                file_name="bread_volume_prediction_summary.csv",
                mime="text/csv"
            )

        result_csv = result_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Download full prediction result",
            data=result_csv,
            file_name="bread_volume_prediction_result.csv",
            mime="text/csv"
        )
