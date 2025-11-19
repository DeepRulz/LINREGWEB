import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Linear Regression")
st.write("Upload a dataset, choose features and target columns, and view linear regression results with plots.")

debug = st.sidebar
debug.title("Debug Info")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
debug.write(f"File uploaded: {uploaded_file is not None}")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        debug.write("CSV read successfully.")
        debug.write(f"DataFrame shape: {df.shape}")
        debug.write(f"Columns: {list(df.columns)}")
    except Exception as e:
        debug.error(f"Error reading CSV: {e}")
        st.stop()

    st.write("Dataset Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    debug.write(f"Numeric columns detected: {numeric_cols}")

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least two numeric columns.")
        debug.error("Not enough numeric columns.")
    else:
        target_col = st.selectbox("Select Target Column (y)", numeric_cols)
        feature_cols = st.multiselect(
            "Select Feature Columns (X)",
            [col for col in numeric_cols if col != target_col]
        )

        if not feature_cols:
            st.warning("Please select at least one feature column.")
            st.stop()

        debug.write(f"Selected target: {target_col}")
        debug.write(f"Selected features: {feature_cols}")

        X = df[feature_cols]
        y = df[target_col]

        debug.write(f"X shape: {X.shape}")
        debug.write(f"y shape: {y.shape}")

        test_size = st.slider("Test Size (%)", 5, 50, 20) / 100
        debug.write(f"Test size: {test_size}")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            debug.write(f"X_train shape: {X_train.shape}")
            debug.write(f"X_test shape: {X_test.shape}")
        except Exception as e:
            debug.error(f"Train/test split error: {e}")
            st.stop()

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        debug.write(f"Predictions shape: {y_pred.shape}")

        st.subheader("Model Performance")
        st.write(f"Coefficient(s): {dict(zip(feature_cols, model.coef_))}")
        st.write(f"Intercept: {model.intercept_:.4f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")

        debug.write(f"Model coefficients: {dict(zip(feature_cols, model.coef_))}")
        debug.write(f"Intercept: {model.intercept_}")

        st.subheader("Scatter Matrix")
        scatter_data = X_test.copy()
        scatter_data[target_col] = y_test
        fig = sns.pairplot(scatter_data)
        st.pyplot(fig)

        st.subheader("Residuals Plot")
        residuals = y_test - y_pred
        debug.write(f"Residuals sample: {residuals[:5].tolist()}")
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_pred, y=residuals, ax=ax)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        st.pyplot(fig)

        st.subheader("Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(6,6))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_aspect('equal', adjustable='box')
        st.pyplot(fig)

        debug.success("App executed successfully!")
