
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 AI Financial Fraud Detection System")
st.write("Multi-dataset fraud analytics and detection dashboard")

st.divider()

# Upload dataset
uploaded_file = st.file_uploader(
    "Upload any transaction CSV dataset",
    type=["csv"]
)

# Common fraud target column names
TARGET_NAMES = [
    "Class",
    "class",
    "is_fraud",
    "isFraud",
    "fraud",
    "Fraud",
    "fraud_label",
    "Fraud_Label",
    "Fraudulent",
    "is_fraudulent",
    "label",
    "target"
]

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Total Features", len(df.columns))
        col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")

        st.divider()

        # Dataset preview
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Detect target column
        matching_columns = [
            col for col in df.columns
            if str(col).strip().lower() in
            [name.lower() for name in TARGET_NAMES]
        ]

        st.subheader("🎯 Fraud Label Detection")

        if matching_columns:

            target = st.selectbox(
                "Select fraud target column",
                matching_columns
            )

            st.success(f"Target column identified: {target}")

            st.write("Target distribution:")
            st.dataframe(
                df[target].value_counts().rename_axis(
                    "Label"
                ).reset_index(name="Count")
            )

            # Display fraud distribution for binary labels
            if df[target].nunique(dropna=True) == 2:

                counts = df[target].value_counts()

                fig, ax = plt.subplots()

                counts.plot(
                    kind="bar",
                    ax=ax,
                    title="Transaction Label Distribution"
                )

                ax.set_xlabel("Transaction Label")
                ax.set_ylabel("Number of Transactions")

                st.pyplot(fig)
                plt.close(fig)

                st.info(
                    "The dataset contains two target classes. "
                    "Verify which class represents fraud before "
                    "calculating fraud statistics."
                )

            else:
                st.warning(
                    "Multiple target classes detected. "
                    "Please verify the fraud label definition."
                )

        else:

            st.warning(
                "No common fraud target column was detected."
            )

            manual_target = st.selectbox(
                "Select target column manually (if available)",
                ["None"] + list(df.columns)
            )

            if manual_target != "None":
                st.write(
                    df[manual_target].value_counts()
                )

        st.divider()

        # Model compatibility
        st.subheader("🤖 Model Compatibility")

        required_features = [
            "Time"
        ] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

        if all(col in df.columns for col in required_features):

            st.success(
                "Dataset contains the 30 expected model features."
            )

            st.info(
                "Feature compatibility is confirmed, but predictions "
                "also require the saved Random Forest model and scaler."
            )

        else:

            st.warning(
                "This dataset has a different feature structure. "
                "The existing Random Forest model cannot directly "
                "predict on it. A compatible model must be trained."
            )

    except Exception as e:

        st.error(f"Unable to process dataset: {e}")

else:

    st.info("Upload a CSV dataset to begin fraud analysis.")
