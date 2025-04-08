import streamlit as st
from deepfake_detector import DeepfakeDetector
from tensorflow.keras.models import load_model
import os
import tempfile

# Initialize detector class
detector = DeepfakeDetector()
MODEL_PATH = "deepfake_model.h5"

# Load the model if it exists
if os.path.exists(MODEL_PATH):
    detector.model = load_model(MODEL_PATH)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "History", "About", "Settings"])

# Dashboard Page
if page == "Dashboard":
    st.title("🔍 Deepfake Detector Dashboard")

    # Layout: left for upload, right for display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Media")
        option = st.radio("Choose input type:", ("Image", "Video"))

        if option == "Image":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        elif option == "Video":
            uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    with col2:
        st.subheader("Result")

        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                media_path = tmp_file.name

            if option == "Image":
                st.image(media_path, caption="Uploaded Image", use_column_width=True)
                result = detector.predict(media_path)
                st.success(f"Prediction: {result}")
            elif option == "Video":
                st.video(media_path)
                with st.spinner("Analyzing video..."):
                    result = detector.predict_video(media_path)
                st.success(f"Prediction: {result}")

            os.remove(media_path)

# History Page
elif page == "History":
    st.title("📜 Detection History")
    st.info("This section can display the history of analyzed media files and their results.")

# About Page
elif page == "About":
    st.title("ℹ️ About Deepfake Detection")
    st.write("""
    Deepfake detection involves identifying synthetic media where a person in an existing image or video is replaced with someone else's likeness. 
    Advanced machine learning techniques are used to create and detect these forgeries.
    """)

    st.subheader("About Anurag Pandey")
    st.write("""
    Anurag Pandey is a passionate developer and data scientist with a keen interest in machine learning and computer vision. 
             """)
    st.markdown("[LinkedIn Profile]")

# Settings Page
elif page == "Settings":
    st.title("⚙️ Settings")
    st.info("This section can include configurable settings for the application, such as model parameters, user preferences, etc.")
