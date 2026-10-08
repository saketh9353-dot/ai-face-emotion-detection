import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np

st.title("🤖 AI Face Emotion Detection")

st.write("Allow camera access and show your face.")

picture = st.camera_input("Take a picture")

if picture is not None:
    bytes_data = picture.getvalue()
    image = cv2.imdecode(
        np.frombuffer(bytes_data, np.uint8),
        cv2.IMREAD_COLOR
    )

    try:
        result = DeepFace.analyze(
            image,
            actions=["emotion"],
            enforce_detection=False
        )

        emotion = result[0]["dominant_emotion"]
        confidence = result[0]["emotion"][emotion]

        st.success(f"Emotion: {emotion}")
        st.info(f"Confidence: {confidence:.1f}%")

    except Exception as error:
        st.error(f"Analysis error: {error}")