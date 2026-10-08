import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np

st.set_page_config(
    page_title="AI Face Emotion Detection",
    page_icon="🤖"
)

st.title("🤖 AI Face Emotion Detection")
st.write("Detect emotions from multiple faces.")

picture = st.camera_input("📷 Take a picture")

if picture is not None:

    bytes_data = picture.getvalue()

    image = cv2.imdecode(
        np.frombuffer(bytes_data, np.uint8),
        cv2.IMREAD_COLOR
    )

    try:
        # Analyze all visible faces
        results = DeepFace.analyze(
            img_path=image,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            align=True
        )

        if not isinstance(results, list):
            results = [results]

        face_count = 0

        for result in results:

            region = result.get("region", {})

            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 0)
            h = region.get("h", 0)

            if w <= 0 or h <= 0:
                continue

            face_count += 1

            emotions = result["emotion"]
            dominant = result["dominant_emotion"]

            confidence = emotions[dominant]

            # Draw face box
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            label = f"Face {face_count}: {dominant} {confidence:.1f}%"

            cv2.putText(
                image,
                label,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        # Display image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(
            image_rgb,
            caption=f"Detected {face_count} face(s)",
            use_container_width=True
        )

        # Emotion details
        st.subheader("🎭 Emotion Analysis")

        for i, result in enumerate(results):

            region = result.get("region", {})
            w = region.get("w", 0)
            h = region.get("h", 0)

            if w <= 0 or h <= 0:
                continue

            emotions = result["emotion"]
            dominant = result["dominant_emotion"]

            st.markdown(f"### 👤 Face {i + 1}")
            st.success(
                f"Dominant emotion: **{dominant.upper()}**"
            )

            # Show every emotion
            for emotion, score in sorted(
                emotions.items(),
                key=lambda item: item[1],
                reverse=True
            ):
                st.write(
                    f"**{emotion.capitalize()}** — {score:.1f}%"
                )
                st.progress(
                    min(int(score), 100)
                )

    except Exception as error:
        st.error(f"Analysis error: {error}")
