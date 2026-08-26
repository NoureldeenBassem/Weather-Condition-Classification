"""
Weather Condition Classification - Streamlit app
Author: Eng. Noureldin Bassem Mohamed

Upload a photo from a weather camera and the model says which of the eleven
conditions it shows: dew, fogsmog, frost, glaze, hail, lightning, rain,
rainbow, rime, sandstorm, snow.

The model is the one trained in weather-classification.ipynb, it is only loaded
here, never retrained. Preprocessing (mobilenet_v2 rescaling) is baked inside
the saved model, so the only thing this file does to an uploaded photo is
convert it to RGB and resize it to the training size.
"""

import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weather_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "class_names.json")

# below this the model is not really sure, in the real product these photos
# would go to the review team instead of being tagged automatically
REVIEW_THRESHOLD = 0.70


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        meta = json.load(f)
    return model, meta["class_names"], tuple(meta["img_size"])


def predict(model, image, img_size):
    img = image.convert("RGB").resize(img_size)
    batch = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    return model.predict(batch, verbose=0)[0]


st.set_page_config(page_title="Weather Condition Classification", page_icon="🌦️")

st.title("🌦️ Weather Condition Classification")
st.write(
    "Upload a photo from a weather camera and the model will tag it with one of "
    "eleven weather conditions. Trained on the Weather Image Recognition dataset "
    "with MobileNetV2 transfer learning."
)

model, class_names, img_size = load_model()

uploaded = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Waiting for a photo. The classes are: " + ", ".join(class_names) + ".")
else:
    image = Image.open(uploaded)

    col_image, col_result = st.columns(2)

    with col_image:
        st.image(image, caption="uploaded photo", use_container_width=True)

    probs = predict(model, image, img_size)
    top = int(probs.argmax())
    confidence = float(probs[top])

    with col_result:
        st.subheader(class_names[top])
        st.metric("confidence", f"{confidence:.1%}")

        if confidence < REVIEW_THRESHOLD:
            st.warning("The model is not confident here, this one would go to a human reviewer.")
        else:
            st.success("Confident enough to tag automatically.")

    st.write("**All eleven classes**")
    st.bar_chart({"probability": {name: float(p) for name, p in zip(class_names, probs)}})

    ranked = sorted(zip(class_names, probs), key=lambda pair: pair[1], reverse=True)
    st.write(" · ".join(f"{name}: {p:.1%}" for name, p in ranked[:5]))
