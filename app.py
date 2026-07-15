import base64
from io import BytesIO
import os
import time
from openai import OpenAI
import streamlit as st
from PIL import Image

os.makedirs("uploads", exist_ok=True)

client = OpenAI( api_key=os.getenv("OPENAI_API_KEY"))

st.title("Image Captioning App")

st.write("Upload an image and get a caption for it using OpenAI's GPT-3 model.")
st.write("This app uses the OpenAI API to generate captions for images. Please make sure you have an API key set up in your environment variables.")

prompt = st.text_area("Describe your edit", "Coloque o neymar nessa foto sem alterar as caracteristicas originais da imagem, apenas adicione o menino ney na foto.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]) 

path = ""
if uploaded_file is not None:
    path = os.path.join("uploads", uploaded_file.name)
    image_ = Image.open(uploaded_file)
    st.image(image_, caption='Uploaded Image.')
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

result = None
if st.button("Generate"):
        with st.spinner('Loading data, please wait...'):
            st.session_state.result = client.images.edit(
            model="gpt-image-2-2026-04-21",
            image=open(path, "rb"),
            prompt=prompt
        )



if result in st.session_state:
    st.success('Done!')
    image_bytes = base64.b64decode(st.session_state.result.data[0].b64_json)
    image = Image.open(BytesIO(image_bytes))
    st.image(image)

    timestamp = time.time()
    os.makedirs("generated", exist_ok=True)
    with open(f"generated/menino_ney_generator_{timestamp}.png", "wb") as f:
        f.write(image_bytes)
