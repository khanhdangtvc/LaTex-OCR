from PIL import Image
import streamlit as st
from ocr import LlamaOCR, hf_login

@st.cache_resource(show_spinner="Loading application")
def load_model():
    hf_login()
    model = LlamaOCR()

    return model



def image_uploader():
    image_max_size = 10*1024*1024
    uploaded_file = st.file_uploader("Upload an equation image", type =["png", "jpg", "jpeg", "webp"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        if uploaded_file.size >= image_max_size:
            image = image.thumbnail((1280, 1280))
        st.image(image)
        return image
    
def main():
    model = load_model()

    image = image_uploader()
    if image:
        latex = model.extract_latex(image)
        st.latex(latex)


if __name__ == "__main__":
    main()