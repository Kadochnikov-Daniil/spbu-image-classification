import streamlit as st
from PIL import Image

import segregator as seg

@st.cache(allow_output_mutation=True)
def init():
    return seg.initialize()

def main():
    st.title("Your FAV app!")

    st.write("If you are struggling with identifying what is on the picture, this app is for you!")
    st.write("This app configures the most appropriate decision based on SegFormer model.")
    st.write("Model source: https://huggingface.co/keras-io/semantic-segmentation")

    st.subheader("Choose a picture")
    input_picture = st.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=False, label_visibility="collapsed")
    col1, col2 = st.columns(2, gap="small")
    if input_picture is not None:
        image = Image.open(input_picture)
        with col1:
            st.subheader("Selected image")
            st.image(image, use_column_width=True)
            button = st.button("Apply", type="primary")
        if button:
            with st.spinner("Processing..."):
                segregator = init()
                results = seg.classify(segregator, image)
                with col2:
                    st.subheader("Results")
                    for x in results:
                        st.write(x["label"], x["score"])
    st.caption("by Daniil Kadochnikov")

if __name__ == "__main__":
    main()
