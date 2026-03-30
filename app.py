import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

st.title('Sentence Transformer')

text = st.text_area("Enter your text")

if st.button("Generate Embeddings"):
    if text:
        model = load_model()
        embedding = model.encode(text).tolist()
        st.write("Embedding: ")
        st.write(embedding)
    else:
        st.warning("Please enter the text")