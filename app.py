import streamlit as st 
from sentence_transformers import SentenceTransformer

def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

st.title('Sentence Transformer')

text = st.text_area("Enter your text")

if st.button("Generate Embeddigns"):
    if text:
        embedding =model.encode(text).tolist()
        st.write("Embedding: ")
        st.write(embedding)
    else:
        st.warning("Please enter the text")