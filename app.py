from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

print("API KEY =", os.getenv("GOOGLE_API_KEY"))

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

gemini_model = genai.GenerativeModel("models/gemini-3.5-flash")

def extract_text_from_pdf(pdf_file):
    text = ""

    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")

    for page in pdf:
        text += page.get_text()

    return text


def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(text)

    return chunks


def create_vector_store(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings).astype("float32"))

    return model, index, chunks


st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎤",
    layout="centered"
)
def search_resume(question, model, index, chunks):

    question_embedding = model.encode([question])

    distances, indices = index.search(
        np.array(question_embedding).astype("float32"),
        k=3
    )

    results = []

    for idx in indices[0]:
        results.append(chunks[idx])

    return results
def ask_gemini(question, context):

    prompt = f"""
You are an AI Interview Coach.

Use ONLY the context below to answer.

Context:
{context}

Question:
{question}
"""

    response = gemini_model.generate_content(prompt)

    return response.text

st.title("🎤 AI Interview Coach")

st.write("Welcome! Upload your resume and practice interviews with AI.")

uploaded_file = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)
if uploaded_file is not None:

    st.success("✅ Resume uploaded successfully!")

    resume_text = extract_text_from_pdf(uploaded_file)

    chunks = chunk_text(resume_text)

    model, index, chunks = create_vector_store(chunks)

    st.success("✅ Resume indexed successfully!")

    st.subheader("Resume Chunks")

    for i, chunk in enumerate(chunks):
        st.write(f"### Chunk {i+1}")
        st.write(chunk)

    st.write(f"Total Chunks: {len(chunks)}")

    st.success("🎉 RAG Vector Database Created!")

    st.divider()

    if st.button("🎤 Start Interview"):

        context = "\n\n".join(chunks)

        prompt = f"""
You are a professional HR interviewer.

This is the candidate's resume:

{context}

Generate ONLY the first interview question.

Do not answer it.
Do not ask multiple questions.
Ask only one interview question.
"""

        response = gemini_model.generate_content(prompt)

        st.subheader("🎤 Interview Question")

        st.write(response.text)