from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
import os
from dotenv import load_dotenv

# -----------------------------
# GEMINI CONFIGURATION
# -----------------------------

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

gemini_model = genai.GenerativeModel(
    "models/gemini-3.5-flash"
)

# -----------------------------
# PDF FUNCTIONS
# -----------------------------

def extract_text_from_pdf(pdf_file):

    text = ""

    pdf = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf"
    )

    for page in pdf:
        text += page.get_text()

    return text


# -----------------------------
# CHUNKING
# -----------------------------

def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    return chunks


# -----------------------------
# VECTOR DATABASE
# -----------------------------

def create_vector_store(chunks):

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    return model, index


# -----------------------------
# SEARCH
# -----------------------------

def search_resume(question, model, index, chunks):

    question_embedding = model.encode([question])

    distances, indices = index.search(
        np.array(question_embedding).astype("float32"),
        k=5
    )

    retrieved_chunks = []

    for idx in indices[0]:
        retrieved_chunks.append(chunks[idx])

    return "\n\n".join(retrieved_chunks)
# -----------------------------
# GEMINI FUNCTIONS
# -----------------------------

def generate_interview_questions(context):

    prompt = f"""
You are a professional HR interviewer.

Below is a candidate's resume.

Resume:
{context}

Generate exactly 10 interview questions.

Rules:
- Ask one question per line.
- Number them from 1 to 10.
- Mix HR, Technical, Project, Behavioral and Problem Solving questions.
- Do NOT provide answers.
"""

    response = gemini_model.generate_content(prompt)

    questions = []

    for line in response.text.split("\n"):

        line = line.strip()

        if line != "":
            questions.append(line)

    return questions


def evaluate_interview(context, questions, answers):

    qa = ""

    for q, a in zip(questions, answers):

        qa += f"""

Question:
{q}

Answer:
{a}

"""

    prompt = f"""
You are a Senior Technical Interviewer.

Candidate Resume:

{context}

Below are the interview questions and answers.

{qa}

Evaluate the candidate.

Return your response in this format.

Overall Score: /100

Technical Knowledge:
/10

Communication:
/10

Confidence:
/10

Problem Solving:
/10

HR Skills:
/10

Strengths:
- Point 1
- Point 2
- Point 3

Weaknesses:
- Point 1
- Point 2
- Point 3

Suggestions:
- Point 1
- Point 2
- Point 3

Finally tell whether the candidate is:

Selected
or
Rejected

with one paragraph explaining why.
"""

    response = gemini_model.generate_content(prompt)

    return response.text


# -----------------------------
# STREAMLIT PAGE
# -----------------------------

st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎤",
    layout="wide"
)


st.title("🎤 AI Interview Coach")

st.write(
    "Upload your resume and take a complete AI Interview."
)

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)
if uploaded_file is not None:

    st.success("✅ Resume uploaded successfully!")

    resume_text = extract_text_from_pdf(uploaded_file)

    chunks = chunk_text(resume_text)

    model, index = create_vector_store(chunks)

    context = "\n\n".join(chunks)

    st.success("✅ Resume Indexed Successfully!")

    st.write(f"Total Chunks : {len(chunks)}")

    st.divider()

    if st.button("🎤 Generate Interview"):

        with st.spinner("Generating Interview Questions..."):

            questions = generate_interview_questions(context)

        st.session_state["questions"] = questions

        st.session_state["context"] = context

    if "questions" in st.session_state:

        st.subheader("📋 Interview Questions")

        answers = []

        for i, question in enumerate(st.session_state["questions"]):

            st.markdown(f"### {question}")

            answer = st.text_area(
                f"Answer {i+1}",
                key=f"answer_{i}"
            )

            answers.append(answer)

        st.session_state["answers"] = answers

        st.divider()

        if st.button("📊 Evaluate Interview"):

            with st.spinner("Evaluating Interview..."):

                report = evaluate_interview(
                    st.session_state["context"],
                    st.session_state["questions"],
                    st.session_state["answers"]
                )

            st.session_state["report"] = report
            # -----------------------------
# SHOW FINAL REPORT
# -----------------------------

if "report" in st.session_state:

    st.divider()

    st.subheader("📈 Interview Evaluation Report")

    st.markdown(st.session_state["report"])

    st.balloons()

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("AI Interview Coach")

st.sidebar.info("""
Features Included

✅ Resume Upload

✅ PDF Text Extraction

✅ Chunking

✅ Sentence Transformers Embeddings

✅ FAISS Vector Database

✅ Retrieval-Augmented Generation (RAG)

✅ Gemini 3.5 Flash

✅ 10 AI Generated Interview Questions

✅ 10 Answer Boxes

✅ Complete Interview Evaluation

✅ Overall Score

✅ Technical Score

✅ HR Score

✅ Communication Score

✅ Strengths

✅ Weaknesses

✅ Suggestions

""")

st.sidebar.markdown("---")

st.sidebar.success("Version 2.0")

st.sidebar.caption(
    "Built using Streamlit, FAISS, Sentence Transformers and Gemini"
)