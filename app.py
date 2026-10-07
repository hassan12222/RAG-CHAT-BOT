
import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="RoadAnomaly Research AI",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# TOKEN / CONTEXT LIMITS
# ============================================================

MAX_CHAT_CONTEXT_CHARS = 14000
MAX_ANALYSIS_CONTEXT_CHARS = 12000
MAX_COMPARISON_CONTEXT_CHARS = 12000
MAX_CHARS_PER_CHUNK = 3000
MAX_CHARS_PER_PAPER = 2200


# ============================================================
# SESSION STATE
# ============================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "quick_question" not in st.session_state:
    st.session_state.quick_question = ""


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    try:

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

        return embeddings

    except Exception as e:

        st.error("Could not load the embedding model.")

        st.exception(e)

        return None


# ============================================================
# READ PDF
# ============================================================

def read_pdf(uploaded_file):

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.getbuffer())

            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)

        documents = loader.load()

        total_chars = sum(
            len(doc.page_content)
            for doc in documents
        )

        if total_chars < 50:

            raise ValueError(
                "The PDF does not contain enough readable text."
            )

        for doc in documents:

            doc.metadata["source_file"] = uploaded_file.name
            doc.metadata["document_type"] = "Research Paper"

        return documents

    except Exception as e:

        st.error(
            f"Could not read {uploaded_file.name}: {e}"
        )

        return []

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)


# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

def build_knowledge_base(uploaded_files):

    all_documents = []

    progress = st.progress(0)

    for index, uploaded_file in enumerate(uploaded_files):

        documents = read_pdf(uploaded_file)

        all_documents.extend(documents)

        progress.progress(
            int(((index + 1) / len(uploaded_files)) * 100)
        )

    if not all_documents:

        st.error("No readable documents were found.")

        return None, []

    # --------------------------------------------------------
    # SPLIT DOCUMENTS INTO CHUNKS
    # --------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(
        all_documents
    )

    # Add unique IDs
    for chunk in chunks:

        chunk.metadata["chunk_id"] = str(
            uuid.uuid4()
        )

    # --------------------------------------------------------
    # LOAD EMBEDDINGS
    # --------------------------------------------------------

    embeddings = load_embedding_model()

    if embeddings is None:

        return None, []

    # --------------------------------------------------------
    # CREATE CHROMA DATABASE
    # --------------------------------------------------------

    try:

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="road_anomaly_research"
        )

        st.success(
            f"Knowledge base created successfully with "
            f"{len(chunks)} chunks."
        )

        return vectorstore, chunks

    except Exception as e:

        st.error("Failed to build the knowledge base.")

        st.exception(e)

        return None, []


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    vectorstore,
    question,
    top_k=3,
    strategy="MMR"
):

    if vectorstore is None:

        return []

    top_k = min(top_k, 6)

    try:

        if strategy == "MMR":

            fetch_k = min(
                max(top_k * 3, 8),
                12
            )

            results = vectorstore.max_marginal_relevance_search(
                question,
                k=top_k,
                fetch_k=fetch_k
            )

        else:

            results = vectorstore.similarity_search(
                question,
                k=top_k
            )

        return results

    except Exception as e:

        st.error(
            f"Search failed: {e}"
        )

        return []


# ============================================================
# CREATE CHAT CONTEXT
# ============================================================

def create_context(documents):

    context_parts = []

    total_chars = 0

    for index, doc in enumerate(documents):

        text = doc.page_content[:MAX_CHARS_PER_CHUNK]

        source = doc.metadata.get(
            "source_file",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            0
        )

        try:

            page = int(page) + 1

        except:

            page = 1

        section = (
            f"[S{index + 1}] "
            f"Source: {source} | "
            f"Page: {page}\n"
            f"{text}"
        )

        if (
            total_chars + len(section)
            > MAX_CHAT_CONTEXT_CHARS
        ):

            break

        context_parts.append(section)

        total_chars += len(section)

    return "\n\n".join(context_parts)


# ============================================================
# DOCUMENT SAMPLE
# ============================================================

def document_sample(documents):

    text_parts = []

    total_chars = 0

    for doc in documents[:12]:

        text = doc.page_content[:3000]

        if (
            total_chars + len(text)
            > MAX_ANALYSIS_CONTEXT_CHARS
        ):

            break

        text_parts.append(text)

        total_chars += len(text)

    return "\n\n".join(text_parts)


# ============================================================
# COMPARISON SAMPLE
# ============================================================

def comparison_document_sample(
    documents,
    selected_papers
):

    grouped = {}

    for doc in documents:

        source = doc.metadata.get(
            "source_file",
            "Unknown"
        )

        if source not in selected_papers:

            continue

        if source not in grouped:

            grouped[source] = []

        grouped[source].append(doc)

    final_parts = []

    total_chars = 0

    for paper in selected_papers:

        paper_docs = grouped.get(
            paper,
            []
        )

        paper_text = ""

        for doc in paper_docs:

            remaining = (
                MAX_CHARS_PER_PAPER
                - len(paper_text)
            )

            if remaining <= 0:

                break

            paper_text += (
                doc.page_content[:remaining]
                + "\n"
            )

        paper_text = paper_text[
            :MAX_CHARS_PER_PAPER
        ]

        section = (
            f"===== {paper} =====\n"
            f"{paper_text}"
        )

        if (
            total_chars + len(section)
            > MAX_COMPARISON_CONTEXT_CHARS
        ):

            break

        final_parts.append(section)

        total_chars += len(section)

    return "\n\n".join(final_parts)


# ============================================================
# CREATE LLM
# ============================================================

def create_llm(
    api_key,
    model_name,
    temperature
):

    if not api_key:

        return None

    try:

        llm = ChatGroq(
            groq_api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=1200
        )

        return llm

    except Exception as e:

        st.error(
            f"Could not create Groq model: {e}"
        )

        return None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

st.sidebar.subheader("Groq Settings")

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)

model_name = st.sidebar.selectbox(
    "Model",
    [
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant"
    ]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1
)

st.sidebar.subheader("Search Settings")

search_strategy = st.sidebar.selectbox(
    "Search Strategy",
    [
        "MMR",
        "Similarity"
    ]
)

top_k = st.sidebar.slider(
    "Retrieved Chunks",
    min_value=1,
    max_value=6,
    value=3
)


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🚦 RoadAnomaly Research AI")

st.write(
    "A research assistant for analyzing "
    "AI-based road anomaly and accident detection papers."
)


# ============================================================
# PAPER UPLOAD
# ============================================================

st.header("📚 Research Paper Library")

uploaded_files = st.file_uploader(
    "Upload research papers in PDF format",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    st.write(
        f"{len(uploaded_files)} paper(s) selected."
    )

    for file in uploaded_files:

        st.write(
            f"📄 {file.name}"
        )

    if st.button(
        "Build Knowledge Base",
        type="primary"
    ):

        with st.spinner(
            "Reading papers and creating embeddings..."
        ):

            vectorstore, documents = (
                build_knowledge_base(
                    uploaded_files
                )
            )

            if vectorstore is not None:

                st.session_state.vectorstore = (
                    vectorstore
                )

                st.session_state.documents = (
                    documents
                )

                st.session_state.chat_history = []

                st.success(
                    "Your research library is ready."
                )


# ============================================================
# CHECK KNOWLEDGE BASE
# ============================================================

if st.session_state.vectorstore is None:

    st.info(
        "Upload research papers and click "
        "'Build Knowledge Base' to start."
    )

    st.stop()


documents = st.session_state.documents


# ============================================================
# METRICS
# ============================================================

papers = set()

for doc in documents:

    papers.add(
        doc.metadata.get(
            "source_file",
            "Unknown"
        )
    )

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Research Papers",
        len(papers)
    )

with col2:

    st.metric(
        "Pages",
        len(documents)
    )

with col3:

    st.metric(
        "Text Chunks",
        len(documents)
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "💬 Research Chat",
        "🔎 Semantic Search",
        "📊 Research Analysis",
        "🔬 Compare Papers",
        "🧪 RAG Lab"
    ]
)


# ============================================================
# RESEARCH CHAT
# ============================================================

with tab1:

    st.header("Research Chat")

    st.write(
        "Ask questions about the research papers "
        "you uploaded."
    )

    # --------------------------------------------------------
    # QUICK QUESTIONS
    # --------------------------------------------------------

    st.subheader("Quick Questions")

    quick_questions = [
        "What datasets were used?",
        "What models were used?",
        "What methodology was used?",
        "What were the main results?",
        "What research gaps were identified?"
    ]

    selected_question = st.selectbox(
        "Choose a question",
        ["Select a question"] + quick_questions
    )

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and "sources" in message
            ):

                st.subheader("Sources")

                for source in message["sources"]:

                    st.write(
                        source
                    )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    user_question = st.chat_input(
        "Ask something about your research papers..."
    )

    if (
        selected_question != "Select a question"
        and not user_question
    ):

        user_question = selected_question

    if user_question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):

            st.write(
                user_question
            )

        retrieved_docs = retrieve_documents(
            st.session_state.vectorstore,
            user_question,
            top_k,
            search_strategy
        )

        context = create_context(
            retrieved_docs
        )

        llm = create_llm(
            api_key,
            model_name,
            temperature
        )

        if llm is None:

            st.warning(
                "Enter your Groq API key in the sidebar."
            )

        else:

            prompt = f"""
You are a research assistant specializing in
AI-based road anomaly and accident detection.

Answer the user's question using ONLY the
research paper context provided below.

If the answer is not available in the papers,
say that the information was not found in
the uploaded papers.

Be concise and research-focused.

Always mention source labels such as [S1],
[S2] when using information from the papers.

Research Context:

{context}

User Question:

{user_question}
"""

            try:

                with st.chat_message("assistant"):

                    with st.spinner(
                        "Analyzing the papers..."
                    ):

                        response = llm.invoke(
                            prompt
                        )

                        answer = response.content

                    st.write(answer)

                    st.subheader("Sources")

                    sources = []

                    for index, doc in enumerate(
                        retrieved_docs
                    ):

                        source = doc.metadata.get(
                            "source_file",
                            "Unknown"
                        )

                        page = doc.metadata.get(
                            "page",
                            0
                        )

                        try:

                            page = int(page) + 1

                        except:

                            page = 1

                        source_text = (
                            f"[S{index + 1}] "
                            f"{source} — "
                            f"Page {page}"
                        )

                        sources.append(
                            source_text
                        )

                        st.write(
                            source_text
                        )

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )

            except Exception as e:

                if "413" in str(e):

                    st.error(
                        "The request was too large. "
                        "Try reducing Retrieved Chunks "
                        "to 2 or 3."
                    )

                else:

                    st.error(
                        f"Error generating answer: {e}"
                    )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

with tab2:

    st.header("🔎 Semantic Search")

    st.write(
        "Find the most relevant sections from "
        "your research papers."
    )

    search_query = st.text_input(
        "Enter your research question"
    )

    if st.button(
        "Search Papers"
    ):

        if not search_query:

            st.warning(
                "Please enter a search question."
            )

        else:

            results = retrieve_documents(
                st.session_state.vectorstore,
                search_query,
                top_k,
                search_strategy
            )

            st.write(
                f"Found {len(results)} relevant sections."
            )

            for index, doc in enumerate(
                results
            ):

                source = doc.metadata.get(
                    "source_file",
                    "Unknown"
                )

                page = doc.metadata.get(
                    "page",
                    0
                )

                try:

                    page = int(page) + 1

                except:

                    page = 1

                with st.expander(
                    f"[S{index + 1}] "
                    f"{source} — Page {page}"
                ):

                    st.write(
                        doc.page_content
                    )


# ============================================================
# RESEARCH ANALYSIS
# ============================================================

with tab3:

    st.header("📊 Research Analysis")

    st.write(
        "Generate a general analysis of the "
        "uploaded research papers."
    )

    analysis_type = st.selectbox(
        "Analysis Type",
        [
            "Summary",
            "Datasets",
            "Models and Techniques",
            "Results",
            "Research Gaps"
        ]
    )

    if st.button(
        "Generate Analysis"
    ):

        if not api_key:

            st.warning(
                "Enter your Groq API key in the sidebar."
            )

        else:

            context = document_sample(
                documents
            )

            llm = create_llm(
                api_key,
                model_name,
                temperature
            )

            prompt = f"""
You are analyzing research papers about
AI-based road anomaly and accident detection.

Analyze the papers based on:

{analysis_type}

Use only the information provided below.

Give a clear and concise research-oriented answer.

Research Papers:

{context}
"""

            try:

                with st.spinner(
                    "Analyzing research papers..."
                ):

                    response = llm.invoke(
                        prompt
                    )

                st.write(
                    response.content
                )

            except Exception as e:

                st.error(
                    f"Analysis failed: {e}"
                )


# ============================================================
# COMPARE PAPERS
# ============================================================

with tab4:

    st.header("🔬 Compare Research Papers")

    st.write(
        "Compare different research papers "
        "based on their methods, datasets, "
        "models and results."
    )

    paper_names = sorted(
        list(papers)
    )

    selected_papers = st.multiselect(
        "Select papers to compare",
        paper_names
    )

    comparison_topic = st.selectbox(
        "Compare By",
        [
            "Methodology",
            "Dataset",
            "Models",
            "Results",
            "Limitations",
            "Overall Comparison"
        ]
    )

    if st.button(
        "Compare Papers"
    ):

        if len(selected_papers) < 2:

            st.warning(
                "Please select at least two papers."
            )

        elif not api_key:

            st.warning(
                "Enter your Groq API key in the sidebar."
            )

        else:

            context = comparison_document_sample(
                documents,
                selected_papers
            )

            llm = create_llm(
                api_key,
                model_name,
                temperature
            )

            prompt = f"""
Compare the selected research papers about
AI-based road anomaly and accident detection.

Comparison topic:

{comparison_topic}

Selected Papers:

{context}

Create a clear comparison.

Mention each paper separately.

Do not invent information that is not
available in the provided context.
"""

            try:

                with st.spinner(
                    "Comparing papers..."
                ):

                    response = llm.invoke(
                        prompt
                    )

                st.write(
                    response.content
                )

            except Exception as e:

                if "413" in str(e):

                    st.error(
                        "The comparison request is too large. "
                        "Try selecting fewer papers."
                    )

                else:

                    st.error(
                        f"Comparison failed: {e}"
                    )


# ============================================================
# RAG LAB
# ============================================================

with tab5:

    st.header("🧪 RAG Lab")

    st.write(
        "Test how different retrieval strategies "
        "affect the retrieved research sections."
    )

    lab_question = st.text_input(
        "Enter a research question for the RAG test"
    )

    if st.button(
        "Run RAG Test"
    ):

        if not lab_question:

            st.warning(
                "Enter a question first."
            )

        else:

            st.subheader(
                "Similarity Search"
            )

            similarity_results = retrieve_documents(
                st.session_state.vectorstore,
                lab_question,
                top_k,
                "Similarity"
            )

            for index, doc in enumerate(
                similarity_results
            ):

                source = doc.metadata.get(
                    "source_file",
                    "Unknown"
                )

                page = doc.metadata.get(
                    "page",
                    0
                )

                try:

                    page = int(page) + 1

                except:

                    page = 1

                st.write(
                    f"{index + 1}. "
                    f"{source} — Page {page}"
                )

            st.subheader(
                "MMR Search"
            )

            mmr_results = retrieve_documents(
                st.session_state.vectorstore,
                lab_question,
                top_k,
                "MMR"
            )

            for index, doc in enumerate(
                mmr_results
            ):

                source = doc.metadata.get(
                    "source_file",
                    "Unknown"
                )

                page = doc.metadata.get(
                    "page",
                    0
                )

                try:

                    page = int(page) + 1

                except:

                    page = 1

                st.write(
                    f"{index + 1}. "
                    f"{source} — Page {page}"
                )


# ============================================================
# FOOTER INFORMATION
# ============================================================

st.divider()

st.write(
    "RoadAnomaly Research AI | "
    "RAG-based research assistant for "
    "AI road anomaly detection studies."
)
