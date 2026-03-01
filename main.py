import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
import tiktoken
import streamlit as st
import time
import tempfile
from PIL import Image
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv

load_dotenv()

OCI_CONFIG_PATH      = os.getenv("OCI_CONFIG_PATH")
OCI_CONFIG_PROFILE   = os.getenv("OCI_CONFIG_PROFILE")
OCI_COMPARTMENT_ID   = os.getenv("OCI_COMPARTMENT_ID")
OCI_SERVICE_ENDPOINT = os.getenv("OCI_SERVICE_ENDPOINT")
OCI_COHERE_MODEL_ID  = os.getenv("OCI_COHERE_MODEL_ID")

if 'llm' not in st.session_state:
    st.session_state.llm = ChatOCIGenAI(
        model_id=OCI_COHERE_MODEL_ID,
        service_endpoint=OCI_SERVICE_ENDPOINT,
        compartment_id=OCI_COMPARTMENT_ID,
        auth_type="API_KEY",
        auth_profile=OCI_CONFIG_PROFILE,
        model_kwargs={"temperature": 0, "max_tokens": 2048},
        provider="cohere",
    )

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-multilingual-v3.0",
        service_endpoint=OCI_SERVICE_ENDPOINT,
        compartment_id=OCI_COMPARTMENT_ID,
        auth_type="API_KEY",
        auth_profile=OCI_CONFIG_PROFILE,
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'wind_logo' not in st.session_state:
    st.session_state.wind_logo = Image.open('comp logo.png')

custom_html = """
<div class="banner">
    <img src="https://wind-is.com/wp-content/uploads/2022/05/Original-Logo-01.png" width="100px" height="100px" alt="Banner Image">
</div>
<style>.banner { align: center; }</style>
"""
st.components.v1.html(custom_html)
st.title("Chat with Document")

def render_messages():
    for message in st.session_state.chat_history:
        avatar = st.session_state.wind_logo if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["message"])

def create_hybrid_chunks(docling_result, source_name):
    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.encoding_for_model("gpt-4o"),
        max_tokens=128 * 1024,
    )
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=512, merge_peers=True, include_headings=True)
    return [
        Document(page_content=chunk.text, metadata={"source": source_name, "chunk_index": i})
        for i, chunk in enumerate(chunker.chunk(dl_doc=docling_result.document))
    ]

def create_text_chunks(text_content, source_name):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return [Document(page_content=t, metadata={"source": source_name}) for t in splitter.split_text(text_content)]

def build_vectorstore(chunks):
    vectorstore = FAISS.from_documents(chunks, embedding=st.session_state.embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 15})

def get_conversational_answer(user_input, retriever):
    history_str = ""
    for msg in st.session_state.chat_history[-6:]:
        role = "Human" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['message']}\n"

    context = "\n\n".join([doc.page_content for doc in retriever.invoke(user_input)])

    prompt = f"""You are a helpful and professional AI assistant designed to answer user questions based on uploaded documents.

Rules:
1. You can respond to greetings.
2. Use the conversation history to understand context and follow-up questions.
3. If the question is unrelated to the document, reply: "I can only provide information based on the provided document."
4. Always respond in the same language as the user's question (English or Arabic).
5. Be concise but thorough. Omit null/nan/none values from answers.

Conversation History:
{history_str}

Context from document:
{context}

Current Question: {user_input}

Helpful Answer:"""

    return st.session_state.llm.invoke(prompt).content

if 'retriever' not in st.session_state:
    source_type = st.radio("Choose file type:", ["PDF", "Excel"])

    file_type = 'pdf' if source_type == "PDF" else ['xls', 'xlsx']
    file = st.file_uploader(f"Upload your {source_type}", type=file_type)

    if file:
        with st.status(f"Processing {source_type}..."):
            try:
                suffix = '.pdf' if source_type == "PDF" else os.path.splitext(file.name)[1]
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                tmp.write(file.read())
                tmp.close()

                converter = DocumentConverter()
                result = converter.convert(tmp.name)
                st.write("✓ File converted successfully")

                chunks = create_hybrid_chunks(result, file.name)

                if not chunks and source_type == "Excel":
                    import pandas as pd
                    content = []
                    for sheet in pd.ExcelFile(tmp.name).sheet_names:
                        df = pd.read_excel(tmp.name, sheet_name=sheet)
                        content.append(f"\n\n## Sheet: {sheet}\n")
                        if not df.empty:
                            content.append(" | ".join(str(c) for c in df.columns))
                            for _, row in df.iterrows():
                                content.append(" | ".join(str(v) for v in row.values))
                    chunks = create_text_chunks("\n".join(content), file.name)

                os.unlink(tmp.name)

                if not chunks:
                    st.error("Could not extract content from the file.")
                else:
                    st.write(f"✓ Created {len(chunks)} chunks for processing.")
                    st.session_state.retriever = build_vectorstore(chunks)
                    st.success("File processed! You can now start chatting.")
                    st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

else:
    if user_input := st.chat_input("You:"):
        render_messages()
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant", avatar=st.session_state.wind_logo):
            with st.spinner("Assistant is typing..."):
                response_text = get_conversational_answer(user_input, st.session_state.retriever)
            placeholder = st.empty()
            full_response = ""
            for chunk in response_text:
                full_response += chunk
                time.sleep(0.01)
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

        st.session_state.chat_history.append({"role": "user", "message": user_input})
        st.session_state.chat_history.append({"role": "assistant", "message": response_text})
        st.rerun()
    else:
        render_messages()