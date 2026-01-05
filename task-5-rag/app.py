import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="Math Assistant", page_icon="🔢", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_rag():
    """Load documents and initialize the RAG chain once."""
    loader = PyPDFLoader("./data/string_theory.pdf")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_db = Chroma.from_documents(
        documents=chunks, embedding=local_embeddings, collection_name="string-rag"
    )

    llm = ChatOllama(model="llama3")

    template = """
    You are a specialized mathematical assistant. Use the following pieces of retrieved 
    context to answer the user's question. 

    1. If the context contains mathematical formulas, explain them step-by-step.
    2. If you don't know the answer based on the context, say that you don't know. 
    3. Provide as helpful of an answer as you can based on the provided context

    Context: {context}
    Question: {question}
    Helpful Answer:"""

    custom_prompt = PromptTemplate.from_template(template)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": custom_prompt},
    )


with st.sidebar:
    st.title("📚 RAG Settings")
    st.info("Currently searching: **string_theory.pdf**")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.title("String Theory Assistant")
st.caption("Powered by Llama3 and LangChain")

rag_chain = initialize_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about string theory equations..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and calculating..."):
            response = rag_chain.invoke(prompt)
            full_response = response["result"]
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
