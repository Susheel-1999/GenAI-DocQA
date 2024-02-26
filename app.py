import streamlit as st
import io
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
import os

# Environmental variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ''
os.environ["LLM"] = 'google/flan-t5-xxl'
os.environ["MAX_LENGTH"] = '64'
os.environ["EMB_MODEL"] = 'all-MiniLM-L6-v2'
os.environ["CHUNK_SIZE"] = '200'
os.environ["CHUNK_OVERLAP"] = '10'

llm = HuggingFaceHub(
    repo_id=os.environ.get("LLM"), 
    model_kwargs={"temperature": 0.5, "max_length": int(os.environ.get("MAX_LENGTH"))})

template = """
Try to answer the Question based on the Context.
Context: {context}
Question: {question}
Answer:"""

prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=llm)

embedding_model = HuggingFaceEmbeddings(model_name=os.environ.get("EMB_MODEL"))

def read_pdf(files):
    try:
        all_pdf_texts = ""
        for file_contents in files:
            reader = PdfReader(file_contents)
            pdf_texts = [p.extract_text().strip() for p in reader.pages]
            pdf_texts = [text for text in pdf_texts if text]
            all_pdf_texts += "\n\n".join(pdf_texts)
        return all_pdf_texts
    except Exception as e:
        print("Error faced in Read PDF -",e)
        return ""

def chunking(doc_text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = int(os.environ.get("CHUNK_SIZE")), chunk_overlap = int(os.environ.get("CHUNK_OVERLAP")),
                                                       length_function = len)
        chunks = text_splitter.split_text(doc_text)
        return chunks
    except Exception as e:
        print("Error faced in Chunking -",e)
        return doc_text

def vectorize_text(pdf_chunks):
    with st.spinner("Indexing into DB..."):
        return embedding_model.embed_documents(pdf_chunks)

def main():
    # UI
    st.set_page_config(page_title="inquiry")
    st.title("Document Inquiry Tool")
    st.caption("Document Inquiry Tool is designed to respond comprehensively to questions posed about the provided document, regardless of the section from which the questions originate.")
    st.subheader("Step 1 - Upload the Document")

    # File uploader
    uploaded_files = st.file_uploader("Choose a file", type=["pdf"], accept_multiple_files=True)
    pdf_chunks = []
    rerun_switch = False

    # Initialize session state
    if "ip_files" not in st.session_state:
        st.session_state.ip_files = []
        st.session_state.pdf_texts = ""
    if 'db' not in st.session_state:
        st.session_state.db = None
    if "pdf_chunks" not in st.session_state:
        st.session_state.pdf_chunks = []

    if uploaded_files != []:
        with st.spinner("Reading the file..."):
            if st.session_state.ip_files != uploaded_files:
                st.session_state.ip_files = uploaded_files
                st.session_state.pdf_texts = read_pdf(uploaded_files)
                rerun_switch = True # to reindex with all new files

        # Collapsible section for Preview
        with st.expander("click here to see the document content", expanded=False):
            st.text_area("Document Content Preview", st.session_state.pdf_texts, height=400)

        # Chunking
        with st.spinner("Chunking..."):
            if st.session_state.pdf_chunks == [] or rerun_switch:
                st.session_state.pdf_chunks = chunking(st.session_state.pdf_texts)
                st.session_state.pdf_chunks = list(map(lambda x: Document(x), st.session_state.pdf_chunks))

        # Vectorizing
        with st.spinner("Indexing into DB..."):
            if st.session_state.db is None or rerun_switch:
                st.session_state.db = FAISS.from_documents(st.session_state.pdf_chunks, embedding_model)
                rerun_switch = False

        # Section for user query
        st.subheader("Step 2 - Ask a Question")
        user_query = st.text_area("Type your question here", height=100)
        topn = st.session_state.db.similarity_search(user_query, fetch_k=5)

        # Fetch Answer Button
        if st.button("Find Answer"):
            with st.spinner("Generating..."):
                st.success(llm_chain.run({"question": user_query, "context": topn}))
    else:
        # Reset the DB
        ids = []
        for i in range(len(st.session_state.pdf_chunks)):
            try:
                ids.append(st.session_state.db.index_to_docstore_id[i])
            except:
                break
        try:
            st.session_state.db.delete(ids)
        except Exception as e:
            pass
        st.session_state.pdf_chunks = []
        st.session_state.db = None

if __name__ == '__main__':
    main()