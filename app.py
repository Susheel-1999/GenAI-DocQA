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
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ''

llm = HuggingFaceHub(
    repo_id="google/flan-t5-xxl", 
    model_kwargs={"temperature": 0.5, "max_length": 64})

template = """
Try to answer the Question based on the Context.
Context: {context}
Question: {question}
Answer:"""

prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=llm)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def read_pdf(file_contents):
    try:
        reader = PdfReader(file_contents)
        pdf_texts = [p.extract_text().strip() for p in reader.pages]
        pdf_texts = [text for text in pdf_texts if text]
        pdf_texts = "\n\n".join(pdf_texts)
        return pdf_texts
    except Exception as e:
        print("Error faced in Read PDF -",e)
        return ""

def chunking(doc_text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 10,
                                                       length_function = len)
        chunks = text_splitter.split_text(doc_text)
        return chunks
    except Exception as e:
        prnt("Error faced in Chunking -",e)
        return doc_text

def vectorize_text(pdf_chunks):
    with st.spinner("Indexing into DB..."):
        return embedding_model.embed_documents(pdf_chunks)

def main():
    # UI
    st.set_page_config(page_title="documentQA")
    st.title("Document Q&A")
    st.caption("Document Q&A is designed to respond comprehensively to questions posed about the provided document, regardless of the section from which the questions originate.")
    st.subheader("Step 1 - Upload the Document")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

    if uploaded_file is not None:
        # Read PDF content
        file_contents = uploaded_file.read()
        file_contents = io.BytesIO(file_contents)
        
        with st.spinner("Reading the file..."):
            pdf_texts = read_pdf(file_contents)

        # Collapsible section for Preview
        with st.expander("click here to see the document content", expanded=False):
            st.text_area("Document Content Preview", pdf_texts, height=400)

        # Chunking
        with st.spinner("Chunking..."):
            pdf_chunks = chunking(pdf_texts)
            pdf_chunks = list(map(lambda x: Document(x), pdf_chunks))

        # Vectorizing
        with st.spinner("Indexing into DB..."):
            db = FAISS.from_documents(pdf_chunks, embedding_model)

        # Section for user query
        st.subheader("Step 2 - Ask a Question")
        user_query = st.text_area("Type your question here", height=100)
        topn = db.similarity_search(user_query, fetch_k=5)

        # Fetch Answer Button
        if st.button("Find Answer"):
            with st.spinner("Generating..."):
                st.success(llm_chain.run({"question": user_query, "context":topn}))

if __name__ == '__main__':
    main()