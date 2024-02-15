import streamlit as st
from pypdf import PdfReader

st.title("File Reader with Streamlit")
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
print(uploaded_file)
file_contents = uploaded_file.read()
st.text(file_contents)
# # Read the Pdf file
# reader = PdfReader("/content/Furniture -Tender Document.pdf")

# # Clean text
# pdf_texts = [p.extract_text().strip() for p in reader.pages]

# # Ignore empty text
# pdf_texts = [text for text in pdf_texts if text]

