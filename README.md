# GenAI-Document Q&A
Document Q&A is designed to respond comprehensively to questions posed about the provided document, regardless of the section from which the questions originate.

# Steps to run Streamlit app:
1. To create a Hugging face user access tokens or use an existing one, visit: https://huggingface.co/settings/tokens.
![image](https://github.com/Susheel-1999/GenAI-DocumentQA/assets/63583210/5e58ac63-4fd8-4f81-9aef-bedc6d8c169d)

2. Create a new environment: <br>
```conda create -p genai python==3.9 -y```

3. Activate the environment: <br>
```conda activate genai```

4. Install the requirements: <br>
```pip install -r requirements.txt```

5. Run the Streamlit application: <br>
```streamlit run app.py```

# Workflow:
1. Upload one or more PDF files. It will take little time to load. At backend, it will process, read, chunk and index the pdf files.
2. We can able to see the preview of the content. Expand to look into the content.
3. Ask the question that we have to know from the documents.

**Quick start:** https://huggingface.co/spaces/susheel-1999/documentQA

# About Techniques:
Langchain is a framework for developing applications powered by language models. It enables applications that are context-aware and reason.
1. **Chunking process** - It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.<br>```from langchain.text_splitter import RecursiveCharacterTextSplitter```<br><br>
   **Types of chunking:**<br>
     i) _Character Text Splitter_ - Splitting text based on the characters. <br>
     ii) _Recursive Character Text character_ -Text is split based on sequences of characters. This method is particularly effective for retaining the structure of paragraphs and sentences.<br>
     iii) _Document Based Splitter_ - Text is split based on the structure of documents. This approach caters to specific document formats, such as Python-based documents, HTML, markup, and more.<br>
     iv) _Semantic Chunking_ - Aims to identify points in the text where sentence similarity varies significantly (potentially with a threshold while considering the following sentence). These identified points serve as separators for creating meaningful chunks.<br>
2. **Integration of Hugging Face Models and Embeddings** - Langchain seamlessly incorporates and provides access to Hugging Face models and embeddings. Users can leverage the following functionalities.
   <br>_Emebeddings:_  ```from langchain_community.embeddings import HuggingFaceEmbeddings```
   <br>_LLMs:_  ```from langchain_community.llms import HuggingFaceHub```
3. **Integration of VectorDB** - Langchain seamlessly incorporates and provides supports for many VectorDB. <br>FAISS - (Facebook AI Similarity Search) is a library developed by Facebook AI Research specifically for efficient similarity search and clustering of dense vectors. It's particularly useful in applications involving large-scale vector search, where you need to find the nearest neighbors of a given vector among a massive dataset. It have support to various index type, optimized for both CPU and GPU and designed to handle billions of vectors efficiently, making it highly scalable. Example usecases: Image search, document search, recommendation engine and etc. <br> ```from langchain_community.vectorstores import FAISS```
4. **Schema** - Class for storing a piece of text and associated metadata. TO conver <br> ```from langchain.schema import Document```
5. **Prompt Template** - A template of a prompt can be easily designed with the help of the PromptTemplate class.<br> ```from langchain.prompts import PromptTemplate```
6. **LLM chain** - The LLMChain class is used to execute the PromptTemplate. <br> ```from langchain.chains import LLMChain```

Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. In just a few minutes we can build and deploy powerful data apps. 
1. **Session State** - Session State is a way to share variables between reruns, for each user session.

## Why Reterival Augmented Technique for Question Answering Task or any task?:
1. **Technique 1: Stuff** <br> Uses ALL of the text from the documents in the prompt.  It actually doesn’t work in Scenario where the data exceeds the token limit and causes rate-limiting errors.
2. **Technique 2: map_reduce**<br> It separates texts into batches, feeds each batch with the question to LLM separately, and comes up with the final answer based on the answers from each batch.
3. **Technique 3: refine**<br> It separates texts into batches, feeds the first batch to LLM, and feeds the answer and the second batch to LLM. It refines the answer by going through all the batches.
4. **Technique 4: map-rerank**<br>  It separates texts into batches, feeds each batch to LLM, returns a score of how fully it answers the question, and comes up with the final answer based on the high-scored answers from each batch. <br> One issue with using Technique 1, 2, 3, and 4 are that it can be very costly because you are feeding more text and multiple hits to OpenAI API and the API is charged by the number of tokens. A better solution is RAG (Retrieval Augmented Generation) which retrieve relevant text chunks first and only use the relevant text chunks in the language model. 
5. **Technique 5: RAG**  <br> Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response. <br>
![image](https://github.com/Susheel-1999/GenAI-DocumentQA/assets/63583210/a55e8fbb-6c75-4e8c-9104-df6f1e00f614)
_**Steps involved:**_ <br>
  i. Document Indexing into VectorDB <br>
  ii. Data Retriever <br>
  iii. Data Augmentation and Prompt Engineering <br>
  iv. Querying <br>

# Reference:
Langchain - https://python.langchain.com/docs/get_started/introduction  <br>
OpenAI - https://platform.openai.com/docs/introduction <br>
Streamlit - https://docs.streamlit.io/library/api-reference/session-state
