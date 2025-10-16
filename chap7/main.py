# __import__('pysqlite3') 
# import sys 
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# sqlite3 대신에 pysqlite3 사용 하는 임시적인 방법
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain import hub
import tempfile
load_dotenv()
import streamlit as st 
import os

# 제목 
st.title("ChatPDF") 
st.write("-------")

# 파일 업로드 
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"]) 
st.write("-------")
# loader 
# loader = PyPDFLoader("chap7/unsu.pdf")

def pdf_to_document(uploaded_file):
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_filepath = os.path.join(tmp_dir.name, uploaded_file.name) 
    with open (tmp_filepath, "wb") as f: 
        f.write(uploaded_file.getvalue()) 
    loader = PyPDFLoader(tmp_filepath)
    pages = loader.load_and_split()
    return pages  

if uploaded_file is not None: 
    pages = pdf_to_document(uploaded_file)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=20, 
        length_function=len
    )
    texts = text_splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = Chroma.from_documents(texts, embeddings)

    #User input 
    st.header("PDF에게 질문해보세요!!") 
    question = st.text_input("질문을 입력하세요")

    if st.button("질문하기"):
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(), 
            llm = llm
            ) 
        prompt = hub.pull("rlm/rag-prompt")

        #Retriever 
        
        #prompt template


        # Generate 
        def format_docs(docs): 
            return "\n\n".join(doc.page_content for doc in docs) 

        rag_chain = (
            {"context": retriever_from_llm | format_docs, 
            "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser() 
        )

        with st.spinner("wait for it ..."):
            result = rag_chain.invoke(question)
        st.write(result)
