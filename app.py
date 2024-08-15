import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
import tempfile
##Environment Initilisation
load_dotenv()
os.environ['LANCHAIN_API_KEY'] = os.getenv('LANCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
groq_api = os.getenv('GROQ_API_KEY')


## This is Prompt Template for chatinng with doc    
prompt = ChatPromptTemplate.from_template("""
Answer to the Question from the document only
<context>
{context}
</context>
Question : {input}                                                                                                                                                                        
""")

##File from user
pdf = st.file_uploader("Upload your PDF document",type=('pdf'),key='pdf') 
if pdf !=None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf.read())
        tmp_file_path = tmp_file.name

    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(tmp_file_path)
    pdf = loader.load()

    st.session_state.box = st.selectbox('Which model',['llama-3.1-70b-versatile','llama-3.1-405b-reasoning',
                                                       'gemma2-9b-it','mixtral-8x7b-32768',''])
button = st.button('Start Embedding')
    

if button:
    st.write('Progress')
    bar = st.progress(0,text='Progess..')
    st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size = 100,chunk_overlap=10)
    bar.progress(10,text='Progress..')
    docs = st.session_state.splitter.split_documents(pdf)
    bar.progress(20)
    st.session_state.embedding= HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    bar.progress(60)
    st.session_state.vector_db = FAISS.from_documents(docs,st.session_state.embedding)
    bar.progress(100)
        

    ##loading model
    
    st.session_state.llm = ChatGroq(model=st.session_state.box,api_key=groq_api)
    
    ##Creating Chain
    st.session_state.document_chain = create_stuff_documents_chain(st.session_state.llm,prompt)
    ## Adding pdf to chain
    st.session_state.retirever =  st.session_state.vector_db.as_retriever()
    st.session_state.chain = create_retrieval_chain(st.session_state.retirever,st.session_state.document_chain)
        
        



user_input = st.text_input('Enter your Query')

if user_input:
    st.write(st.session_state.chain.invoke({'input':user_input})["answer"])


    


