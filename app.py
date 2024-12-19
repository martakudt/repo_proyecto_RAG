import streamlit as st
import pandas as pd
import numpy as np
import os
import cohere
import warnings

warnings.filterwarnings("ignore")

from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="lector_pdf",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.header(f"LECTOR DE PDF 📚")
st.text("Este proyecto consiste en una aplicación interactiva desarrollada con Streamlit que permite cargar y leer archivos PDF, y realizar preguntas sobre su contenido. Utilizando un modelo de procesamiento de lenguaje natural (NLP), la aplicación extrae texto del documento y responde de manera precisa y contextualizada a las consultas del usuario, facilitando la búsqueda de información específica dentro del archivo. La interfaz intuitiva permite cargar fácilmente los archivos, escribir preguntas y ver las respuestas de forma directa, lo que hace de esta herramienta una solución eficiente para consultar manuales, libros, informes o cualquier documento extenso.")
st.image("images.png")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

file = st.file_uploader("Upload a PDF file", accept_multiple_files=False, type="pdf")
from langchain.document_loaders import PyPDFDirectoryLoader
source_data_folder = "./MisDatos"
if file:
    with open(source_data_folder + "/pdf.pdf", "wb") as f:
        f.write(file.getvalue())

    
    # Leyendo los PDFs del directorio configurado
    loader = PyPDFDirectoryLoader(source_data_folder)
    data_on_pdf = loader.load()

    if data_on_pdf:
        st.success("PDF cargado con éxito.")
        text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=2000,
        chunk_overlap=200
        )
        splits = text_splitter.split_documents(data_on_pdf)
        print(splits[0])
        
        # embeddings_model = CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"], user_agent="antonio")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["GOOGLE_API_KEY"])
        

        path_db = "./VectorDB"  # Ruta a la base de datos del vector store

        # Crear el vector store a partir de tus documentos 'splits'
        from langchain.vectorstores import FAISS
        # Crear o cargar el vector store
        if os.path.exists(path_db+"/index.faiss"):
            # Cargar el índice FAISS existente
            vectorstore = FAISS.load_local(path_db, embeddings_model, allow_dangerous_deserialization=True)
        else:
            # Crear el vector store a partir de tus documentos 'splits'
            vectorstore = FAISS.from_documents(
                documents=splits, 
                embedding=embeddings_model,
                # allow_dangerous_deserialization=True
            )
            # Guardar el índice FAISS
            vectorstore.save_local(path_db)
        
        retriever = vectorstore.as_retriever()
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.environ["GOOGLE_API_KEY"])
        prompt = hub.pull("rlm/rag-prompt")
       
        def format_docs(docs):
            # Funcion auxiliar para enviar el contexto al modelo como parte del prompt
            print(docs[0])
            return "\n\n".join(doc.page_content for doc in docs)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
            )

user_question = st.text_input("¿Qué quieres saber sobre el PDF?", "")

if st.button("Hacer pregunta") and data_on_pdf:
            result = rag_chain.invoke(user_question)
            st.write("Respuesta:", result)