import sys
import os
import streamlit as st
sys.path.append(os.path.abspath('../../'))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator

"""
Quiz Builder with Streamlit and LangChain
"""
if __name__ == "__main__":
    st.header("Quizzify")

    # Configuration for EmbeddingClient
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "quizify-432223",
        "location": "us-central1"
    }
    
    screen = st.empty() # Screen 1, ingest documents
    with screen.container():
        st.header("Quizzify")
        #Initalize DocumentProcessor and Ingest Documents
        processor = DocumentProcessor()
        processor.ingest_documents()
        #Initalize the EmbeddingClient
        embed_client = EmbeddingClient(**embed_config)
        #Initialize the ChromaCollectionCreator
        chroma_creator = ChromaCollectionCreator(processor, embed_client)

        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            
            topic = st.text_input(label="Quiz Topic", placeholder="What is the topic?")
            num_question = st.slider(label="How many questions?", min_value=1, max_value=10, value=5) #default to 5
            
            document = None
            
            submitted = st.form_submit_button("Generate a Quiz!")
            if submitted:
                chroma_creator.create_chroma_collection() 
                document = chroma_creator.query_chroma_collection(topic)
    #when finding the document with closest match, display a new screen of its content           
    if document:
        screen.empty() # Screen 2
        with st.container():
            st.header("Query Chroma for Topic, top Document: ")
            st.write(document)