import sys
import os
import streamlit as st
import re

sys.path.append(os.path.abspath('../../'))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient


# Import Task libraries
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

os.environ['GRPC_DNS_RESOLVER'] = 'native'

# Remove non-printable characters
def clean_text(text):
    return re.sub(r'[^\x20-\x7E]', '', text)

#creating a ChromeCollection, but utilizing processor and embeddings configuration created prviously
class ChromaCollectionCreator:
    """
    Initializes the ChromaCollectionCreator with a DocumentProcessor instance and embeddings configuration.
    param processor: An instance of DocumentProcessor that has processed documents.
    param embeddings_config: An embedding client for embedding documents.
    """
    def __init__(self, processor, embed_model):
        self.processor = processor      
        self.embed_model = embed_model  
        self.db = None                  
    
    #Create a Chroma collection from the documents processed by the DocumentProcessor instance.
    def create_chroma_collection(self):    
        # Check if the document exists
        if len(self.processor.pages) == 0:
            st.error("No documents found!", icon="🚨")
            return
        
        #if file exists, split it into chunks which can be put into ChromaCollection
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
        )

        #Put all of the chunks into texts
        texts = []
        for doc in self.processor.pages:
            chunk = text_splitter.split_text(doc.page_content)
            texts.extend(chunk)
        texts = [clean_text(doc) for doc in texts]
        if texts is not None:
            st.success(f"Successfully split pages to {len(texts)} documents!", icon="✅")

        #cast to Document type
        documents = []
        for text in texts:
            documents.append(Document(page_content=text))

        #Create a Chroma collection from the documents
        try:
            self.db = Chroma.from_documents(documents=documents, embedding=self.embed_model)
        except Exception as e:
            for i, doc in enumerate(texts):
                print(f"Document {i}: {doc}")
                #something's wrong if the document is too short
                if len(doc) <= 10:
                    print(f"Bad document found at index {i}!")
                    raise e
            print("Error during Chroma.from_documents:")
            print(f"Number of documents: {len(texts)}")
            raise e
        
        if self.db:
            st.success("Successfully created Chroma Collection!", icon="✅")
        else:
            st.error("Failed to create Chroma Collection!", icon="🚨")

    """
    Queries the created Chroma collection for documents similar to the query.
    param query: The query string to search for in the Chroma collection.
    Returns the first matching document from the collection with similarity score.
    """
    def query_chroma_collection(self, query) -> Document:
        if self.db:
            docs = self.db.similarity_search_with_relevance_scores(query)
            if docs:
                return docs[0]
            else:
                st.error("No matching documents found!", icon="🚨")
        else:
            st.error("Chroma Collection has not been created!", icon="🚨")

#Testing
if __name__ == "__main__":
    processor = DocumentProcessor() # Initialize from Task 3
    processor.ingest_documents()
    
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "quizify-432223",
        "location": "us-central1"
    }
    
    embed_client = EmbeddingClient(**embed_config)
    chroma_creator = ChromaCollectionCreator(processor, embed_client)
    
    with st.form("Load Data to Chroma"):
        st.write("Select PDFs for Ingestion, then click Submit")
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            chroma_creator.create_chroma_collection()