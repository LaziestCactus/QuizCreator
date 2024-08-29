from langchain_google_vertexai import VertexAIEmbeddings
import os
os.environ['GRPC_DNS_RESOLVER'] = 'native'

class EmbeddingClient:
    """
    Parameters:
    - model_name: A string representing the name of the model to use for embeddings.
    - project: The Google Cloud project ID where the embedding model is hosted.
    - location: The location of the Google Cloud project, such as 'us-central1'.
    """ 
    def __init__(self, model_name, project, location):
        # Initialize the VertexAIEmbeddings client
        try:
            self.client = VertexAIEmbeddings(
                model_name=model_name,
                project=project,
                location=location
            )
        #Notify if doesn't work
        except Exception as e:
            print(f"Failed to initialize client: {e}")
            self.client = None

    """
    param query: The text query to embed.
    return: The embeddings for the query or None if the operation fails.
    """
    def embed_query(self, query):
        vectors = self.client.embed_query(query)
        return vectors
    
    """
    Retrieve embeddings for multiple documents.
    param documents: A list of text documents to embed.
    return: A list of embeddings for the given documents.
    """
    def embed_documents(self, documents):
        try:
            return self.client.embed_documents(documents)
        except AttributeError:
            print("Method embed_documents not defined for the client.")
            return None
        
key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if key_path:
    print(f"Current service account key file: {key_path}")
else:
    print("GOOGLE_APPLICATION_CREDENTIALS is not set.")

if __name__ == "__main__":
    model_name = "textembedding-gecko@003"
    project = "quizify-432223"
    location = "us-central1"

    embedding_client = EmbeddingClient(model_name, project, location)
    #testing vertexAI embedding
    vectors = embedding_client.embed_query("Hello World!")
    if vectors:
        print(vectors)
        print("Successfully used the embedding client!")