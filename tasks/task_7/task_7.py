import streamlit as st
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
import os
import sys
sys.path.append(os.path.abspath('../../'))

class QuizGenerator:
    """
    Initializes the QuizGenerator with a required topic, the number of questions for the quiz,
    and an optional vectorstore for querying related information.

    :param topic: A string representing the required topic of the quiz.
    :param num_questions: An integer representing the number of questions to generate for the quiz, up to a maximum of 10.
    :param vectorstore: An optional vectorstore instance (e.g., ChromaDB) to be used for querying information related to the quiz topic.
    """
    def __init__(self, topic=None, num_questions=1, vectorstore=None):
        if not topic:
            self.topic = "General Knowledge" #default
        else:
            self.topic = topic

        if num_questions > 10:
            raise ValueError("Number of questions cannot exceed 10.")
        self.num_questions = num_questions

        self.vectorstore = vectorstore
        self.llm = None
        self.system_template = """
            You are a subject matter expert on the topic: {topic}
            
            Follow the instructions to create a quiz question:
            1. Generate a question based on the topic provided and context as key "question"
            2. Provide 4 multiple choice answers to the question as a list of key-value pairs "choices"
            3. Provide the correct answer for the question from the list of answers as key "answer"
            4. Provide an explanation as to why the answer is correct as key "explanation"
            
            You must respond as a JSON object with the following structure:
            {{
                "question": "<question>",
                "choices": [
                    {{"key": "A", "value": "<choice>"}},
                    {{"key": "B", "value": "<choice>"}},
                    {{"key": "C", "value": "<choice>"}},
                    {{"key": "D", "value": "<choice>"}}
                ],
                "answer": "<answer key from choices list>",
                "explanation": "<explanation as to why the answer is correct>"
            }}
            
            Context: {context}
            """
        
    """
    Initialize the LLM for quiz question generation.
    - Use the VertexAI class to create an instance of the LLM with the specified configurations.
    - Assign the created LLM instance to the 'self.llm' attribute for later use in question generation.
    """
    def init_llm(self):
        self.llm = VertexAI(
            model_name= "gemini-pro",
            temperature = 0.2,
            max_output_tokens = 1024
        )

    """
    Generate a quiz question using the topic provided and context from the vectorstore.  
    This method leverages the vectorstore to retrieve relevant context for the quiz topic, then utilizes the LLM to generate a structured quiz question in JSON format. The process involves retrieving documents, creating a prompt, and invoking the LLM to generate a question.
    - Utilize 'RunnableParallel' and 'RunnablePassthrough' to create a chain that integrates document retrieval and topic processing.
    - Format the system template with the topic and retrieved context to create a comprehensive prompt for the LLM.
    - Use the LLM to generate a quiz question based on the prompt and return the structured response.
    """   
    def generate_question_with_vectorstore(self):
        if not self.llm:
            self.init_llm()
        if not self.vectorstore:
            raise ValueError("Vectorstore is not initialized.")
        
        from langchain_core.runnables import RunnablePassthrough, RunnableParallel
        #retrieving relevant data from the vectorstore based on the topic
        retriever = self.vectorstore.db.as_retriever()

        template = PromptTemplate.from_template(self.system_template)
        
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "topic": RunnablePassthrough()}
        )
        
        chain = setup_and_retrieval | template | self.llm

        # Invoke the chain with the topic as input
        response = chain.invoke(self.topic)
        return response
    
# Test the Object
if __name__ == "__main__":
    
    from tasks.task_3.task_3 import DocumentProcessor
    from tasks.task_4.task_4 import EmbeddingClient
    from tasks.task_5.task_5 import ChromaCollectionCreator
    
    
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "quizify-432223",
        "location": "us-central1"
    }
    
    screen = st.empty()
    with screen.container():
        st.header("Quiz Builder")
        processor = DocumentProcessor()
        processor.ingest_documents()
    
        embed_client = EmbeddingClient(**embed_config)
    
        chroma_creator = ChromaCollectionCreator(processor, embed_client)

        question = None
    
        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            
            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
            
            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()
                
                st.write(topic_input)
                
                # Test the Quiz Generator!!
                generator = QuizGenerator(topic_input, questions, chroma_creator)
                question = generator.generate_question_with_vectorstore()

    if question:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Question: ")
            st.write(question)