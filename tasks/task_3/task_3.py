import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
import uuid

class DocumentProcessor:
    """
    This class encapsulates the functionality for processing uploaded PDF documents using Streamlit
    and Langchain's PyPDFLoader. It provides a method to render a file uploader widget, process the
    uploaded PDF files, extract their pages, and display the total number of pages extracted.
    """
    def __init__(self):
        self.pages = []  # List to keep track of pages from all documents
    
    def ingest_documents(self):
        """
        Renders a file uploader in a Streamlit app, processes uploaded PDF files,
        extracts their pages, and updates the self.pages list with the total number of pages.
        """
        #Create on streamlit, a file uploader widget to allow users to upload PDF files.
        uploaded_files = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=True, help="You can upload multiple")
        
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                # Generate a unique identifier to append to the file's original name
                unique_id = uuid.uuid4().hex #hexadecimal string
                original_name, file_extension = os.path.splitext(uploaded_file.name) #take out pdf
                temp_file_name = f"{original_name}_{unique_id}{file_extension}" #example_123.pdf
                temp_file_path = os.path.join(tempfile.gettempdir(), temp_file_name)

                # Write the uploaded PDF to a temporary file
                with open(temp_file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue()) #writes the binary content into the temp file-path

                # Use PyPDFLoader to load uploaded files and load them into "pages"
                loader = PyPDFLoader(temp_file_path)
                pages = loader.load()
                num_pages = len(pages)
                st.write(f"The PDF contains {num_pages} pages.")
                
                # Append pages to pages list (locally)
                for p in pages: 
                    self.pages.append(p)

                # Clean up by deleting the temporary file.
                os.unlink(temp_file_path)
            
            # Display the total number of pages processed.
            st.write(f"Total pages processed: {len(self.pages)}")
            
#testing
if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.ingest_documents()
