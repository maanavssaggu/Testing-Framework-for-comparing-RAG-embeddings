from pydantic import BaseModel
from src.model import Model
from src.vectorstore import VectorStoreManager
from langchain_core.documents import Document
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from langchain_cohere import CohereEmbeddings
from src.schemas.question import Question
from langchain_openai import OpenAIEmbeddings
import os

class Pipeline:
    def __init__(self,  
                 model: Model, 
                 vector_store_manager: VectorStoreManager,
                 input_query: str = None,
                 embedding_function: Optional[Embeddings] = OpenAIEmbeddings(model="text-embedding-3-large")) -> None:
        """
        Initialize the Pipeline with components for embedding, vector store, and model querying.
        """
        self.embedding_function = embedding_function
        self.embedding = embedding_function.model
        self.model = model
        self.vector_store_manager = vector_store_manager
        self.vector_store = vector_store_manager.vector_store
        self.input_query = input_query

    def process_data(self):
        """
        Ingest new data into the vector store.
        """
        self.vector_store_manager.ingest_data()
    
    def retrieve(self, input_query: str = None) -> List[str]:
        """
        Retrieve documents and their IDs based on the input query.
        """
        print(f"retrieving documents with embedding: {self.embedding}")
        
        # Perform vector similarity search
        results = self.vector_store.similarity_search_with_score(input_query, 
                                                                 k=5,
                                                                 filter={"embedding": self.embedding})
    
        # Extract document IDs, or None if ID does not exist
        sources = [doc.metadata.get("id", None) for doc, _score in results]    
        return results, sources 

    def generate(self, input_query: str, retrieved_documents: Optional[List[Document]] = None) -> str:
        """
        Generate a response based on the input query and optionally retrieved documents.
        """
        self.process_data()  # Update knowledge base
        print(f"input query has type: {type(input_query)}")

        # Retrieve documents if not provided
        if retrieved_documents is None:
            retrieved_documents, sources = self.retrieve(input_query)

        # Generate response using the model and retrieved context
        print("starting to generate a response")
        context_text = "\n\n---\n\n".join([doc[0].page_content for doc in retrieved_documents])
        
        response = self.model.query(input_query, context_txt=context_text)
        return response, sources
