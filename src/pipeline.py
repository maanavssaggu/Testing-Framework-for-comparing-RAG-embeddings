from pydantic import BaseModel
from src.model import Model
from src.vectorstore import VectorStoreManager
from langchain_core.documents import Document
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from langchain_cohere import CohereEmbeddings
from langchain_openai import OpenAIEmbeddings
import os

class Pipeline():
    def __init__(self,  
                 model: Model, 
                 vector_store_manager: VectorStoreManager,
                 input_query: str = None,
                 embedding_function: Optional[Embeddings] = OpenAIEmbeddings(model="text-embedding-3-large")) -> None:
        self.embedding_function = embedding_function
        self.embedding = embedding_function.model
        self.model = model
        self.vector_store_manager = vector_store_manager
        self.vector_store = vector_store_manager.vector_store
        self.input_query = input_query

    # preprocess the data, see if there anything new knowledge
    def process_data(self):
        self.vector_store_manager.ingest_data()
    
    # retrieves do ids 
    def retrieve(self, input_query: str = None)->List[str]:
        """
            input: input query 
            output: (retrieved documents, corresponding document ids)
        """
        print(f"retrieving documents with embedding: {self.embedding}")
        # perform vector similarity search  

        results = self.vector_store.similarity_search_with_score(input_query, 
                                                                 k=5,
                                                                 filter={"embedding":self.embedding})
    
        # stores ids to retrieve documents None if id does not exist 
        sources = [doc.metadata.get("id", None) for doc, _score in results]    
        return results, sources 

    def generate(self, input_query:str, retrieved_documents: Optional[List[Document]]=None) -> str:
        # update knowledge base 
        self.process_data()
        print(f"input query has type: {type(input_query)}")

        # retrieve
        if retrieved_documents is None:
            retrieved_documents, sources = self.retrieve(input_query)

        # generate 
        print("starting to generate a response")
        context_text = "\n\n---\n\n".join([doc[0].page_content for doc in retrieved_documents])
        
        response = self.model.query(input_query, context_txt=context_text)
        return response, sources

