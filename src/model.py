import numpy as np
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Dict, List
from langchain_core.pydantic_v1 import BaseModel, Field
from src.schemas.test_case import TestCase


class Model():
    def __init__(self,)->None:
        self.RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
            [
                (
                    "system", 
                    "You are to answer questions taking into consideration the following context: {context}"
                ),
                (
                    "human", "Answer the question with the above context, lets think about this step by step: {query}"
                )
            ]
        )

        self.TEST_CASE_TEMPLATE = ChatPromptTemplate.from_messages(
            [
                (
                    "system", 
                    " create a question to context given. The question needs to have a boolean answer (True or False). The context: {context}"
                )
            ]
        )

        try: 
            load_dotenv()
            OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
            COHERE_API_KEY = os.getenv('COHERE_API_KEY')

            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                max_tokens=500,
                timeout=1000,
                max_retries=2,
            )

            self.chain = self.RAG_PROMPT_TEMPLATE | self.llm
            print("succsefully initilsied the model")
        except Exception as e:
            print(f"error loading Model {e}")
        
    def query(self, query:str, context_txt:str) -> str:
        try:
            print("invoking the model")
            
            return self.chain.invoke(
                {
                    "context": context_txt, 
                    "query": query,
                }
            )
        except Exception as e:
            print(f"An error occurred when invoking the model {e}")
        
    def generate_qa_pair(self, document_content:str, doc_id:str)->TestCase:
        """
        generates a dictionary which creates question and answers based on documnets in the knowledge base
        """
        ret = {}
        test_case_chain = self.TEST_CASE_TEMPLATE | self.llm.with_structured_output(Question)
        try: 

            ret["QA"] = test_case_chain.invoke(
                {
                    "context": document_content
                }
            )
            ret["doc_id"] = doc_id
            
            print('succesfully generated QA pair, ')
            return TestCase(ret)
        except Exception as e:
            print("exception occured: {e}")
        

class Question(BaseModel):

    question: str = Field(description="The setup of the joke")
    answer: str = Field(description="The punchline to the joke")
    



