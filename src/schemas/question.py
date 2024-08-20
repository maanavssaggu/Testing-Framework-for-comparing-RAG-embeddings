from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class Question(BaseModel):

    question: str = Field(description="The setup of the joke")
    answer: str = Field(description="The punchline to the joke")
    
