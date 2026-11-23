from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgenticGraphState(TypedDict):
    question: str
    
    intent: str
    
    blog_post: str
    
    messages: Annotated[list[AnyMessage], add_messages]
    
    chunks: list
    
    searches: int
    
    search_done: bool
    
    coverage: dict # covered, missing, next_query
    
    attempts: int
    
    answer: str
    
    critique: str
    
    repairs: int
    
    verified: bool
    
    refused: bool