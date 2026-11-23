from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from typing_extensions import Literal
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel, Field
from llm_model import llm
from prompts import (
    CLASSIFY_SYSTEM,
    ANSWER_DIRECT_SYSTEM,
    CHECK_ANSWER_SYSTEM,
    GIVE_UP_MESSAGE,
    GRADE_SYSTEM,
    RESEARCH_SYSTEM,
    WRITE_ANSWER_SYSTEM
)
from state import AgenticGraphState
from tools import retrieve_relevant_document
import json
#====
# Reset Node
#====

def reset_node(state):
    return{
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        "chunks": [],
        "searches": 0,
        "attempts": 0,
        "repairs": 0,
        "search_done": False,
        "answer": "",
        "critique": "",
        "coverage":{},
        "verified": False,
        "refused": False
    }
    

#====
# Classify node
#====

class ClassifyNodeSchema(BaseModel):
    intent: Literal["in_scope", "out_of_scope"] = Field(
        description="Return in_scope if its about our indexed posts, or else return out_of_scope"
    )
    blog_post: str = Field(
        default="",
        description="harness-engineering or diffusion-models if one blog post is targeted, else empty"
    )


def classify_node(state:AgenticGraphState):
    classify_chain =ChatPromptTemplate.from_messages([
        ("system",CLASSIFY_SYSTEM),
        ("human","{question}")
    ]) | llm.with_structured_output(ClassifyNodeSchema)
    
    llm_response = classify_chain.invoke({"question":state["question"]})
    
    return {
        "intent": llm_response.intent,
        "blog_post": llm_response.blog_post
    }
    
    
#====
# Answer Direct when out_of_scope
#====
def answer_direct_node(state: AgenticGraphState):
    llm_response = llm.invoke(
        [SystemMessage(content=ANSWER_DIRECT_SYSTEM), 
        HumanMessage(content=state["question"])]
    )
    return{
        "answer": llm_response.content
    }   
    
#=====
# Research node after user question is in_scope
#=====
llm_with_tools = llm.bind_tools([retrieve_relevant_document])

def research_node(state:AgenticGraphState):
    messages = [
        SystemMessage(
            content=RESEARCH_SYSTEM.format(
                question=state["question"],
                blog_post=state.get("blog_post", "any")
            )
        )
    ] + state.get("messages")
    
    llm_response = llm_with_tools.invoke(messages)
    
    return{
        "messages": [llm_response]
    }
    

#=====
# Tools Node
#=====
MAX_SEARCHES=4
def tools_node(state:AgenticGraphState):
    last_message = state["messages"][-1]

    if state.get("searches",0)>= MAX_SEARCHES:
        messages=[
            ToolMessage(
                content="No more searches left. Answer with what you already have",
                tool_call_id=call["id"],
                name=call["name"]
            )   
            for call in last_message.tool_calls
        ]
        return {"messages": messages,"search_done": True}
    

    messages, chunks = [],[]
    
    for call in last_message.tool_calls:
        relevant_docs = retrieve_relevant_document.invoke(call["args"])
        chunks.extend(relevant_docs)
        messages.append(
            ToolMessage(
                content=json.dumps({"chunks": relevant_docs}),
                tool_call_id= call["id"],
                name=call["name"]
            )
        )
        
    return {
        "messages": messages,
        "chunks": list(state.get("chunks") or []) + chunks,
        "searches": state.get("searches", 0)+1
    }
    


#====
# Grade node - Critique yourself
#====

def format_chunks(chunks):
    blocks=[]
    for item in (chunks or []):
        blocks.append(
            f"[{item.get("blog_post","")}]"
            f"{item.get("text","")}"
        )

    return "\n\n".join(blocks)

class GradeNodeCoverageSchema(BaseModel):
    covered: bool=Field(description="Return True if every part of the question is supported")
    missing: str = Field(default="", description="What is still missing if anything")
    next_query: str = Field(default="", description="One better search query of not covered")

def grade_node(state:AgenticGraphState):
    grade_chain = ChatPromptTemplate.from_messages([
        ("system",GRADE_SYSTEM),
        ("human", "Question: {question} \n Chunks:\n{chunks}")
    ]) | llm.with_structured_output(GradeNodeCoverageSchema)
    
    llm_response = grade_chain.invoke(
        {"question":state["question"], "chunks":format_chunks(state["chunks"])}
    )
    return {
        "coverage": llm_response.model_dump()
    }
    
    
    
#====
# Refine node
#====

def refine_node(state:AgenticGraphState):
    coverage = state["coverage"]

    instructions = HumanMessage(
        content=(
            f"That chunks were not enough. Still missing:{coverage.get("missing")}"
            f"Search again, starting with: {coverage.get("next_query")}"
        )   
    )
    
    return{
        "messages":[instructions],
        "attempts": state.get("attempts",0)+1,
        "searches":0,
        "search_done":False
    }
    

#=====
# Write answer node = The corrective section was satisfied
#=====
def write_answer_node(state:AgenticGraphState):
    write_answer_chain = ChatPromptTemplate.from_messages([
        ("system", WRITE_ANSWER_SYSTEM),
        ("human","Question: {question} \n Chunks: {chunks} {critique}")
    ]) | llm
    
    critique = state.get("critique","")
    critique_block = ""
    if critique:
        critique_block=(f"\n\n A previous attempt was rejected by the verifier: {critique}"
        "Fix that. Use only what the chunk says"
        )
    else: 
        critique_block=""
    
    llm_response = write_answer_chain.invoke({
        "question":state["question"],
        "chunks": format_chunks(state.get("chunks")),
        "critique": critique_block
    })
    
    return{
        "answer": llm_response.content
    }
    

#====
# Check answer node
#====
class AnswerVerificationSchema(BaseModel):
    grounded: bool = Field(description="Return False if any claim is unsupported or credited to the wrong post")
    abstained: bool = Field(default=False,description="True if the answer says the chunks lack the answer")
    unsupported_claims: list[str]= Field(default_factory=list)
    
def check_answer_node(state:AgenticGraphState):
    check_answer_chain = ChatPromptTemplate.from_messages([
        ("system",CHECK_ANSWER_SYSTEM),
        ("human", "Question: {question} \n Answer:\n{answer} \nChunks:\n{chunks}")
    ]) | llm.with_structured_output(AnswerVerificationSchema)
    
    check = check_answer_chain.invoke({
        "question":state["question"],
        "answer": state.get("answer",""),
        "chunks": format_chunks(state.get("chunks"))
    })
    
    if check.grounded:
        critique=""
    else:
        critique="; ".join(check.unsupported_claims)
    return{
        "verified": check.grounded,
        "refused": check.abstained,
        "critique": critique
    }
    

#===
# Fix answer node
#===
def fix_answer_node(state:AgenticGraphState):
    return{
        "repairs": state.get("repairs",0)+1
    }
    
#=====
# Give up node
#=====
def give_up_node(state:AgenticGraphState):
    return{
        "answer":GIVE_UP_MESSAGE,
        "refused": True
    }