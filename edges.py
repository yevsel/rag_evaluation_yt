from state import AgenticGraphState
from langgraph.graph import END
def route_after_classify_node(state:AgenticGraphState):
    if state["intent"] == "out_of_scope":
        return "answer_direct_node"
    return "research_node"
    
    
def route_after_research_node(state:AgenticGraphState):
    last_message = state["messages"][-1]
    if last_message.tool_calls and not state["search_done"]:
        return "tools_node"
    return "grade_node"
    
    
MAX_CORRECTIVE_ATTEMPTS=2
def route_after_grade_node(state:AgenticGraphState):
    covered = state["coverage"].get("covered")
    if covered:
        return "write_answer_node"
    if state.get("attempts",0)>=MAX_CORRECTIVE_ATTEMPTS:
        return "write_answer_node"
    return "refine_node"
    
    
MAX_REPAIRS=2
def route_after_check_answer_node(state:AgenticGraphState):
    if state.get("verified"):
        return "end"
    if state.get("repairs",0)<MAX_REPAIRS:
        return "fix_answer_node"
    return "give_up_node"