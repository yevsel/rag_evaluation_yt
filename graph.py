from langgraph.graph import END,START, StateGraph
from state import AgenticGraphState
from nodes import reset_node, classify_node,research_node,tools_node,answer_direct_node,grade_node,refine_node,write_answer_node,check_answer_node,fix_answer_node,give_up_node
from edges import route_after_classify_node,route_after_research_node,route_after_grade_node,route_after_check_answer_node

builder = StateGraph(AgenticGraphState)

builder.add_node("reset_node",reset_node)
builder.add_node("classify_node",classify_node)
builder.add_node("research_node", research_node)
builder.add_node("tools_node",tools_node)
builder.add_node("answer_direct_node",answer_direct_node)
builder.add_node("grade_node",grade_node)
builder.add_node("refine_node",refine_node)
builder.add_node("write_answer_node",write_answer_node)
builder.add_node("check_answer_node",check_answer_node)
builder.add_node("fix_answer_node",fix_answer_node)
builder.add_node("give_up_node",give_up_node)


builder.add_edge(START,"reset_node")
builder.add_edge("reset_node","classify_node")
builder.add_conditional_edges("classify_node",
    route_after_classify_node,
    {"answer_direct_node":"answer_direct_node", "research_node":"research_node"}
)

builder.add_conditional_edges("research_node",
    route_after_research_node,
    {"tools_node":"tools_node", "grade_node":"grade_node"}
)
builder.add_edge("tools_node","research_node")

builder.add_conditional_edges("grade_node",
    route_after_grade_node,
    {"refine_node":"refine_node", "write_answer_node":"write_answer_node"}
)

builder.add_edge("refine_node","research_node")

builder.add_edge("write_answer_node","check_answer_node")

builder.add_conditional_edges("check_answer_node",
    route_after_check_answer_node,
    {"end":END,"fix_answer_node":"fix_answer_node","give_up_node":"give_up_node"}
)
builder.add_edge("fix_answer_node","write_answer_node")

graph = builder.compile()
