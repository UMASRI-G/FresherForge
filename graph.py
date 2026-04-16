from langgraph.graph import StateGraph, END
from agent_state import AgentState
from tools import parse_resume, scrape_jobs, match_jobs, generate_resume

def should_continue(state: AgentState):
    if not state.get('skills_extracted'):
        return "scrape"
    if not state.get('jobs'):
        return "scrape"
    if not state.get('matches'):
        return "match"
    return "generate"

workflow = StateGraph(AgentState)
workflow.add_node("parse", parse_resume)
workflow.add_node("scrape", scrape_jobs)
workflow.add_node("match", match_jobs)
workflow.add_node("generate", generate_resume)

workflow.set_entry_point("parse")
workflow.add_conditional_edges("parse", should_continue, {
    "scrape": "scrape",
    "match": "match",
    "generate": "generate",
    END: END
})
workflow.add_edge("scrape", "match")
workflow.add_edge("match", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()