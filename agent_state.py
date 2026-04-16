from typing import TypedDict, Dict, Any, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    resume_text: str
    skills_extracted: List[str]
    jobs: List[Dict[str, Any]]
    matches: List[Dict[str, Any]]
    tailored_resume: str
    messages: List[BaseMessage]