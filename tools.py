import os
import re
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader

load_dotenv()

llm = None
model = None  # Lazy load on first use
llm_disabled = False

COMMON_SKILLS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "sql",
    "html",
    "css",
    "react",
    "node",
    "django",
    "flask",
    "fastapi",
    "streamlit",
    "langchain",
    "langgraph",
    "machine learning",
    "deep learning",
    "nlp",
    "data analysis",
    "pandas",
    "numpy",
    "tensorflow",
    "pytorch",
    "git",
    "docker",
]

def get_llm():
    global llm
    global llm_disabled
    if llm_disabled:
        raise RuntimeError("LLM temporarily disabled due to previous API failure.")
    if llm is None:
        if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            raise ValueError(
                "Gemini API key missing. Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your .env file."
            )
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    return llm

def get_model():
    global model
    if model is None:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer: {e}")
            print("Using mock embeddings instead.")
            model = None
    return model


def extract_skills_fallback(text: str) -> list[str]:
    lowered = text.lower()
    skills = [skill for skill in COMMON_SKILLS if skill in lowered]
    if skills:
        return skills[:10]

    words = re.findall(r"\b[A-Z][A-Za-z0-9#+.-]{1,}\b", text)
    return list(dict.fromkeys(words))[:10]

# Tool 1: Parse Resume PDF/Text
def parse_resume(state: dict) -> dict:
    # Prefer uploaded PDF text, otherwise use the provided resume text.
    resume_file = state.get('resume_file')
    if resume_file:
        reader = PdfReader(resume_file)
        text = ' '.join(page.extract_text() or '' for page in reader.pages)
    else:
        text = state.get('resume_text', '')

    prompt = f"Extract top 10 skills from this resume: {text[:2000]}"
    try:
        skills = get_llm().invoke(prompt).content.split(', ')
        skills = [skill.strip() for skill in skills if skill.strip()]
    except Exception as exc:
        global llm_disabled
        llm_disabled = True
        print(f"Warning: falling back to local skill extraction: {exc}")
        skills = extract_skills_fallback(text)

    return {"resume_text": text, "skills_extracted": skills[:10]}

# Tool 2: Scrape Jobs
def scrape_jobs(state: dict) -> dict:
    query = state.get("job_query") or "fresher AI engineer jobs Chennai Python OR LangGraph"
    try:
        search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_KEY"))
        jobs = search.run(query)
        jobs_list = eval(jobs)[:5] if isinstance(jobs, str) else jobs[:5]
    except Exception as exc:
        print(f"Warning: falling back to sample jobs: {exc}")
        jobs_list = [
            {"title": "Python Developer Intern", "company": "SampleCo", "snippet": "Python, SQL, APIs, Git", "link": ""},
            {"title": "Junior AI Engineer", "company": "SampleAI", "snippet": "LangChain, Python, ML, FastAPI", "link": ""},
            {"title": "Data Analyst Fresher", "company": "SampleData", "snippet": "Python, Pandas, NumPy, SQL", "link": ""},
        ]
    return {"jobs": jobs_list}

# Tool 3: Match Skills (FAISS Vector Search)
def match_jobs(state: dict) -> dict:
    m = get_model()
    if m is None:
        fallback_matches = []
        for job in state.get('jobs', [])[:3]:
            fallback_matches.append({"job": job, "score": "0.70"})
        return {"matches": fallback_matches}  # Return top matches if model fails
    
    skills_vec = m.encode(' '.join(state['skills_extracted']))
    job_descs = [job.get('snippet', '') for job in state['jobs']]
    job_vecs = m.encode(job_descs)
    
    index = faiss.IndexFlatL2(384)  # MiniLM dim
    index.add(np.array([skills_vec]))
    
    if len(job_vecs) == 0:
        return {"matches": []}
        
    distances, indices = index.search(np.array(job_vecs), 1)
    
    matches = []
    max_dist = np.max(distances)
    if max_dist == 0:
        max_dist = 1e-9
        
    for i, job in enumerate(state['jobs']):
        score = 1 - (distances[i][0] / max_dist)  # 0-1 score
        if score > 0.5:
            matches.append({"job": job, "score": f"{score:.2f}"})
    
    matches.sort(key=lambda x: float(x["score"]), reverse=True)
    return {"matches": matches}

# Tool 4: Generate Tailored Resume
def generate_resume(state: dict) -> dict:
    matches = state.get('matches', [])
    if not matches:
        return {
            "tailored_resume": (
                "Tailored Resume Draft\n\n"
                f"Skills: {', '.join(state.get('skills_extracted', []))}\n"
                f"Resume Summary: {state.get('resume_text', '')[:1000]}"
            )
        }

    first_match = matches[0]
    top_match = first_match.get('job', first_match) if isinstance(first_match, dict) else {}
    prompt = f"""
    Tailor this resume for job: {top_match['title']} at {top_match.get('company', '')}
    JD: {top_match.get('snippet', '')}
    Skills: {state.get('skills_extracted', [])}
    Original resume: {state.get('resume_text', '')[:3000]}
    Output only the new resume text.
    """
    try:
        new_resume = get_llm().invoke(prompt).content
    except Exception as exc:
        global llm_disabled
        llm_disabled = True
        print(f"Warning: falling back to template resume generation: {exc}")
        new_resume = (
            "Tailored Resume Draft\n\n"
            f"Target Role: {top_match.get('title', '')} at {top_match.get('company', '')}\n"
            f"Matched Skills: {', '.join(state.get('skills_extracted', []))}\n"
            f"Job Keywords: {top_match.get('snippet', '')}\n"
            f"Original Resume Summary: {state.get('resume_text', '')[:1000]}"
        )
    return {"tailored_resume": new_resume}