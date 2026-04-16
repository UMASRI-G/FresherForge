from graph import app
import streamlit as st  # For UI
from streamlit.runtime.scriptrunner import get_script_run_ctx


def is_streamlit_runtime() -> bool:
    return get_script_run_ctx() is not None

if __name__ == "__main__" and not is_streamlit_runtime():
    # CLI Test
    input_state = {
        "resume_text": "Your resume text here or upload PDF path.",
        "resume_file": "path/to/your_resume.pdf"  # Optional
    }
    result = app.invoke(input_state)
    print("Matches:", result['matches'])
    print("Tailored Resume:", result['tailored_resume'][:500])

if is_streamlit_runtime():
    # Streamlit UI (streamlit run main.py)
    st.title("🛠️ FresherForge MVP")
    resume_file = st.file_uploader("Upload Resume PDF")
    query = st.text_input("Job Keywords", "AI Engineer fresher Chennai")

    if st.button("Run Agent"):
        if resume_file is None:
            st.error("Upload a resume PDF before running the agent.")
            st.stop()

        state = {"resume_file": resume_file, "job_query": query}
        try:
            with st.spinner("Running the agent..."):
                result = app.invoke(state)
        except Exception as exc:
            st.error(f"Agent failed: {exc}")
            st.stop()

        st.success("Top Matches:")
        matches = result.get('matches', [])
        if not matches:
            st.info("No strong matches were found.")
        for m in matches:
            job = m.get('job', {})
            st.write(f"**{m.get('score', '0.00')}** - {job.get('title', 'Untitled job')}")

        st.markdown("### Tailored Resume")
        st.text_area("", result.get('tailored_resume', ''), height=300)
