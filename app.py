import os
import io
import json
import time
import tempfile
import traceback
from typing import List, Tuple, Dict, Any
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

# Optional, may be heavy
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    SentenceTransformer = None
    _HAS_ST = False

# Transformers used only when available
try:
    import torch
except Exception:
    torch = None

# NLP helpers
try:
    import docx2txt
except Exception:
    docx2txt = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# Grammar tool optional
try:
    import language_tool_python
    _HAS_LANGTOOL = True
except Exception:
    language_tool_python = None
    _HAS_LANGTOOL = False

# Lightweight fallback if embeddings unavailable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import plotly.graph_objects as go

# -------------------- Configuration --------------------
APP_TITLE = "CT-Learner Pro — Clean Rewrite"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # reasonably small

PAUL_CT_RUBRIC = {
    "Clarity": {"patterns": ["for example", "for instance", "e.g.", "such as"], "feedback": "Provide examples to illustrate your points."},
    "Accuracy": {"patterns": ["according to", "study", "data", "%", "cite"], "feedback": "Cite sources or data to support claims."},
    "Relevance": {"patterns": ["related to", "in relation to", "pertaining to"], "feedback": "Keep focus on the question/problem."},
    "Significance": {"patterns": ["important", "main", "key", "central"], "feedback": "Highlight the main idea/priority."},
    "Logic": {"patterns": ["therefore", "because", "thus", "hence", "however"], "feedback": "Ensure conclusions follow from evidence."},
    "Precision": {"patterns": ["specifically", "exactly", "precisely"], "feedback": "Use specific language and avoid vagueness."},
    "Fairness": {"patterns": ["on the other hand", "although", "however", "despite"], "feedback": "Consider alternative viewpoints."},
    "Depth": {"patterns": ["because", "although", "complex", "in depth"], "feedback": "Examine complexities, not just surface-level points."},
    "Breadth": {"patterns": ["alternatively", "different perspective", "other view"], "feedback": "Consider a range of perspectives/solutions."}
}

# -------------------- Utilities --------------------
@st.cache_resource
def get_embedding_model():
    """Attempt to load SentenceTransformer; if not available, return None (fallback to TF-IDF)."""
    if _HAS_ST and SentenceTransformer is not None:
        try:
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            return model
        except Exception:
            return None
    return None


def read_text_file(uploaded_file) -> str:
    """Read text from uploaded file: txt, docx, pdf. Robust to missing optional libs."""
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    # txt
    if name.endswith(".txt"):
        try:
            return data.decode('utf-8')
        except Exception:
            try:
                return data.decode('latin-1')
            except Exception:
                return ""

    # docx
    if name.endswith('.docx'):
        if docx2txt is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                    tmp.write(data)
                    tmp.flush()
                    txt = docx2txt.process(tmp.name)
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
                return txt or ""
            except Exception:
                return ""
        else:
            return ""  # docx2txt missing

    # pdf
    if name.endswith('.pdf'):
        if pdfplumber is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(data)
                    tmp.flush()
                    path = tmp.name
                text_pages = []
                with pdfplumber.open(path) as pdf:
                    for p in pdf.pages:
                        txt = p.extract_text() or ""
                        text_pages.append(txt)
                try:
                    os.unlink(path)
                except Exception:
                    pass
                return "\n".join(text_pages)
            except Exception:
                return ""
        else:
            return ""

    # fallback: try decode
    try:
        return data.decode('utf-8')
    except Exception:
        try:
            return data.decode('latin-1')
        except Exception:
            return ""


def embed_texts_with_fallback(texts: List[str]):
    """Return embeddings. Prefer SentenceTransformer; fallback to TF-IDF vectors."""
    model = get_embedding_model()
    texts = [t if t is not None else "" for t in texts]

    if model is not None:
        try:
            vectors = model.encode(texts, convert_to_numpy=True)
            return vectors, 'st'
        except Exception:
            pass

    # TF-IDF fallback — returns dense vectors
    tf = TfidfVectorizer(max_features=2048, stop_words='english')
    vecs = tf.fit_transform(texts).toarray()
    return vecs, 'tfidf'


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    if a is None or b is None:
        return 0.0
    try:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
        s = cosine_similarity(a, b)[0, 0]
        return float(s)
    except Exception:
        return 0.0

# -------------------- Grading --------------------

def heuristic_grade(model_ans: str, student_ans: str) -> Dict[str, Any]:
    """Simple grading: similarity (0-100) + optional grammar penalty."""
    texts = [model_ans, student_ans]
    vecs, method = embed_texts_with_fallback(texts)
    sim = 0.0
    try:
        sim = cosine_sim(vecs[0], vecs[1])
    except Exception:
        sim = 0.0

    sim_pct = max(0.0, min(1.0, sim))
    base_score = round(sim_pct * 100, 2)

    # grammar check (optional)
    grammar_info = {"available": False, "issues": 0}
    if _HAS_LANGTOOL and language_tool_python is not None:
        try:
            tool = language_tool_python.LanguageTool('en-US')
            matches = tool.check(student_ans)
            grammar_info = {"available": True, "issues": len(matches), "examples": [m.message for m in matches[:5]]}
            penalty = min(40, len(matches) * 1.5)
        except Exception:
            penalty = 0.0
    else:
        penalty = 0.0

    final = max(0.0, round(base_score - penalty, 2))

    return {"final_score": final, "similarity": sim_pct, "base_score": base_score, "penalty": penalty, "grammar": grammar_info}


def apply_rubric(rubric: dict, model_ans: str, student_ans: str) -> Dict[str, Any]:
    """Very small rubric support. Expect rubric of form: {"criteria": [{"name":"Content","weight":0.7,"type":"similarity"}, ...]}"""
    try:
        criteria = rubric.get('criteria', [])
    except Exception:
        return heuristic_grade(model_ans, student_ans)

    if not criteria:
        return heuristic_grade(model_ans, student_ans)

    # compute similarity once
    vecs, method = embed_texts_with_fallback([model_ans, student_ans])
    sim = cosine_sim(vecs[0], vecs[1])
    sim_pct = max(0.0, min(1.0, sim))

    total_weight = sum(c.get('weight', 0) for c in criteria) or 1.0
    total = 0.0
    breakdown = []
    for c in criteria:
        name = c.get('name', 'criterion')
        w = c.get('weight', 0) / total_weight
        t = c.get('type', 'similarity')
        if t == 'similarity':
            subs = sim_pct * 100
        elif t == 'presence':
            # check simple keyword presence
            keywords = c.get('keywords', [])
            subs = 0.0
            if keywords:
                text_lower = student_ans.lower()
                hits = sum(1 for k in keywords if k.lower() in text_lower)
                subs = min(100.0, (hits / len(keywords)) * 100.0)
            else:
                subs = sim_pct * 100
        else:
            subs = sim_pct * 100
        total += subs * w
        breakdown.append({"name": name, "weight": round(w, 3), "subscore": round(subs, 2)})

    final = round(total, 2)
    return {"final_score": final, "breakdown": breakdown, "similarity": sim_pct}

# -------------------- Critical Thinking --------------------

def sentence_split(text: str) -> List[str]:
    import re
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]


def heuristic_ct_scores(text: str) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, List[str]]]:
    """Return (scores 0..1), suggestions, highlighted sentences per standard."""
    sents = sentence_split(text)
    lower = text.lower()
    scores = {}
    suggestions = {}
    highlights = {k: [] for k in PAUL_CT_RUBRIC.keys()}

    wc = len(text.split()) if text else 0

    for standard, info in PAUL_CT_RUBRIC.items():
        pats = info['patterns']
        hits = 0
        for p in pats:
            if p in lower:
                hits += 1
        # Score heuristic: presence of indicators + some length factor
        base = min(1.0, 0.2 + 0.25 * hits + 0.005 * min(400, wc))
        scores[standard] = float(min(1.0, base))
        suggestions[standard] = info['feedback']

        # collect sentences with patterns
        for s in sents:
            sl = s.lower()
            if any(p in sl for p in pats):
                highlights[standard].append(s)

    return scores, suggestions, highlights

# -------------------- Visualization --------------------

def create_ct_radar(ct_scores: Dict[str, float], title: str = "CT Profile") -> go.Figure:
    categories = list(ct_scores.keys())
    values = [ct_scores[k] for k in categories]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself', name='CT'))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], visible=True)), showlegend=False, title=title, height=420)
    return fig

# -------------------- Streamlit UI --------------------

def main():
    st.set_page_config(APP_TITLE, layout='wide')
    st.title(APP_TITLE)

    # session state
    if 'grading' not in st.session_state:
        st.session_state.grading = []
    if 'ct' not in st.session_state:
        st.session_state.ct = []

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", ["Dashboard", "Grading", "CT Analysis", "Resources"] )
        st.markdown("---")
        st.markdown("**Model status**")
        emb_model = get_embedding_model()
        st.write("SentenceTransformers:", "Available" if emb_model is not None else "Unavailable — TF-IDF fallback")
        st.write("Grammar tool:", "Available" if _HAS_LANGTOOL else "Unavailable")
        st.markdown("---")
        if st.button("Clear All Results"):
            st.session_state.grading = []
            st.session_state.ct = []
            st.experimental_rerun()

    if page == "Dashboard":
        show_dashboard()
    elif page == "Grading":
        show_grading()
    elif page == "CT Analysis":
        show_ct()
    elif page == "Resources":
        show_resources()


def show_dashboard():
    st.subheader("Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total graded", len(st.session_state.grading))
    with col2:
        st.metric("CT analyses", len(st.session_state.ct))

    if st.session_state.grading:
        df = pd.DataFrame(st.session_state.grading)
        st.dataframe(df[['name','final_score','timestamp']].sort_values('timestamp', ascending=False).head(20))
    else:
        st.info("No grading done yet — go to Grading tab")


def show_grading():
    st.header("Automated Grading")
    with st.expander("Upload materials"):
        model_file = st.file_uploader("Model answer (txt/docx/pdf)", type=['txt','docx','pdf'])
        model_text_area = st.text_area("Or paste model answer (overrides uploaded)", height=120)
        student_files = st.file_uploader("Student submissions (multiple)", accept_multiple_files=True, type=['txt','docx','pdf'])
        student_paste = st.text_area("Or paste submissions separated by \n---\n", height=120)
        rubric_file = st.file_uploader("Optional rubric (json)", type=['json'])

    model_text = model_text_area.strip() or (read_text_file(model_file) if model_file else "")

    if st.button("Start grading"):
        if not model_text.strip():
            st.error("Please supply a model answer (upload or paste)")
            return

        # build student list
        students = []  # list of tuples (name, text)
        if student_files:
            for f in student_files:
                txt = read_text_file(f)
                if txt.strip():
                    students.append((f.name, txt.strip()))
        if student_paste:
            parts = [p.strip() for p in student_paste.split('\n---\n') if p.strip()]
            for i, p in enumerate(parts):
                students.append((f"Student_{i+1}", p))

        if not students:
            st.error("No student submissions provided")
            return

        # rubric
        rubric = None
        if rubric_file:
            try:
                rubric = json.loads(rubric_file.read().decode('utf-8'))
            except Exception as e:
                st.warning(f"Could not parse rubric JSON: {e}")
                rubric = None

        prog = st.progress(0)
        results = []
        for i, (name, text) in enumerate(students):
            prog.progress(int((i / max(1, len(students))) * 100))
            try:
                if rubric:
                    res = apply_rubric(rubric, model_text, text)
                else:
                    res = heuristic_grade(model_text, text)
                record = {
                    'name': name,
                    'final_score': res.get('final_score', 0),
                    'similarity': res.get('similarity', 0),
                    'timestamp': datetime.now().isoformat()
                }
                results.append({**record, 'details': res})
            except Exception as e:
                results.append({'name': name, 'error': str(e), 'timestamp': datetime.now().isoformat()})
        prog.progress(100)
        st.session_state.grading = results
        st.success(f"Graded {len(results)} submissions")

    # Display results
    if st.session_state.grading:
        st.markdown("### Last grading results")
        for r in st.session_state.grading:
            if 'error' in r:
                st.error(f"{r.get('name')}: {r.get('error')}")
                continue
            score = r.get('final_score', 0)
            st.write(f"**{r.get('name')}** — **{score}/100**")
            with st.expander("Details"):
                st.json(r.get('details', {}))


def show_ct():
    st.header("Critical Thinking (CT) Analysis")
    uploaded = st.file_uploader("Upload student files for CT (multiple)", accept_multiple_files=True, type=['txt','docx','pdf'])
    pasted = st.text_area("Or paste texts separated by \n---\n", height=150)

    if st.button("Run CT analysis"):
        submissions = []
        if uploaded:
            for f in uploaded:
                txt = read_text_file(f)
                if txt.strip():
                    submissions.append((f.name, txt))
        if pasted:
            parts = [p.strip() for p in pasted.split('\n---\n') if p.strip()]
            for i, p in enumerate(parts):
                submissions.append((f"Pasted_{i+1}", p))

        if not submissions:
            st.error("No inputs provided for CT analysis")
            return

        prog = st.progress(0)
        out = []
        for i, (name, text) in enumerate(submissions):
            prog.progress(int((i / max(1, len(submissions))) * 100))
            try:
                scores, suggestions, highlights = heuristic_ct_scores(text)
                out.append((name, scores, suggestions, highlights))
            except Exception as e:
                out.append((name, {}, {}, {}))
        prog.progress(100)
        st.session_state.ct = out
        st.success(f"Analyzed {len(out)} submissions")

    if st.session_state.ct:
        for name, scores, suggestions, highlights in st.session_state.ct:
            with st.expander(f"{name}"):
                st.subheader("CT Radar")
                fig = create_ct_radar(scores, title=f"CT Profile — {name}")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Improvement Suggestions")
                for k, v in suggestions.items():
                    st.write(f"**{k}** — {v} (score: {scores.get(k):.2f})")

                st.subheader("Highlighted sentences")
                for k, sents in highlights.items():
                    if sents:
                        st.markdown(f"**{k}**")
                        for s in sents[:6]:
                            st.write(f"- {s}")


def show_resources():
    st.header("Resources & Notes")
    st.markdown("- Use `python -m spacy download en_core_web_sm` to locally install spaCy small model if needed.")
    st.markdown("- If SentenceTransformers fails to install on Streamlit Cloud or your environment is resource-constrained, the app falls back to TF-IDF similarity.")
    st.markdown("- For production / large scale use, consider placing heavy models behind an API or using a dedicated inference service.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.exception(e)
        traceback.print_exc()
