"""
CT-Learner-Pro V1.0 - Unified Educational Platform
Advanced RAG-based Auto-Grader with Critical Thinking Analysis
Enhanced with heat maps, advanced visualizations, and 0-10 grading scale
"""

import os
import io
import re
import json
import math
import tempfile
import time
import requests
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter, defaultdict
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# File processing
import docx
import pdfplumber
try:
    import docx2txt
except Exception:
    docx2txt = None

try:
    import language_tool_python
    lang_tool = language_tool_python.LanguageTool("en-US")
except Exception:
    lang_tool = None

# NLP & ML
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==================== STYLING & CONFIGURATION ====================
COLOR_SCHEME = {
    "primary": "#2E7D5B",  # Academic green
    "secondary": "#4A90E2", # Trust blue
    "accent": "#FF6B35",    # Highlight orange
    "background": "#F8F9FA", # Clean white
    "text": "#2C3E50",      # Professional dark
    "success": "#27AE60",   # Achievement green
    "warning": "#F39C12",
    "danger": "#E74C3C",
    "info": "#3498DB",
    "light": "#ECF0F1",
    "dark": "#2C3E50"
}

# ==================== CRITICAL THINKING RUBRIC ====================
PAUL_CT_RUBRIC = {
    "Clarity": {
        "description": "Demonstrate clarity in conversation; provide examples to illustrate the point as appropriate.",
        "feedback_q": "Could you elaborate further; give an example or illustrate what you mean?",
        "patterns": ["for example", "for instance", "e.g.", "such as", "to illustrate", "in other words", "specifically"],
        "color": COLOR_SCHEME["primary"]
    },
    "Accuracy": {
        "description": "Provide accurate and verifiable information to support the ideas/position.",
        "feedback_q": "How could we check on that; verify or test; find out if that is true?",
        "patterns": ["http", "www.", "cite", "according to", "%", "data", "study", "research", "survey", "statistics", "source"],
        "color": COLOR_SCHEME["secondary"]
    },
    "Relevance": {
        "description": "Respond to the issues/question/problem with related information. Avoid irrelevant details.",
        "feedback_q": "How does that relate to the problem; bear on the question; help us with the issue?",
        "patterns": ["related to", "regarding", "pertaining to", "in relation to", "connected to"],
        "color": COLOR_SCHEME["info"]
    },
    "Significance": {
        "description": "Able to identify the central idea. Contribute with important and new points.",
        "feedback_q": "Is this the most important problem to consider? Which of these facts are most important?",
        "patterns": ["main", "central", "important", "key", "primary", "crucial", "essential", "significant"],
        "color": COLOR_SCHEME["success"]
    },
    "Logic": {
        "description": "Organize each piece of information in a logical order so it makes sense to others.",
        "feedback_q": "Does all this make sense together? Does what you say follow from the evidence?",
        "patterns": ["therefore", "because", "thus", "hence", "however", "but", "consequently", "as a result", "so that"],
        "color": COLOR_SCHEME["accent"]
    },
    "Precision": {
        "description": "Select specific information, stay focused and avoid redundancy.",
        "feedback_q": "Could you be more specific; be more exact; give more details?",
        "patterns": ["specifically", "exactly", "precisely", "in particular", "specifically", "detailed"],
        "color": "#DDA0DD"
    },
    "Fairness": {
        "description": "Demonstrate open-mindedness, consider pros and cons and challenge assumptions.",
        "feedback_q": "Am I sympathetically representing the viewpoints of others? Do I have vested interests?",
        "patterns": ["on the other hand", "although", "consider", "pros and cons", "however", "both", "despite", "alternatively"],
        "color": "#98D8C8"
    },
    "Depth": {
        "description": "Being thorough; examine the intricacies in the argument.",
        "feedback_q": "What are some of the complexities of this question? What difficulties must we deal with?",
        "patterns": ["because", "although", "since", "whereas", "in depth", "intricacy", "complex", "complexity", "thorough"],
        "color": "#F7DC6F"
    },
    "Breadth": {
        "description": "Able to offer / consider alternative views or solutions.",
        "feedback_q": "Do we need another perspective? What are alternative ways?",
        "patterns": ["alternatively", "another view", "different perspective", "other view", "in contrast", "on the contrary"],
        "color": "#BB8FCE"
    }
}

# ==================== ENHANCED STYLES ====================
CUSTOM_CSS = f"""
<style>
    .main-header {{
        font-size: 2.8rem !important;
        color: {COLOR_SCHEME['primary']};
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
    }}
    .platform-subtitle {{
        font-size: 1.3rem !important;
        color: {COLOR_SCHEME['text']};
        text-align: center;
        margin-bottom: 2.5rem;
        font-family: 'Source Sans Pro', sans-serif;
    }}
    .module-card {{
        background: linear-gradient(135deg, {COLOR_SCHEME['background']} 0%, #FFFFFF 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid {COLOR_SCHEME['primary']};
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }}
    .module-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }}
    .metric-card {{
        background-color: {COLOR_SCHEME['light']};
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid {COLOR_SCHEME['primary']};
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }}
    .success-box {{
        padding: 1.2rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #27AE60;
        margin: 1rem 0;
    }}
    .warning-box {{
        padding: 1.2rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #F39C12;
        margin: 1rem 0;
    }}
    .progress-bar {{
        height: 10px;
        background-color: #e9ecef;
        border-radius: 6px;
        margin: 0.8rem 0;
        overflow: hidden;
    }}
    .progress-fill {{
        height: 100%;
        background: linear-gradient(90deg, {COLOR_SCHEME['primary']}, {COLOR_SCHEME['secondary']});
        transition: width 0.5s ease-in-out;
        border-radius: 6px;
    }}
    .feedback-item {{
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background-color: {COLOR_SCHEME['light']};
        border-left: 4px solid {COLOR_SCHEME['primary']};
        transition: all 0.3s ease;
    }}
    .feedback-item:hover {{
        background-color: {COLOR_SCHEME['background']};
        transform: translateX(5px);
    }}
    .highlight-sentence {{
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        border-left: 4px solid;
        background-color: rgba(255, 255, 255, 0.8);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}
    .grade-badge {{
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 0.5rem 0;
    }}
</style>
"""

# ==================== CORE UTILITIES ====================
@st.cache_resource(show_spinner="🔄 Loading AI engines...")
def load_models():
    """Load all required models"""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load emotion model if needed
    emotion_model = None
    try:
        emotion_model = {
            "tok": AutoTokenizer.from_pretrained("j-hartmann/emotion-english-roberta-large"),
            "model": AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-roberta-large"),
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }
        emotion_model["model"].to(emotion_model["device"])
    except Exception as e:
        st.warning(f"Emotion model not available: {e}")
    
    return embedding_model, emotion_model

embedding_model, emotion_model = load_models()

def read_text_file(uploaded_file) -> str:
    """Enhanced file reader with progress tracking"""
    if uploaded_file is None:
        return ""
    
    with st.status(f"📄 Processing {uploaded_file.name}...", state="running") as status:
        try:
            content = uploaded_file.getvalue()
            name = uploaded_file.name.lower()
            
            if name.endswith(".txt"):
                result = content.decode("utf-8")
            elif name.endswith(".docx"):
                if docx2txt:
                    tmp_path = f"/tmp/temp_upload_{int(time.time())}.docx"
                    with open(tmp_path, "wb") as f:
                        f.write(content)
                    result = docx2txt.process(tmp_path)
                else:
                    result = ""
                    st.warning("📝 docx2txt not installed; please install with: pip install docx2txt")
            elif name.endswith(".pdf"):
                result = extract_text_from_pdf_bytes(content)
            else:
                result = content.decode("utf-8")
                
            status.update(label=f"✅ Processed {uploaded_file.name}", state="complete")
            return result
        except Exception as e:
            status.update(label=f"❌ Error processing {uploaded_file.name}", state="error")
            return ""

def extract_text_from_pdf_bytes(b: bytes) -> str:
    """Extract text from PDF bytes"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(b); f.flush()
        tmp = f.name
    try:
        text_pages = []
        with pdfplumber.open(tmp) as pdf:
            for p in pdf.pages:
                text_pages.append(p.extract_text() or "")
        return "\n".join(text_pages)
    except Exception:
        return ""
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed texts with progress tracking"""
    texts = [t if t is not None else "" for t in texts]
    
    progress_text = "🔍 Analyzing text similarities..."
    my_bar = st.progress(0, text=progress_text)
    
    for i in range(100):
        time.sleep(0.01)
        my_bar.progress(i + 1, text=progress_text)
    
    vectors = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    my_bar.empty()
    
    return vectors

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity"""
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])

# ==================== ENHANCED GRADING MODULE (0-10 SCALE) ====================
def apply_rubric_json(rubric: dict, model_ans: str, student_ans: str) -> Dict[str, Any]:
    """Apply rubric-based grading with 0-10 scale"""
    criteria = rubric.get("criteria", [])
    if not criteria:
        return heuristic_grade(model_ans, student_ans)

    with st.status("📊 Applying rubric criteria...", state="running") as status:
        vecs = embed_texts([model_ans, student_ans])
        sim = cosine_sim(vecs[0], vecs[1])
        sim_norm = max(0.0, min((sim + 1) / 2.0, 1.0))
        g = grammar_check(student_ans)
        issues = g["issues_count"] if g.get("available") else None

        total_weight = sum(c.get("weight", 0) for c in criteria) or 1.0
        total_score = 0.0
        breakdown = []
        
        for i, c in enumerate(criteria):
            name = c.get("name", f"Criterion {i+1}")
            w = c.get("weight", 0) / total_weight
            t = c.get("type", "similarity")
            subscore = 0.0
            
            if t == "similarity":
                subscore = sim_norm * 10  # 0-10 scale
            elif t == "grammar_penalty":
                if issues is None:
                    subscore = 10.0
                else:
                    penalty_per = c.get("penalty_per_issue", 0.15)  # Adjusted for 0-10 scale
                    subscore = max(0.0, 10.0 - penalty_per * issues)
            else:
                subscore = sim_norm * 10
                
            total_score += subscore * w
            breakdown.append({
                "criterion": name, 
                "weight": round(w,3), 
                "subscore": round(subscore,2),
                "type": t
            })
        
        status.update(label="✅ Rubric applied successfully", state="complete")

    final_score = round(total_score, 2)
    return {
        "final_score": final_score, 
        "breakdown": breakdown, 
        "similarity": sim_norm, 
        "grammar": g,
        "grading_method": "rubric"
    }

def heuristic_grade(model_ans: str, student_ans: str) -> Dict[str, Any]:
    """Heuristic grading fallback with 0-10 scale"""
    with st.status("🎯 Computing similarity scores...", state="running") as status:
        vecs = embed_texts([model_ans, student_ans])
        sim = cosine_sim(vecs[0], vecs[1])
        sim_norm = max(0.0, min((sim + 1) / 2.0, 1.0))
        base = sim_norm * 10  # 0-10 scale
        g = grammar_check(student_ans)
        penalty = 0.0
        
        if g.get("available"):
            issues = g["issues_count"]
            penalty = min(4.0, issues * 0.15)  # Adjusted for 0-10 scale
            
        final = round(max(0.0, base - penalty), 2)
        breakdown = [
            {"criterion": "Content Similarity", "weight": 0.8, "subscore": round(base,2), "type": "similarity"},
            {"criterion": "Grammar & Mechanics", "weight": 0.2, "subscore": round(max(0, 10 - penalty),2), "type": "grammar"}
        ]
        status.update(label="✅ Automatic grading completed", state="complete")
        
    return {
        "final_score": final, 
        "breakdown": breakdown, 
        "similarity": sim_norm, 
        "grammar": g, 
        "penalty": penalty,
        "grading_method": "heuristic"
    }

def grammar_check(text: str) -> Dict[str, Any]:
    """Grammar checking with language tool"""
    if not lang_tool or not text.strip():
        return {"available": False, "issues_count": 0, "examples": []}
    
    with st.status("🔍 Checking grammar and spelling...", state="running") as status:
        matches = lang_tool.check(text)
        examples = []
        for m in matches[:6]:
            context = text[max(0, m.offset-30): m.offset+30]
            examples.append({
                "message": m.message, 
                "context": context,
                "suggestions": m.replacements[:3]
            })
        status.update(label=f"✅ Found {len(matches)} grammar issues", state="complete")
        
    return {"available": True, "issues_count": len(matches), "examples": examples}

# ==================== ENHANCED VISUALIZATION FUNCTIONS ====================
def create_heatmap(data: pd.DataFrame, title: str, color_scale: str = "Viridis") -> go.Figure:
    """Create an enhanced heatmap with annotations"""
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale=color_scale,
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Criteria",
        yaxis_title="Students",
        height=500,
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor=COLOR_SCHEME['background']
    )
    
    return fig

def create_comprehensive_dashboard(grading_results, ct_results):
    """Create enhanced dashboard with multiple visualizations"""
    if not grading_results and not ct_results:
        st.info("No data available for dashboard. Complete some analyses first.")
        return
    
    st.subheader("📊 Comprehensive Analytics Dashboard")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_grade = np.mean([r.get('final_score', 0) for r in grading_results]) if grading_results else 0
        st.metric("Average Grade", f"{avg_grade:.1f}/10")
    
    with col2:
        avg_ct = np.mean([np.mean(list(r[1].values())) for r in ct_results]) if ct_results else 0
        st.metric("Average CT Score", f"{avg_ct:.2f}")
    
    with col3:
        strong_performers = len([r for r in grading_results if r.get('final_score', 0) >= 8])
        st.metric("Strong Performers", f"{strong_performers}/{len(grading_results)}")
    
    with col4:
        improvement_needed = len([r for r in ct_results if np.mean(list(r[1].values())) < 0.6])
        st.metric("Need CT Support", f"{improvement_needed}/{len(ct_results)}")
    
    # Enhanced Visualizations in Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Grade Distribution", "🎯 CT Analysis", "📊 Performance Matrix", "📋 Comparative Analysis"])
    
    with tab1:
        # Grade distribution with multiple charts
        if grading_results:
            col1, col2 = st.columns(2)
            with col1:
                grades = [r.get('final_score', 0) for r in grading_results]
                fig = px.histogram(x=grades, nbins=10, title="Grade Distribution",
                                 color_discrete_sequence=[COLOR_SCHEME["primary"]])
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Grade progression (simulated)
                students = [r.get('name', f'Student {i+1}') for i, r in enumerate(grading_results)]
                fig = px.line(x=range(len(grades)), y=sorted(grades), title="Grade Progression",
                            labels={'x': 'Student Rank', 'y': 'Grade'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # CT Analysis Heatmap
        if ct_results:
            ct_df_data = []
            for filename, ct_scores, _, _ in ct_results:
                row = {"Filename": filename}
                row.update(ct_scores)
                ct_df_data.append(row)
            
            ct_df = pd.DataFrame(ct_df_data)
            ct_heatmap_data = ct_df.set_index("Filename")
            
            fig = create_heatmap(ct_heatmap_data, "Critical Thinking Skills Heatmap", "Viridis")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Performance correlation matrix
        if grading_results and ct_results:
            # Create combined performance matrix
            performance_data = []
            for i, (grade_result, ct_result) in enumerate(zip(grading_results, ct_results)):
                row = {
                    'Student': grade_result.get('name', f'Student {i+1}'),
                    'Grade': grade_result.get('final_score', 0),
                    'Avg_CT_Score': np.mean(list(ct_result[1].values()))
                }
                performance_data.append(row)
            
            perf_df = pd.DataFrame(performance_data)
            
            col1, col2 = st.columns(2)
            with col1:
                # Scatter plot: Grade vs CT Score
                fig = px.scatter(perf_df, x='Grade', y='Avg_CT_Score', hover_data=['Student'],
                               title="Grade vs Critical Thinking Correlation",
                               color_discrete_sequence=[COLOR_SCHEME["accent"]])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Performance matrix
                fig = px.density_heatmap(perf_df, x='Grade', y='Avg_CT_Score',
                                       title="Performance Density Matrix",
                                       nbinsx=8, nbinsy=8)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Comparative analysis
        if grading_results:
            # Create comparison bar chart
            students = [r.get('name', f'Student {i+1}') for i, r in enumerate(grading_results)]
            grades = [r.get('final_score', 0) for r in grading_results]
            
            fig = go.Figure(data=[
                go.Bar(name='Grades', x=students, y=grades, 
                      marker_color=COLOR_SCHEME["primary"])
            ])
            
            fig.update_layout(
                title="Student Performance Comparison",
                xaxis_title="Students",
                yaxis_title="Grade (0-10)",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

def create_ct_radar_chart(ct_scores: Dict[str, float], title: str) -> go.Figure:
    """Create radar chart for CT scores"""
    categories = list(ct_scores.keys())
    values = list(ct_scores.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor=f'rgba({int(COLOR_SCHEME["primary"][1:3], 16)}, {int(COLOR_SCHEME["primary"][3:5], 16)}, {int(COLOR_SCHEME["primary"][5:7], 16)}, 0.3)',
        line=dict(color=COLOR_SCHEME["primary"], width=2),
        name='CT Standards'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title=title,
        height=400,
        font=dict(family="Inter, sans-serif")
    )
    return fig

# ==================== CRITICAL THINKING MODULE ====================
def sentence_split(text: str) -> List[str]:
    """Split text into sentences"""
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]

def tokenize_simple(s: str) -> List[str]:
    """Simple tokenization"""
    return re.findall(r"\w+['-]?\w*|\w+", s.lower())

def highlight_ct_sentences(text: str) -> Dict[str, List[Tuple[str, str]]]:
    """Highlight sentences matching CT criteria"""
    highlighted = {standard: [] for standard in PAUL_CT_RUBRIC.keys()}
    sents = sentence_split(text)
    
    for sent in sents:
        sent_lower = sent.lower()
        for standard, data in PAUL_CT_RUBRIC.items():
            for pattern in data["patterns"]:
                if pattern in sent_lower:
                    highlighted[standard].append((sent, data["color"]))
                    break
    
    return highlighted

def heuristic_ct_scores(text: str) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, List[Tuple[str, str]]]]:
    """Calculate CT scores with highlighting"""
    sents = sentence_split(text)
    tokens = tokenize_simple(text)
    word_count = len(tokens)
    scores = {}
    suggestions = {}
    
    # Get highlighted sentences
    highlighted = highlight_ct_sentences(text)
    
    # Calculate scores
    clarity_indicators = ["for example", "for instance", "e.g.", "such as", "to illustrate"]
    clarity_score = 1.0 if any(phrase in text.lower() for phrase in clarity_indicators) else (0.3 if word_count < 50 else 0.5)
    scores["Clarity"] = clarity_score
    suggestions["Clarity"] = PAUL_CT_RUBRIC["Clarity"]["feedback_q"]

    accuracy_indicators = ["http", "www.", "cite", "according to", "%", "data", "study", "reported", "survey"]
    accuracy_score = 1.0 if any(ind in text.lower() for ind in accuracy_indicators) else 0.4
    scores["Accuracy"] = accuracy_score
    suggestions["Accuracy"] = PAUL_CT_RUBRIC["Accuracy"]["feedback_q"]

    if sents:
        first = tokenize_simple(sents[0])
        overlap_counts = sum(1 for sent in sents[1:] if any(w in tokenize_simple(sent) for w in first[:5]))
        relevance_score = min(1.0, (overlap_counts+1) / max(1, len(sents)))
    else:
        relevance_score = 0.0
    scores["Relevance"] = relevance_score
    suggestions["Relevance"] = PAUL_CT_RUBRIC["Relevance"]["feedback_q"]

    sign_ind = ["main", "central", "important", "key", "primary"]
    sign_score = 1.0 if any(w in text.lower() for w in sign_ind) else min(0.9, 0.6 + 0.01 * (word_count/100))
    scores["Significance"] = sign_score
    suggestions["Significance"] = PAUL_CT_RUBRIC["Significance"]["feedback_q"]

    connectors = ["therefore", "because", "thus", "hence", "however", "but", "consequently", "as a result", "so that"]
    logic_score = min(1.0, sum(1 for c in connectors if c in text.lower()) * 0.25)
    scores["Logic"] = logic_score
    suggestions["Logic"] = PAUL_CT_RUBRIC["Logic"]["feedback_q"]

    hedges = ["maybe", "perhaps", "might", "could", "seems", "appears"]
    precision_score = max(0.0, 1.0 - 0.2 * sum(1 for h in hedges if h in text.lower()))
    if word_count < 40:
        precision_score *= 0.5
    scores["Precision"] = precision_score
    suggestions["Precision"] = PAUL_CT_RUBRIC["Precision"]["feedback_q"]

    fairness_ind = ["on the other hand", "although", "consider", "pros and cons", "however", "both", "despite"]
    fairness_score = 1.0 if any(p in text.lower() for p in fairness_ind) else 0.45
    scores["Fairness"] = fairness_score
    suggestions["Fairness"] = PAUL_CT_RUBRIC["Fairness"]["feedback_q"]

    depth_ind = ["because", "although", "since", "whereas", "in depth", "intricacy", "complex", "complexity"]
    depth_score = min(1.0, 0.25 * sum(1 for d in depth_ind if d in text.lower()) + 0.3)
    scores["Depth"] = depth_score
    suggestions["Depth"] = PAUL_CT_RUBRIC["Depth"]["feedback_q"]

    breadth_ind = ["alternatively", "another view", "different perspective", "other view", "in contrast"]
    breadth_score = 1.0 if any(p in text.lower() for p in breadth_ind) else 0.4
    scores["Breadth"] = breadth_score
    suggestions["Breadth"] = PAUL_CT_RUBRIC["Breadth"]["feedback_q"]

    for k in scores:
        scores[k] = float(max(0.0, min(1.0, scores[k])))
    
    return scores, suggestions, highlighted

# ==================== MAIN APPLICATION ====================
def main():
    # Page configuration
    st.set_page_config(
        page_title="CT-Learner Pro V1.0", 
        layout="wide",
        page_icon="🧠",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styles
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🧠 CT-Learner Pro V1.0</div>', unsafe_allow_html=True)
    st.markdown('<div class="platform-subtitle">Unified Educational Platform • 0-10 Grading Scale • Enhanced Analytics</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_module' not in st.session_state:
        st.session_state.current_module = "dashboard"
    if 'grading_results' not in st.session_state:
        st.session_state.grading_results = []
    if 'ct_results' not in st.session_state:
        st.session_state.ct_results = []
    if 'student_data' not in st.session_state:
        st.session_state.student_data = {}
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        # Module selection
        modules = {
            "dashboard": "📊 Dashboard",
            "grading": "🎓 Auto-Grading (0-10)",
            "critical_thinking": "💭 CT Analysis", 
            "progress": "📈 Progress Tracking",
            "resources": "📚 Learning Resources"
        }
        
        for module_id, module_name in modules.items():
            if st.button(module_name, key=f"nav_{module_id}", 
                        use_container_width=True,
                        type="primary" if st.session_state.current_module == module_id else "secondary"):
                st.session_state.current_module = module_id
                st.rerun()
        
        st.markdown("---")
        st.markdown("## ⚙️ System Status")
        
        # System info
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ Grading Engine")
        with col2:
            st.success("✅ CT Analyzer")
        
        st.info(f"🤖 AI Feedback: {'✅ Available' if os.getenv('GROQ_API_KEY') else '🔶 Configure'}")
        st.info(f"📝 Grammar Check: {'✅ Available' if lang_tool else '🔶 Basic'}")
        
        # Quick actions
        st.markdown("---")
        st.markdown("## 🚀 Quick Actions")
        if st.button("🔄 Clear All Data", use_container_width=True):
            st.session_state.grading_results = []
            st.session_state.ct_results = []
            st.session_state.student_data = {}
            st.success("Data cleared successfully!")
    
    # Main content area based on selected module
    if st.session_state.current_module == "dashboard":
        show_dashboard()
    elif st.session_state.current_module == "grading":
        show_grading_module()
    elif st.session_state.current_module == "critical_thinking":
        show_ct_module()
    elif st.session_state.current_module == "progress":
        show_progress_module()
    elif st.session_state.current_module == "resources":
        show_resources_module()

def show_dashboard():
    """Main dashboard view"""
    st.markdown("## 🏠 Welcome to CT-Learner Pro")
    
    # Feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <h3>🎓 Automated Grading (0-10 Scale)</h3>
            <p>AI-powered assignment evaluation with RAG-based scoring on 0-10 scale.</p>
            <ul>
                <li>Rubric-based assessment</li>
                <li>Grammar and style checking</li>
                <li>Personalized feedback</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="module-card">
            <h3>📈 Enhanced Analytics</h3>
            <p>Comprehensive tracking with heat maps and advanced visualizations.</p>
            <ul>
                <li>Performance heat maps</li>
                <li>Comparative analysis</li>
                <li>Progress tracking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="module-card">
            <h3>💭 Critical Thinking</h3>
            <p>Advanced analysis of reasoning skills using Paul's framework.</p>
            <ul>
                <li>9-dimensional CT assessment</li>
                <li>Sentence-level highlighting</li>
                <li>Personalized improvement plans</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="module-card">
            <h3>📚 Learning Resources</h3>
            <p>Curated educational materials for skill development.</p>
            <ul>
                <li>Interactive exercises</li>
               
