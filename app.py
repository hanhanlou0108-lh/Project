import streamlit as st
from transformers import pipeline
import time
import os
import torch
import pandas as pd
from io import BytesIO
from datetime import datetime

st.set_page_config(
    page_title="ReviewGuard System",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CSS_STYLES = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #faf9f7;
        --bg-secondary: #f5f3f0;
        --text-primary: #1a1a1a;
        --text-secondary: #6b6b6b;
        --text-muted: #9a9a9a;
        --accent-teal: #2dd4bf;
        --accent-coral: #fb7185;
        --accent-amber: #fbbf24;
        --accent-violet: #a78bfa;
        --card-shadow: 0 4px 24px rgba(0,0,0,0.06);
        --border-radius: 20px;
        --transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stApp { background: var(--bg-primary); }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 400px;
        background: linear-gradient(180deg, #f0f9ff 0%, #faf9f7 100%);
        z-index: -1;
        opacity: 0.7;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        color: var(--text-primary) !important;
    }
    
    p, span, div, label { font-family: 'DM Sans', sans-serif !important; }
    
    .hero-section {
        text-align: center;
        padding: 2rem 0 1.25rem;
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.75rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdfa 100%);
        border: 1px solid #99f6e4;
        padding: 0.35rem 0.8rem;
        border-radius: 100px;
        font-size: 0.75rem;
        color: #0d9488;
        margin-top: 0.6rem;
        font-weight: 500;
    }
    
    .input-card {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(0,0,0,0.04);
        margin-bottom: 1.25rem;
    }
    
    .section-label {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 0.6rem;
    }
    
    .stTextArea textarea {
        border: 2px solid #e5e5e5;
        border-radius: 12px;
        font-size: 0.9rem;
        padding: 0.8rem;
        background: #fafafa;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--accent-teal);
        background: white;
        box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.1);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #1a1a1a 0%, #333333 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.6rem 1.25rem;
        height: 42px;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    
    .results-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1.25rem 0;
    }
    
    .result-card {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.25rem;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(0,0,0,0.04);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
    }
    
    .result-card.sentiment::before {
        background: linear-gradient(90deg, var(--accent-teal) 0%, var(--accent-violet) 100%);
    }
    
    .result-card.fake::before {
        background: linear-gradient(90deg, var(--accent-coral) 0%, var(--accent-amber) 100%);
    }
    
    .result-header {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.8rem;
    }
    
    .result-icon {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
    }
    
    .result-icon.positive { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); }
    .result-icon.negative { background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%); }
    .result-icon.fake { background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%); }
    .result-icon.real { background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); }
    
    .result-label {
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .result-sublabel { font-size: 0.75rem; color: var(--text-secondary); }
    
    .metric-row {
        display: flex;
        gap: 1rem;
        padding-top: 0.8rem;
        border-top: 1px solid #f0f0f0;
    }
    
    .metric-item { flex: 1; }
    
    .metric-value {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .metric-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
    }
    
    .confidence-bar {
        height: 4px;
        background: #f0f0f0;
        border-radius: 2px;
        overflow: hidden;
        margin-top: 0.3rem;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 2px;
    }
    
    .confidence-fill.high { background: linear-gradient(90deg, var(--accent-teal) 0%, #34d399 100%); }
    .confidence-fill.medium { background: linear-gradient(90deg, var(--accent-amber) 0%, #fcd34d 100%); }
    .confidence-fill.low { background: linear-gradient(90deg, var(--accent-coral) 0%, #fca5a5 100%); }
    
    .summary-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border-radius: var(--border-radius);
        padding: 1.25rem;
        color: white;
        margin: 1.25rem 0;
    }
    
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 0.8rem;
    }
    
    .summary-metric { text-align: center; }
    .summary-metric-value { font-family: 'Playfair Display', serif; font-size: 1.25rem; font-weight: 600; }
    .summary-metric-label { font-size: 0.7rem; color: rgba(255,255,255,0.6); }
    
    .suggestion-box {
        background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
        border: 1px solid #fde047;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    
    .suggestion-icon { font-size: 1rem; }
    .suggestion-text { font-size: 0.85rem; color: #713f12; }
    
    .history-section { margin-top: 1.5rem; }
    .history-title { font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.8rem; }
    
    .history-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }
    
    .history-card {
        background: white;
        border-radius: var(--border-radius);
        padding: 1rem;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(0,0,0,0.04);
    }
    
    .history-card-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-bottom: 0.6rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .history-item {
        background: var(--bg-secondary);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.35rem;
        font-size: 0.75rem;
        border-left: 3px solid transparent;
    }
    
    .history-item.fake { border-left-color: var(--accent-coral); }
    .history-item.negative { border-left-color: var(--accent-violet); }
    
    .history-item-text { color: var(--text-primary); margin-bottom: 0.15rem; line-height: 1.3; }
    .history-item-meta { font-size: 0.65rem; color: var(--text-muted); }
    
    .export-section {
        background: white;
        border-radius: var(--border-radius);
        padding: 1rem 1.25rem;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(0,0,0,0.04);
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 1.25rem;
        gap: 1rem;
    }
    
    .export-stats { display: flex; gap: 2rem; }
    .export-stat { text-align: center; }
    .export-stat-value { font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 600; color: var(--text-primary); }
    .export-stat-label { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; }
    .export-actions { display: flex; gap: 0.6rem; }
    
    .footer {
        text-align: center;
        padding: 1.5rem 0 1rem;
        margin-top: 1.5rem;
        border-top: 1px solid #e5e5e5;
    }
    
    .footer-text { font-size: 0.75rem; color: var(--text-muted); line-height: 1.5; }
    
    .batch-progress {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.25rem;
        box-shadow: var(--card-shadow);
        margin: 1.25rem 0;
    }
    
    .progress-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem; }
    .progress-title { font-weight: 600; color: var(--text-primary); }
    .progress-stats { font-size: 0.8rem; color: var(--text-secondary); }
    
    .stProgress > div > div > div { background: linear-gradient(90deg, var(--accent-teal) 0%, var(--accent-violet) 100%); }
    
    .success-toast {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #6ee7b7;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 0.8rem 0;
    }
    
    .success-icon { font-size: 1.1rem; }
    .success-text { font-size: 0.85rem; color: #065f46; }
    
    .import-section {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px dashed #7dd3fc;
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
        margin-top: 0.8rem;
    }
    
    .import-title { font-weight: 600; color: #0369a1; font-size: 0.85rem; margin-bottom: 0.2rem; }
    .import-desc { font-size: 0.7rem; color: #0c4a6e; opacity: 0.8; }
    
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .results-container, .history-grid, .summary-grid { grid-template-columns: 1fr; }
        .export-section { flex-direction: column; }
        .export-stats { width: 100%; justify-content: space-around; }
        .export-actions { width: 100%; flex-direction: column; }
    }
</style>
"""

st.markdown(CSS_STYLES, unsafe_allow_html=True)


def get_device():
    return 0 if torch.cuda.is_available() else -1


@st.cache_resource
def load_models():
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=get_device()
    )

    fake_pipe = pipeline(
        "text-classification",
        model="hlou0108/fake-review-detector",
        device=get_device()
    )

    return sentiment_pipe, fake_pipe


def analyze_text(text, pipe, is_fake_detection=False):
    default_result = {
        "label": "ERROR", 
        "score": 0.0, 
        "display_label": "Error", 
        "icon": "❌", 
        "icon_class": "negative",
        "inference_time": 0.0
    }
    
    if pipe is None:
        return default_result
    
    if not text or not str(text).strip():
        return default_result
    
    try:
        start_time = time.time()
        result = pipe(str(text).strip())[0]
        inference_time = (time.time() - start_time) * 1000
        
        if is_fake_detection:
            is_fake = result["label"] in ["FAKE", "LABEL_1", "fake"]
            fake_prob = result["score"] if is_fake else 1 - result["score"]
            return {
                "label": result["label"],
                "score": result["score"],
                "display_label": "Suspected Fake" if is_fake else "Genuine Review",
                "icon": "⚠️" if is_fake else "✅",
                "icon_class": "fake" if is_fake else "real",
                "is_fake": is_fake,
                "fake_prob": fake_prob,
                "inference_time": inference_time
            }
        else:
            is_positive = result["label"].upper() == "POSITIVE"
            return {
                "label": result["label"],
                "score": result["score"],
                "display_label": "Positive" if is_positive else "Negative",
                "icon": "😊" if is_positive else "😞",
                "icon_class": "positive" if is_positive else "negative",
                "is_positive": is_positive,
                "inference_time": inference_time
            }
    except Exception as e:
        st.warning(f"Analysis error: {str(e)}")
        return default_result


def get_confidence_class(score):
    if score >= 0.8:
        return "high"
    elif score >= 0.6:
        return "medium"
    return "low"


def render_result_card(result, card_class):
    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-header">
            <div class="result-icon {result['icon_class']}">{result['icon']}</div>
            <div>
                <div class="result-label">{result['display_label']}</div>
                <div class="result-sublabel">Predicted: {result['label']}</div>
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-value">{result['score']:.1%}</div>
                <div class="metric-label">Confidence</div>
                <div class="confidence-bar">
                    <div class="confidence-fill {get_confidence_class(result['score'])}" 
                         style="width: {result['score']*100}%"></div>
                </div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{result['inference_time']:.0f}ms</div>
                <div class="metric-label">Inference Time</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def safe_get_time(timestamp_str):
    try:
        parts = timestamp_str.split()
        return parts[1] if len(parts) > 1 else "N/A"
    except (AttributeError, IndexError):
        return "N/A"


def render_history_card(title, icon, records, item_class):
    st.markdown(f"""
    <div class="history-card">
        <div class="history-card-title">{icon} {title}</div>
    """, unsafe_allow_html=True)
    
    if records:
        for i, r in enumerate(records[-5:], 1):
            text = r.get('Review Text', 'N/A')[:45]
            fake_prob = r.get('Fake Probability', 'N/A')
            time_str = safe_get_time(r.get('Timestamp', 'N/A'))
            st.markdown(f"""
            <div class="history-item {item_class}">
                <div class="history-item-text">#{i} {text}...</div>
                <div class="history-item-meta">{fake_prob} | {time_str}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="color: var(--text-muted); font-size: 0.8rem; padding: 0.4rem;">No records yet</div>', unsafe_allow_html=True)


def get_excel_buffer(fake_reviews, negative_reviews):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, data in [("Fake Reviews", fake_reviews), ("Negative Reviews", negative_reviews)]:
            df = pd.DataFrame(data) if data else pd.DataFrame({"Message": ["No data to export"]})
            df.to_excel(writer, sheet_name=name, index=False)
    output.seek(0)
    return output


def get_suggestion(sentiment_result, fake_result):
    if sentiment_result["is_positive"] and fake_result["fake_prob"] > 0.7:
        return "💡", "This positive review shows high fake risk. Manual verification recommended."
    elif not sentiment_result["is_positive"]:
        return "📝", "Negative feedback detected. Consider addressing issues to improve customer experience."
    elif fake_result["fake_prob"] < 0.3:
        return "✨", "High credibility review. Valuable for product improvement insights."
    return "🔍", "Moderate confidence. Cross-reference with additional signals for better judgment."


def process_batch_reviews(reviews, sentiment_pipe, fake_pipe, progress_bar):
    fake_reviews = []
    negative_reviews = []
    valid_reviews = [r for r in reviews if r and str(r).strip()]
    total = len(valid_reviews)
    
    if total == 0:
        return fake_reviews, negative_reviews
    
    for i, review in enumerate(valid_reviews):
        review_text = str(review).strip()
        
        sentiment_result = analyze_text(review_text, sentiment_pipe, is_fake_detection=False)
        fake_result = analyze_text(review_text, fake_pipe, is_fake_detection=True)
        
        record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Review Text": review_text[:200],
            "Sentiment": sentiment_result["display_label"],
            "Sentiment Confidence": f"{sentiment_result['score']:.2%}",
            "Fake Probability": f"{fake_result['fake_prob']:.2%}",
            "Is Fake": "Yes" if fake_result["is_fake"] else "No"
        }
        
        if fake_result["is_fake"]:
            fake_reviews.append(record)
        if not sentiment_result["is_positive"]:
            negative_reviews.append(record)
        
        progress_bar.progress((i + 1) / total)
    
    return fake_reviews, negative_reviews


def init_session_state():
    defaults = {
        "fake_reviews_history": [],
        "negative_reviews_history": [],
        "input_text": "",
        "last_uploaded_file_name": "",
        "show_success": False,
        "success_message": "",
        "clear_history_flag": False,
        "export_and_clear_flag": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    """Main application function"""

    init_session_state()
    
    if st.session_state.clear_history_flag:
        st.session_state.fake_reviews_history = []
        st.session_state.negative_reviews_history = []
        st.session_state.clear_history_flag = False
        st.session_state.show_success = True
        st.session_state.success_message = "History cleared successfully!"
        st.rerun()
    
    if st.session_state.export_and_clear_flag:
        st.session_state.fake_reviews_history = []
        st.session_state.negative_reviews_history = []
        st.session_state.export_and_clear_flag = False
        st.session_state.show_success = True
        st.session_state.success_message = "Export complete! Records cleared."
        st.rerun()
    
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">ReviewGuard System</h1>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Loading AI models..."):
        sentiment_pipe, fake_pipe = load_models()
    
    if not sentiment_pipe and not fake_pipe:
        st.error("Failed to load models. Please check configuration.")
        return
    
    if st.session_state.show_success:
        st.markdown(f"""
        <div class="success-toast">
            <div class="success-icon">✅</div>
            <div class="success-text">{st.session_state.success_message}</div>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.show_success = False
    
    st.markdown('<div class="section-label">Enter Review Text</div>', unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Review text",
        value=st.session_state.input_text,
        height=100,
        placeholder="Paste or type a product review here...",
        label_visibility="collapsed"
    )
    
    st.markdown('<div class="section-label" style="margin-top: 0.8rem;">Quick Examples <span style="color: var(--accent-coral);">━</span> Negative <span style="color: var(--accent-amber);">━</span> Fake-like</div>', unsafe_allow_html=True)
    
    examples = [
        "Great product, fast delivery!",
        "Terrible quality, total waste!",
        "AMAZING!!! BEST EVER!!! BUY NOW!!!",
        "Not as described, very disappointed",
        "Good value for money overall",
        "Worst purchase I ever made!",
        "EXCELLENT!!! FIVE STARS!!! MUST HAVE!",
        "Average product, nothing special",
        "PERFECT!!! LIFE CHANGING!!! AMAZING!!!",
    ]
    
    for row in range(3):
        cols = st.columns(3)
        for col_idx in range(3):
            idx = row * 3 + col_idx
            if idx < len(examples):
                example_text = examples[idx]
                with cols[col_idx]:
                    display_text = example_text[:22] + ("..." if len(example_text) > 22 else "")
                    if st.button(display_text, key=f"ex_{idx}", use_container_width=True):
                        st.session_state.input_text = example_text
                        st.rerun()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analyze_btn = st.button("🔍 Analyze Review", type="primary", use_container_width=True)
    
    with col2:
        clear_input_btn = st.button("🗑️ Clear Input", type="secondary", use_container_width=True)
    
    st.markdown("""
    <div class="import-section">
        <div class="import-title">📁 Batch Import Reviews from Excel</div>
        <div class="import-desc">Upload .xlsx/.xls file with reviews in first column. Auto-detects fake & negative reviews.</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("batch_import_form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "Choose Excel file",
            type=["xlsx", "xls"],
            key="batch_uploader",
            label_visibility="collapsed"
        )
        process_btn = st.form_submit_button("📊 Process Batch", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if clear_input_btn:
        st.session_state.input_text = ""
        st.rerun()
    
    if process_btn and uploaded_file is not None:
        current_file_name = uploaded_file.name
        
        if current_file_name != st.session_state.last_uploaded_file_name:
            try:
                df = pd.read_excel(uploaded_file)
                
                if df.empty:
                    st.warning("The uploaded file is empty.")
                else:
                    review_column = df.columns[0]
                    reviews = df[review_column].dropna().tolist()
                    
                    if not reviews:
                        st.warning("No valid reviews found in the file.")
                    else:
                        st.markdown('<div class="batch-progress">', unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="progress-header">
                            <div class="progress-title">📊 Processing {len(reviews)} Reviews</div>
                            <div class="progress-stats">Please wait...</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        progress_bar = st.progress(0)
                        batch_fake, batch_negative = process_batch_reviews(
                            reviews, sentiment_pipe, fake_pipe, progress_bar
                        )
                        
                        st.session_state.fake_reviews_history.extend(batch_fake)
                        st.session_state.negative_reviews_history.extend(batch_negative)
                        st.session_state.last_uploaded_file_name = current_file_name
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.session_state.show_success = True
                        st.session_state.success_message = (
                            f"Batch complete! Found {len(batch_fake)} fake reviews "
                            f"and {len(batch_negative)} negative reviews."
                        )
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            st.info("This file has already been processed. Upload a different file to process new reviews.")
    
    if analyze_btn and user_input.strip():
        sentiment_result = analyze_text(user_input, sentiment_pipe, is_fake_detection=False)
        fake_result = analyze_text(user_input, fake_pipe, is_fake_detection=True)
        
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            render_result_card(sentiment_result, "sentiment")
        
        with col2:
            render_result_card(fake_result, "fake")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        total_time = sentiment_result["inference_time"] + fake_result["inference_time"]
        st.markdown(f"""
        <div class="summary-card">
            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.6); text-transform: uppercase; letter-spacing: 0.1em;">Analysis Summary</div>
            <div class="summary-grid">
                <div class="summary-metric">
                    <div class="summary-metric-value">{sentiment_result['display_label']} {sentiment_result['icon']}</div>
                    <div class="summary-metric-label">Sentiment</div>
                </div>
                <div class="summary-metric">
                    <div class="summary-metric-value">{fake_result['fake_prob']:.0%}</div>
                    <div class="summary-metric-label">Fake Risk</div>
                </div>
                <div class="summary-metric">
                    <div class="summary-metric-value">{total_time:.0f}ms</div>
                    <div class="summary-metric-label">Total Time</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        suggestion_icon, suggestion_text = get_suggestion(sentiment_result, fake_result)
        st.markdown(f"""
        <div class="suggestion-box">
            <div class="suggestion-icon">{suggestion_icon}</div>
            <div class="suggestion-text">{suggestion_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Review Text": user_input[:200],
            "Sentiment": sentiment_result["display_label"],
            "Sentiment Confidence": f"{sentiment_result['score']:.2%}",
            "Fake Probability": f"{fake_result['fake_prob']:.2%}",
            "Is Fake": "Yes" if fake_result["is_fake"] else "No"
        }
        
        if fake_result["is_fake"]:
            st.session_state.fake_reviews_history.append(record)
        if not sentiment_result["is_positive"]:
            st.session_state.negative_reviews_history.append(record)
    
    elif analyze_btn:
        st.warning("Please enter a review to analyze.")
    
    st.markdown('<div class="history-section">', unsafe_allow_html=True)
    st.markdown('<div class="history-title">Session History</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        render_history_card("Fake Reviews", "⚠️", 
                           st.session_state.fake_reviews_history, "fake")
    with col2:
        render_history_card("Negative Reviews", "😞",
                           st.session_state.negative_reviews_history, "negative")
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="export-section">
        <div class="export-stats">
            <div class="export-stat">
                <div class="export-stat-value">{len(st.session_state.fake_reviews_history)}</div>
                <div class="export-stat-label">Fake Reviews</div>
            </div>
            <div class="export-stat">
                <div class="export-stat-value">{len(st.session_state.negative_reviews_history)}</div>
                <div class="export-stat-label">Negative Reviews</div>
            </div>
        </div>
        <div class="export-actions">
    """, unsafe_allow_html=True)
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        if st.download_button(
            label="📥 Export & Clear",
            data=get_excel_buffer(st.session_state.fake_reviews_history, 
                                 st.session_state.negative_reviews_history),
            file_name=f"review_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="export_clear_btn"
        ):
            st.session_state.export_and_clear_flag = True
            st.rerun()
    
    with col_exp2:
        st.download_button(
            label="📥 Export Only",
            data=get_excel_buffer(st.session_state.fake_reviews_history, 
                                 st.session_state.negative_reviews_history),
            file_name=f"review_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="export_only_btn"
        )
    
    with col_exp3:
        if st.button("🗑️ Clear History", use_container_width=True, key="clear_history_btn"):
            st.session_state.clear_history_flag = True
            st.rerun()


if __name__ == "__main__":
    main()
