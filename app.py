"""
JEE Sensei - Multimodal Input Application
ChatGPT-Style UI Redesign
"""

import streamlit as st
import os
import re
import tempfile
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import custom modules
from ocr.ocr_engine import OCREngine
from asr.whisper_engine import WhisperEngine
from utils.confidence import (
    format_confidence_percentage,
    get_confidence_color
)
from agents.orchestrator import MathMentorOrchestrator


# Page configuration
st.set_page_config(
    page_title="JEE Sensei",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CHATGPT-STYLE CSS
# ============================================================================
def get_chatgpt_css():
    return """
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding: 1rem 2rem 6rem 2rem;
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* App header */
    .app-header {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1.5rem;
    }
    
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .app-subtitle {
        color: #8b8b9e;
        font-size: 1rem;
        font-weight: 400;
    }
    
    /* Chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem 0;
        min-height: 400px;
    }
    
    /* Message bubbles */
    .message-row {
        display: flex;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    
    .message-row.user {
        justify-content: flex-end;
    }
    
    .message-row.assistant {
        justify-content: flex-start;
    }
    
    .message-bubble {
        max-width: 80%;
        padding: 1rem 1.25rem;
        border-radius: 18px;
        font-size: 0.95rem;
        line-height: 1.6;
        word-wrap: break-word;
    }
    
    .message-bubble.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        border-bottom-right-radius: 4px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .message-bubble.assistant {
        background: linear-gradient(145deg, #2d2d44 0%, #252538 100%);
        color: #e4e4e7;
        border-bottom-left-radius: 4px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .message-label {
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        opacity: 0.7;
    }
    
    .message-label.user {
        color: #a5b4fc;
        text-align: right;
    }
    
    .message-label.assistant {
        color: #94a3b8;
    }
    
    /* Input area - fixed at bottom */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(180deg, transparent 0%, #0f0f23 20%);
        padding: 1.5rem 2rem 2rem 2rem;
        z-index: 1000;
    }
    
    .input-wrapper {
        max-width: 900px;
        margin: 0 auto;
        display: flex;
        gap: 0.75rem;
        align-items: flex-end;
    }
    
    /* Text input styling */
    .stTextArea textarea, .stTextInput input {
        background: #2d2d44 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 16px !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        padding: 1rem 1.25rem !important;
        resize: none !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
        outline: none !important;
    }
    
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: #6b7280 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.875rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.35) !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
    }
    
    .stButton > button p, .stButton > button span, .stButton > button div {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* Form button styling */
    button[kind="primary"], button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    button[kind="primary"] p, button[data-testid="baseButton-primary"] p {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* Secondary/feedback buttons */
    .feedback-btn {
        background: #2d2d44 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e4e4e7 !important;
    }
    
    .feedback-btn:hover {
        background: #3d3d54 !important;
        border-color: #667eea !important;
    }
    
    /* Solution card */
    .solution-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #252538 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .solution-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    
    .solution-icon {
        font-size: 1.5rem;
    }
    
    .solution-title {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .solution-content {
        color: #d1d5db;
        line-height: 1.8;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #9ca3af !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #2d2d44 !important;
        color: #e4e4e7 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }
    
    .streamlit-expanderContent {
        background: #1e1e2e !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        color: #d1d5db !important;
    }
    
    /* Success/Warning/Error messages */
    .stSuccess {
        background: linear-gradient(145deg, #064e3b 0%, #065f46 100%) !important;
        color: #a7f3d0 !important;
        border: 1px solid #10b981 !important;
        border-radius: 12px !important;
    }
    
    .stWarning {
        background: linear-gradient(145deg, #78350f 0%, #92400e 100%) !important;
        color: #fde68a !important;
        border: 1px solid #f59e0b !important;
        border-radius: 12px !important;
    }
    
    .stError {
        background: linear-gradient(145deg, #7f1d1d 0%, #991b1b 100%) !important;
        color: #fecaca !important;
        border: 1px solid #ef4444 !important;
        border-radius: 12px !important;
    }
    
    .stInfo {
        background: linear-gradient(145deg, #1e3a5f 0%, #1e40af 100%) !important;
        color: #bfdbfe !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 12px !important;
    }
    
    /* Divider */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #2d2d44 !important;
        border: 2px dashed rgba(255,255,255,0.2) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #e4e4e7 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e4e4e7 !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #e4e4e7 !important;
    }
    
    .stRadio [data-testid="stMarkdownContainer"] {
        color: #e4e4e7 !important;
    }
    
    /* Select slider */
    .stSelectSlider label {
        color: #e4e4e7 !important;
    }
    
    /* Form styling */
    [data-testid="stForm"] {
        background: linear-gradient(145deg, #2d2d44 0%, #252538 100%) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
    }
    
    /* Caption text */
    .stCaption, caption {
        color: #8b8b9e !important;
    }
    
    /* Code blocks */
    code {
        background: #1e1e2e !important;
        color: #a5b4fc !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 6px !important;
        font-family: 'Fira Code', monospace !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Labels */
    .stTextArea label, .stTextInput label, .stSelectbox label {
        color: #e4e4e7 !important;
        font-weight: 500 !important;
    }
    
    /* Markdown text */
    .stMarkdown, .stMarkdown p {
        color: #e4e4e7 !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #ffffff !important;
    }
    
    /* Audio player */
    audio {
        border-radius: 12px !important;
        background: #2d2d44 !important;
    }
    
    /* Image styling */
    .stImage {
        border-radius: 16px !important;
        overflow: hidden !important;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-badge.success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
    }
    
    .status-badge.warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #ffffff;
    }
    
    .status-badge.error {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #ffffff;
    }
    
    /* Welcome message */
    .welcome-card {
        background: linear-gradient(145deg, #2d2d44 0%, #252538 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .welcome-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .welcome-title {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .welcome-text {
        color: #9ca3af;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Feedback section */
    .feedback-section {
        background: linear-gradient(145deg, #2d2d44 0%, #252538 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .feedback-title {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Quick action buttons */
    .quick-actions {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    
    .quick-action-btn {
        background: #2d2d44;
        border: 1px solid rgba(255,255,255,0.1);
        color: #e4e4e7;
        padding: 0.75rem 1.25rem;
        border-radius: 12px;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .quick-action-btn:hover {
        background: #3d3d54;
        border-color: #667eea;
        transform: translateY(-2px);
    }
    </style>
    """


# ============================================================================
# CHAT MESSAGE COMPONENTS
# ============================================================================
def render_user_message(message: str):
    """Render a user message bubble."""
    st.markdown(f"""
    <div class="message-row user">
        <div style="text-align: right;">
            <div class="message-label user">👤 You</div>
            <div class="message-bubble user">{message}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_assistant_message(message: str, show_label: bool = True):
    """Render an assistant message bubble."""
    import re
    
    # HTML template for fraction-style derivative: d/dx or dA/dx
    def make_derivative_html(func, var):
        if func:
            return f'<span style="display:inline-flex;flex-direction:column;vertical-align:middle;text-align:center;font-style:italic;line-height:1;"><span style="border-bottom:1px solid currentColor;padding:0 2px;">𝑑{func}</span><span style="padding:0 2px;">𝑑{var}</span></span>'
        else:
            return f'<span style="display:inline-flex;flex-direction:column;vertical-align:middle;text-align:center;font-style:italic;line-height:1;"><span style="border-bottom:1px solid currentColor;padding:0 2px;">𝑑</span><span style="padding:0 2px;">𝑑{var}</span></span>'
    
    # HTML template for integral symbol
    def make_integral_html():
        return '<span style="font-size:1.3em;font-style:italic;">∫</span>'
    
    html_message = message
    
    # Remove backticks (code formatting) - show math directly
    html_message = re.sub(r'`([^`]+)`', r'\1', html_message)
    
    # Format derivative notation: derivative of A with respect to x
    # Match complex patterns: d(expression)/dx - handle nested parens by looking ahead for /d
    # Regex explanation: d followed by anything in parens (balanced or not, simple greedy) until we hit /d[var]
    html_message = re.sub(r'\bd\((.+?)\)/d([a-zA-Z])\b', lambda m: make_derivative_html(f"({m.group(1)})", m.group(2)), html_message)
    
    # Match patterns like dA/dx, dV/dr, dy/dx
    html_message = re.sub(r'\bd([A-Za-z])/d([a-zA-Z])\b', lambda m: make_derivative_html(m.group(1), m.group(2)), html_message)
    
    # Also match simple d/dx pattern
    html_message = re.sub(r'\bd/d([a-zA-Z])\b', lambda m: make_derivative_html('', m.group(1)), html_message)
    
    # Format integral symbol: integrate, integral, ∫ -> proper integral symbol
    html_message = html_message.replace('∫', make_integral_html())
    
    # Math formatting: x**2 -> x<sup>2</sup>, x^3 -> x<sup>3</sup>, e^(3x) -> e<sup>3x</sup>
    # Handle parens: x^(n-1) -> x<sup>n-1</sup>
    # Handle parenthesized base: (a+b)^2 -> (a+b)<sup>2</sup>
    html_message = re.sub(r'(\))\^([-+]?[a-zA-Z0-9]+)', r'\1<sup>\2</sup>', html_message)
    html_message = re.sub(r'([a-zA-Z0-9])\^([-+]?[a-zA-Z0-9]+)', r'\1<sup>\2</sup>', html_message)
    html_message = re.sub(r'([a-zA-Z0-9])\^\(([^)]+)\)', r'\1<sup>\2</sup>', html_message)
    
    # Replace multiplication signs with symbol: 4*p -> 4 × p, a*b -> a × b
    # Use Regex to avoid breaking **bold** markdown!
    # Look for * that is NOT part of **
    html_message = re.sub(r'(?<!\*)\*(?!\*)', ' × ', html_message)
    
    # Remove extra spaces around the symbol if needed
    html_message = re.sub(r'\s+×\s+', ' × ', html_message)
    
    # Replace sqrt with √
    html_message = html_message.replace('sqrt(', '√(')
    
    # Clean up * between coefficient and variable in expressions like "= 4 * p"
    html_message = re.sub(r'=\s*(\d+)\s*\*\s*([a-zA-Z])', r'= \1\2', html_message)
    
    # Remove [Source: ...] citations from display
    html_message = re.sub(r'\s*\[Source:\s*[^\]]+\]', '', html_message)
    
    # Convert markdown to HTML
    # Bold: **text** -> <strong>text</strong>
    html_message = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', html_message)
    
    # Line breaks
    html_message = html_message.replace('\n\n', '<br><br>').replace('\n', '<br>')
    
    label_html = '<div class="message-label assistant">🧮 JEE Sensei</div>' if show_label else ''
    st.markdown(f"""
    <div class="message-row assistant">
        <div>
            {label_html}
            <div class="message-bubble assistant">{html_message}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_solution_card(result: dict):
    """Render a solution card with results."""
    confidence = result.get('confidence', 0)
    confidence_pct = f"{confidence:.0%}"
    
    status_class = "success" if result.get('is_correct') else "warning"
    status_text = "Verified ✓" if result.get('is_correct') else "Check Answer"
    
    explanation = result.get('explanation', 'No explanation available.').replace('\n', '<br>')
    
    st.markdown(f"""
    <div class="solution-card">
        <div class="solution-header">
            <span class="solution-icon">💡</span>
            <span class="solution-title">Solution</span>
            <span class="status-badge {status_class}" style="margin-left: auto;">
                {status_text}
            </span>
        </div>
        <div class="solution-content">
            {explanation}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_welcome_card():
    """Render welcome message when no conversation yet."""
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">🧮</div>
        <div class="welcome-title">Welcome to JEE Sensei</div>
        <div class="welcome-text">
            Smarter than your calculator. Faster than your notes.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# CACHED RESOURCES
# ============================================================================
@st.cache_resource
def load_ocr_engine():
    """Load OCR engine (cached to avoid reloading)."""
    try:
        return OCREngine(use_gpu=False, lang='en')
    except Exception as e:
        st.error(f"Failed to load OCR engine: {str(e)}")
        return None


@st.cache_resource
def load_whisper_engine():
    """Load Whisper ASR engine (cached to avoid reloading)."""
    try:
        return WhisperEngine(model_size="base", device="cpu")
    except Exception as e:
        st.error(f"Failed to load Whisper engine: {str(e)}")
        return None


def process_image(image_file, ocr_engine):
    """Process uploaded image with OCR."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(image_file.getvalue())
            tmp_path = tmp_file.name
        
        text, confidence = ocr_engine.extract_text(tmp_path)
        os.unlink(tmp_path)
        
        return text, confidence
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return "", 0.0


def process_audio(audio_file, whisper_engine):
    """Process uploaded audio with ASR."""
    try:
        file_extension = Path(audio_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
        
        text, confidence = whisper_engine.transcribe(tmp_path)
        os.unlink(tmp_path)
        
        return text, confidence
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return "", 0.0


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application function."""
    
    # Apply ChatGPT-style CSS
    st.markdown(get_chatgpt_css(), unsafe_allow_html=True)
    
    # App Header
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🧮 JEE Sensei</div>
        <div class="app-subtitle">Smarter than your calculator. Faster than your notes.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    if 'confidence' not in st.session_state:
        st.session_state.confidence = 0.0
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0  # Used to clear input field
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")
        
        st.markdown("##### 📥 Input Mode")
        input_mode = st.radio(
            "Choose input method:",
            ["📝 Text", "🖼️ Image (OCR)", "🎤 Audio (ASR)"],
            index=0,
            key="sidebar_input_mode"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Capabilities")
        st.markdown("""
        ✅ Square roots & radicals  
        ✅ Algebra & equations  
        ✅ Calculus & derivatives  
        ✅ Probability  
        ✅ Word problems  
        ✅ System of equations
        """)
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_result = None
            st.session_state.extracted_text = ""
            st.session_state.processing_complete = False
            st.rerun()
    
    # Main chat area
    if not st.session_state.chat_history:
        render_welcome_card()
    
    # Display chat history
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            render_user_message(msg['content'])
        else:
            render_assistant_message(msg['content'])
    
    st.markdown("---")
    
    # Initialize additional session states for HITL
    if 'pending_input' not in st.session_state:
        st.session_state.pending_input = None
    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = "📝 Text"
    
    # ========== INPUT AREA WITH MODE DROPDOWN ==========
    # Mode dropdown and input in main area
    col_mode, col_input, col_send = st.columns([1.5, 5, 1])
    
    with col_mode:
        input_mode = st.selectbox(
            "Mode",
            ["📝 Text", "🖼️ Image", "🎤 Audio"],
            index=0,
            label_visibility="collapsed",
            key="main_input_mode"
        )
    
    with col_input:
        if input_mode == "📝 Text":
            user_input = st.text_area(
                "Message",
                height=70,
                placeholder="Ask a math question... (e.g., What is the square root of 144?)",
                label_visibility="collapsed",
                key=f"text_input_{st.session_state.input_key}"
            )
        elif input_mode == "🖼️ Image":
            uploaded_file = st.file_uploader(
                "Upload image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                key="image_uploader"
            )
            user_input = None
        else:  # Audio
            uploaded_file = st.file_uploader(
                "Upload audio",
                type=["wav", "mp3", "m4a", "ogg"],
                label_visibility="collapsed",
                key="audio_uploader"
            )
            user_input = None
    
    with col_send:
        st.markdown("<br>", unsafe_allow_html=True)
        if input_mode == "📝 Text":
            submit_btn = st.button("➤", type="primary", use_container_width=True, help="Submit for review")
        else:
            submit_btn = st.button("🔍", type="primary", use_container_width=True, help="Process file")
    
    # ========== HANDLE FILE UPLOADS (OCR/ASR) ==========
    if input_mode == "🖼️ Image" and submit_btn and 'uploaded_file' in dir() and uploaded_file is not None:
        with st.spinner("📷 Extracting text with OCR..."):
            ocr_engine = load_ocr_engine()
            if ocr_engine:
                text, confidence = process_image(uploaded_file, ocr_engine)
                if text:
                    st.session_state.pending_input = text
                    st.session_state.confidence = confidence
                    st.rerun()
                else:
                    st.warning("No text detected in the image.")
    
    if input_mode == "🎤 Audio" and submit_btn and 'uploaded_file' in dir() and uploaded_file is not None:
        with st.spinner("🎧 Transcribing audio..."):
            whisper_engine = load_whisper_engine()
            if whisper_engine:
                text, confidence = process_audio(uploaded_file, whisper_engine)
                if text:
                    st.session_state.pending_input = text
                    st.session_state.confidence = confidence
                    st.rerun()
                else:
                    st.warning("No speech detected in the audio.")
    
    # ========== HANDLE TEXT INPUT ==========
    if input_mode == "📝 Text" and submit_btn and user_input and user_input.strip():
        st.session_state.pending_input = user_input.strip()
        st.session_state.confidence = 1.0
        st.rerun()
    
    # ========== HUMAN-IN-THE-LOOP CONFIRMATION ==========
    if st.session_state.pending_input:
        st.markdown("---")
        st.markdown("### ✏️ Review & Edit Before Sending")
        
        # Show confidence if from OCR/ASR
        if st.session_state.confidence < 1.0:
            conf_pct = f"{st.session_state.confidence:.0%}"
            st.caption(f"Extraction confidence: {conf_pct}")
        
        # Editable text area
        edited_input = st.text_area(
            "Edit your question if needed:",
            value=st.session_state.pending_input,
            height=120,
            key="edit_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("✅ Confirm & Send", type="primary", use_container_width=True):
                # Add to chat history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': edited_input
                })
                
                # Process with orchestrator
                with st.spinner("🔍 Analyzing your problem..."):
                    try:
                        orchestrator = MathMentorOrchestrator()
                        result = orchestrator.solve_problem(edited_input)
                        
                        st.session_state.current_result = result
                        st.session_state.extracted_text = edited_input
                        st.session_state.processing_complete = True
                        
                        if result['status'] == 'success':
                            answer_text = result.get('answer', 'See explanation')
                            explanation_text = result.get('explanation', '')
                            
                            # Iteratively remove answer text from start (handle multiple repetitions)
                            clean_loop = True
                            while clean_loop:
                                original_len = len(explanation_text)
                                # Remove "Answer:" header if it reappears
                                explanation_text = re.sub(r'^\s*(\*\*|#+\s*)?Answer\s*:?\s*', '', explanation_text, flags=re.IGNORECASE).strip()
                                
                                if answer_text and answer_text.lower() != "see explanation":
                                    ans_pattern = re.escape(answer_text)
                                    ans_pattern = ans_pattern.replace(r'\ ', r'\s*')
                                    explanation_text = re.sub(r'^' + ans_pattern + r'\s*', '', explanation_text, flags=re.IGNORECASE).strip()
                                
                                if len(explanation_text) == original_len:
                                    clean_loop = False

                            # If answer is just "See explanation below", don't double print
                            if "see explanation" in answer_text.lower():
                                response = f"**Answer**:\n\n{explanation_text}"
                            else:
                                response = f"**Answer**: {answer_text}\n\n{explanation_text}"
                        else:
                            response = result.get('explanation', 'I could not solve this problem. Please try rephrasing.')
                        
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response
                        })
                        
                    except Exception as e:
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': f"Sorry, an error occurred: {str(e)}"
                        })
                
                # Clear pending input and increment key to clear text area
                st.session_state.pending_input = None
                st.session_state.input_key += 1  # Rotate key to clear text input
                st.rerun()
        
        with col2:
            if st.button("🗑️ Cancel", use_container_width=True):
                st.session_state.pending_input = None
                st.rerun()
        
        with col3:
            pass  # Empty column for spacing
    
    # Show solution details if available
    if st.session_state.current_result and st.session_state.current_result.get('status') == 'success':
        result = st.session_state.current_result
        
        with st.expander("📊 Solution Details", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{result.get('confidence', 0):.0%}")
            with col2:
                st.metric("Difficulty", result.get('difficulty', 'medium').title())
            with col3:
                if result.get('is_correct'):
                    st.success("✅ Verified")
                else:
                    st.warning("⚠️ Unverified")
        
        if result.get('sources'):
            with st.expander("📚 Knowledge Sources", expanded=False):
                for source in set(result['sources']):
                    st.markdown(f"• {source}")
        
        if result.get('trace'):
            with st.expander("🔧 Agent Trace", expanded=False):
                for step in result['trace']:
                    st.markdown(f"**{step.get('agent', 'Agent')}**")
                    st.json(step.get('output', {}))
        
        # Feedback section
        st.markdown("---")
        st.markdown("### 📝 Was this solution correct?")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("✅ Correct", use_container_width=True):
                try:
                    orchestrator = MathMentorOrchestrator()
                    problem_id = orchestrator.store_solved_problem(
                        st.session_state.extracted_text,
                        result
                    )
                    orchestrator.submit_feedback(problem_id, is_correct=True)
                    st.success("Thanks for the feedback!")
                    st.balloons()
                except:
                    st.success("Thanks for the feedback!")
        
        with col2:
            if st.button("❌ Incorrect", use_container_width=True):
                st.session_state.show_correction = True
        
        # Correction form
        if st.session_state.get('show_correction', False):
            with st.form("correction_form", clear_on_submit=True):
                st.markdown("##### 💡 Provide the correct answer")
                correction = st.text_area(
                    "Correct solution:",
                    placeholder="Enter the correct answer..."
                )
                
                confidence_level = st.select_slider(
                    "Your confidence:",
                    options=["Not sure", "Somewhat confident", "Very confident", "Certain"],
                    value="Somewhat confident"
                )
                
                if st.form_submit_button("Submit Correction", type="primary"):
                    if correction.strip():
                        try:
                            orchestrator = MathMentorOrchestrator()
                            problem_data = {
                                "problem_text": st.session_state.extracted_text,
                                "answer": correction,
                                "topic": "user_contributed",
                                "intent": "user_answer",
                                "solution_steps": correction,
                                "confidence": {"Not sure": 0.25, "Somewhat confident": 0.5, "Very confident": 0.75, "Certain": 1.0}.get(confidence_level, 0.5),
                                "is_correct": True,
                                "user_id": "default",
                                "source": "user_contribution"
                            }
                            orchestrator.memory_store.store_problem(problem_data)
                            st.success("🎉 Thank you! Your answer has been stored.")
                            st.session_state.show_correction = False
                        except:
                            st.success("Thank you for your contribution!")
                    else:
                        st.warning("Please enter a correction.")
    
    # Handle error states with feedback
    elif st.session_state.current_result and st.session_state.current_result.get('status') in ['insufficient_knowledge', 'needs_clarification', 'error']:
        result = st.session_state.current_result
        
        st.markdown("---")
        st.markdown("### 📝 Help improve our knowledge base")
        st.caption("If you know the answer, please share it below.")
        
        with st.form("knowledge_contribution", clear_on_submit=True):
            user_answer = st.text_area(
                "Your answer:",
                placeholder="Enter the correct solution...",
                height=120
            )
            
            confidence_level = st.select_slider(
                "Confidence level:",
                options=["Not sure", "Somewhat confident", "Very confident", "Certain"],
                value="Somewhat confident"
            )
            
            if st.form_submit_button("📤 Submit", type="primary"):
                if user_answer.strip():
                    try:
                        orchestrator = MathMentorOrchestrator()
                        problem_data = {
                            "problem_text": st.session_state.extracted_text,
                            "answer": user_answer,
                            "topic": "user_contributed",
                            "intent": "user_answer",
                            "solution_steps": user_answer,
                            "confidence": {"Not sure": 0.25, "Somewhat confident": 0.5, "Very confident": 0.75, "Certain": 1.0}.get(confidence_level, 0.5),
                            "is_correct": True,
                            "user_id": "default"
                        }
                        orchestrator.memory_store.store_problem(problem_data)
                        st.success("🎉 Thank you! Your contribution will help future users.")
                        st.balloons()
                    except:
                        st.success("Thank you for your contribution!")
                else:
                    st.warning("Please enter an answer.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.85rem; padding: 1rem;">
        Built with ❤️ using <strong>Streamlit</strong>, <strong>EasyOCR</strong>, <strong>Whisper</strong>, and <strong>AI</strong><br>
        <span style="opacity: 0.7;">JEE Sensei — Making math accessible for everyone</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
