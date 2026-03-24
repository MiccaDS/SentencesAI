import streamlit as st
import os
import json
import re
import pandas as pd


st.set_page_config(page_title="SentencesAI", page_icon="🃏", layout="wide")

st.title("🃏 SentencesAI")
st.caption("Create Anki/Quizlet flashcards from any text using free Hugging Face AI")

# Updated to support BOTH .env and secrets.toml with your HUGGINGFACE_API name
def get_hf_token():
    # 1. Streamlit secrets (Cloud / HF Spaces)
    if "HUGGINGFACE_API" in st.secrets:
        return st.secrets["HUGGINGFACE_API"]
    if "HUGGINGFACE" in st.secrets:
        return st.secrets["HUGGINGFACE"].get("token") or st.secrets["HUGGINGFACE"].get("HUGGINGFACE_TOKEN")
    
    # 2. Environment variable (.env file)
    return (os.getenv("HUGGINGFACE_API") 
            or os.getenv("HUGGINGFACE_TOKEN") 
            or os.getenv("HF_TOKEN"))

hf_token = get_hf_token()

if not hf_token:
    st.error("No Hugging Face token found!")
    st.info("""Add this to your .env file:
HUGGINGFACE_API=hf_your_actual_token_here""")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct"
    ], index=0)
    num_cards = st.slider("Number of cards", 4, 20, 8)
    style = st.selectbox("Style", ["Mixed", "Vocabulary", "Cloze", "Q&A", "Sentence"], index=0)

text = st.text_area("Paste your text or topic here", height=220, placeholder="Paste any paragraph, notes, or topic...")

if st.button("Generate Flashcards", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text")
        st.stop()
    
    client = InferenceClient(token=hf_token, model=model)
    
    with st.spinner("Generating flashcards..."):
        prompt = f"""Create exactly {num_cards} flashcards.
Style: {style}
Return ONLY JSON array with 'front', 'back', and 'type' keys.

Text: {text}"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        match = re.search(r'\[.*\]', content, re.DOTALL)
        json_text = match.group(0) if match else content
        
        flashcards = json.loads(json_text)
        
        st.session_state['flashcards'] = flashcards
        st.success(f"✅ Created {len(flashcards)} flashcards!")

# Show results
if 'flashcards' in st.session_state:
    df = pd.DataFrame(st.session_state['flashcards'])
    st.subheader("Your Flashcards")
    st.data_editor(df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download CSV for Anki",
            data=df.to_csv(index=False),
            file_name="sentencesai_deck.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            label="Download JSON",
            data=json.dumps(st.session_state['flashcards'], indent=2),
            file_name="sentencesai_deck.json",
            mime="application/json"
        )