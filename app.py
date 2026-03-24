import streamlit as st
from litellm import completion
import os
import json
import re
import pandas as pd

st.set_page_config(page_title="SentencesAI", page_icon="🃏", layout="wide")

st.title("🃏 SentencesAI")
st.caption("Create Anki/Quizlet flashcards from any text using free Hugging Face AI")

# ====================== TOKEN LOADING ======================
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

if not HUGGINGFACE_API_KEY:
    st.error("❌ No Hugging Face API key found!")
    st.info("""Please add this line to your `.env` file:

HUGGINGFACE_API=hf_your_actual_token_here""")
    st.stop()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", [
        "huggingface/Qwen/Qwen2.5-7B-Instruct",
        "huggingface/meta-llama/Llama-3.2-3B-Instruct"
    ], index=0)
    
    num_cards = st.slider("Number of cards", 4, 20, 8)
    style = st.selectbox("Style", ["Mixed", "Vocabulary", "Cloze", "Q&A", "Sentence"], index=0)

# ====================== MAIN AREA ======================
text = st.text_area("Paste your text or topic here", height=220, 
                    placeholder="Paste any paragraph, notes, article, or topic...")

if st.button("Generate Flashcards", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text")
        st.stop()

    with st.spinner("Generating flashcards... (this may take 15-40 seconds)"):
        try:
            prompt = f"""Create exactly {num_cards} high-quality flashcards from the text below.

Style: {style}

Return ONLY a valid JSON array. No extra text, no markdown, no explanation.

Each card must have these exact keys:
- "front": question or prompt
- "back": answer or explanation  
- "type": one of "Vocabulary", "Cloze", "Q&A", "Sentence", "Definition"

Text:
{text}"""

            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_key=HUGGINGFACE_API_KEY,
                temperature=0.7,
                max_tokens=2500
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON if the model adds extra text
            match = re.search(r'\[.*\]', content, re.DOTALL)
            json_text = match.group(0) if match else content

            flashcards = json.loads(json_text)

            st.session_state['flashcards'] = flashcards
            st.success(f"✅ Created {len(flashcards)} flashcards!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Tip: Try shorter text or switch to the other model.")

# ====================== DISPLAY RESULTS ======================
if 'flashcards' in st.session_state:
    df = pd.DataFrame(st.session_state['flashcards'])
    
    st.subheader("📇 Your Flashcards")
    st.data_editor(df, use_container_width=True, num_rows="dynamic")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download CSV for Anki/Quizlet",
            data=df.to_csv(index=False),
            file_name="sentencesai_deck.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(st.session_state['flashcards'], indent=2),
            file_name="sentencesai_deck.json",
            mime="application/json"
        )