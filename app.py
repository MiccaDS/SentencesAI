import streamlit as st
from litellm import completion
import os
import json
import re
import pandas as pd

st.set_page_config(page_title="SentencesAI", page_icon="🃏", layout="wide")

st.title("🃏 SentencesAI")
st.caption("Create Anki & Quizlet-style flashcards from any text")

# Load API key
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

if not HUGGINGFACE_API_KEY:
    st.error("❌ Hugging Face API key not found!")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox("Model", [
        "huggingface/Qwen/Qwen2.5-7B-Instruct",
        "huggingface/meta-llama/Llama-3.2-3B-Instruct"
    ], index=0)
    num_cards = st.slider("Number of cards", 4, 20, 8)
    style = st.selectbox("Style", ["Mixed", "Vocabulary", "Cloze", "Q&A", "Sentence"], index=0)

text = st.text_area("Paste your text here", height=200, placeholder="Paste paragraph, notes, or topic...")

if st.button("🚀 Generate Flashcards", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text!")
        st.stop()

    with st.spinner("Generating flashcards..."):
        try:
            prompt = f"""Create exactly {num_cards} flashcards from the text below.

Style: {style}

Return ONLY a valid JSON array like this:
[
  {{"front": "Question or prompt", "back": "Answer or explanation", "type": "Vocabulary"}}
]

Text:
{text}"""

            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_key=HUGGINGFACE_API_KEY,
                temperature=0.7,
                max_tokens=3000
            )

            content = response.choices[0].message.content.strip()
            match = re.search(r'\[.*\]', content, re.DOTALL)
            json_text = match.group(0) if match else content

            flashcards = json.loads(json_text)
            st.session_state.flashcards = flashcards
            st.session_state.current_index = 0
            st.success(f"✅ {len(flashcards)} flashcards created!")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ====================== INTERACTIVE FLASHCARD VIEWER ======================
if "flashcards" in st.session_state and st.session_state.flashcards:
    flashcards = st.session_state.flashcards
    index = st.session_state.get("current_index", 0)

    st.subheader(f"Flashcard {index + 1} of {len(flashcards)}")

    # Card display
    card = flashcards[index]
    
    with st.container(border=True):
        st.markdown(f"### **{card['front']}**")
        
        if st.button("🔄 Flip Card", use_container_width=True):
            if "show_back" not in st.session_state:
                st.session_state.show_back = False
            st.session_state.show_back = not st.session_state.show_back

        if st.session_state.get("show_back", False):
            st.markdown("---")
            st.markdown(f"**{card['back']}**")
            st.caption(f"Type: {card.get('type', 'General')}")

    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Previous", disabled=(index == 0)):
            st.session_state.current_index -= 1
            st.session_state.show_back = False
            st.rerun()
    with col3:
        if st.button("Next ➡️", disabled=(index == len(flashcards)-1)):
            st.session_state.current_index += 1
            st.session_state.show_back = False
            st.rerun()

    # Progress bar
    st.progress((index + 1) / len(flashcards))

    # Download buttons
    df = pd.DataFrame(flashcards)
    colA, colB = st.columns(2)
    with colA:
        st.download_button("📥 Download CSV", df.to_csv(index=False), "flashcards.csv", mime="text/csv")
    with colB:
        st.download_button("📥 Download JSON", json.dumps(flashcards, indent=2), "flashcards.json", mime="application/json")

else:
    st.info("Generate flashcards to start studying!")
