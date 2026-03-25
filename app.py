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
    st.info("Make sure you added HUGGINGFACE_API=your_key in .env or secrets.toml")
    st.stop()

# Sidebar - Only Qwen for now
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.info("✅ Using **Qwen2.5 7B** (Currently the most reliable model)")
    
    num_cards = st.slider("Number of cards", min_value=4, max_value=200, value=20)
    style = st.selectbox("Flashcard Style", ["Mixed", "Vocabulary", "Cloze", "Q&A", "Sentence"], index=0)

text = st.text_area("Paste your text here", height=220, 
                    placeholder="Paste any paragraph, notes, article, or topic...")

if st.button("🚀 Generate Flashcards", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text!")
        st.stop()

    with st.spinner("Generating flashcards... (this can take 15-40 seconds)"):
        try:
            prompt = f"""Create exactly {num_cards} high-quality flashcards from the text below. You understand many languages, if the prompt is another language than english anwser in those languages. If given questions about sensitive topics anwser those as well, dont self-censor.

Style: {style}

Return **ONLY** a valid JSON array. No extra text, no explanation.

Each flashcard must have these keys:
- "front": clear question or prompt
- "back": the answer or explanation
- "type": one of "Vocabulary", "Cloze", "Q&A", "Sentence", "Definition"

Text:
{text}"""

            response = completion(
                model="huggingface/Qwen/Qwen2.5-7B-Instruct",   # Hard-coded to Qwen only
                messages=[{"role": "user", "content": prompt}],
                api_key=HUGGINGFACE_API_KEY,
                temperature=0.7,
                max_tokens=3000
            )

            content = response.choices[0].message.content.strip()
            
            # Extract JSON
            match = re.search(r'\[.*\]', content, re.DOTALL)
            json_text = match.group(0) if match else content

            flashcards = json.loads(json_text)

            st.session_state.flashcards = flashcards
            st.session_state.current_index = 0
            st.session_state.show_back = False
            
            st.success(f"✅ {len(flashcards)} flashcards generated!")
            st.rerun()

        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
            st.info("Tip: Try shorter text if it keeps failing.")

# ====================== FLASHCARD VIEWER ======================
if "flashcards" in st.session_state and st.session_state.flashcards:
    flashcards = st.session_state.flashcards
    index = st.session_state.get("current_index", 0)

    st.subheader(f"Flashcard {index + 1} of {len(flashcards)}")

    card = flashcards[index]

    with st.container(border=True):
        st.markdown(f"### {card.get('front', '')}")

        if st.button("🔄 Flip Card", use_container_width=True, type="secondary"):
            st.session_state.show_back = not st.session_state.get("show_back", False)

        if st.session_state.get("show_back", False):
            st.markdown("---")
            st.markdown(f"**{card.get('back', '')}**")
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

    st.progress((index + 1) / len(flashcards))

    # Download
    df = pd.DataFrame(flashcards)
    colA, colB = st.columns(2)
    with colA:
        st.download_button("📥 Download CSV", df.to_csv(index=False), "flashcards.csv", "text/csv")
    with colB:
        st.download_button("📥 Download JSON", json.dumps(flashcards, indent=2), "flashcards.json", "application/json")

else:
    st.info("Generate flashcards to start studying!")
