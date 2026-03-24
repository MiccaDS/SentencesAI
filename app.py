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

Return ONLY a valid JSON array. Each object must have "front", "back", and "type".

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
            
            # Initialize session state properly
            st.session_state.flashcards = flashcards
            st.session_state.current_index = 0
            st.session_state.show_back = False
            
            st.success(f"✅ {len(flashcards)} flashcards created!")
            st.rerun()

        except Exception as e:
            st.error(f"Error generating flashcards: {str(e)}")

# ====================== INTERACTIVE FLASHCARD VIEWER ======================
if "flashcards" in st.session_state and st.session_state.flashcards:
    flashcards = st.session_state.flashcards
    index = st.session_state.current_index

    st.subheader(f"Flashcard {index + 1} of {len(flashcards)}")

    card = flashcards[index]

    # Card container
    with st.container(border=True):
        st.markdown(f"### {card['front']}")

        if st.button("🔄 Flip to see answer", use_container_width=True, type="secondary"):
            st.session_state.show_back = not st.session_state.show_back

        if st.session_state.show_back:
            st.markdown("---")
            st.markdown(f"**Answer:** {card['back']}")
            if "type" in card:
                st.caption(f"Type: {card['type']}")

    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Previous", disabled=(index == 0), use_container_width=True):
            st.session_state.current_index -= 1
            st.session_state.show_back = False
            st.rerun()
    
    with col3:
        if st.button("Next ➡️", disabled=(index == len(flashcards)-1), use_container_width=True):
            st.session_state.current_index += 1
            st.session_state.show_back = False
            st.rerun()

    # Progress
    st.progress((index + 1) / len(flashcards))

    # Downloads
    df = pd.DataFrame(flashcards)
    colA, colB = st.columns(2)
    with colA:
        st.download_button("📥 Download CSV", df.to_csv(index=False), "flashcards.csv", "text/csv")
    with colB:
        st.download_button("📥 Download JSON", json.dumps(flashcards, indent=2), "flashcards.json", "application/json")

else:
    st.info("👆 Generate some flashcards to start studying!")