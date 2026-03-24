import streamlit as st
from litellm import completion
import os
import json
import re
import pandas as pd

st.set_page_config(page_title="SentencesAI", page_icon="🃏", layout="wide")

st.title("🃏 SentencesAI")
st.caption("Create Anki & Quizlet flashcards from any text")

# Load API key
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

if not HUGGINGFACE_API_KEY:
    st.error("❌ Hugging Face API key not found!")
    st.info("Add `HUGGINGFACE_API=your_key_here` to your `.env` file (local) or `secrets.toml` (on Streamlit Cloud)")
    st.stop()

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox(
        "Choose Model",
        [
            "huggingface/Qwen/Qwen2.5-7B-Instruct",
            "huggingface/meta-llama/Llama-3.2-3B-Instruct"
        ],
        index=0
    )
    num_cards = st.slider("Number of flashcards", min_value=4, max_value=20, value=8)
    style = st.selectbox("Flashcard Style", ["Mixed", "Vocabulary", "Cloze", "Q&A", "Sentence"], index=0)

# Main input
text = st.text_area(
    "Paste your text here",
    height=250,
    placeholder="Paste any text, paragraph, article, lecture notes, or even just a topic..."
)

if st.button("🚀 Generate Flashcards", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please paste some text first!")
        st.stop()

    with st.spinner("Generating flashcards... This can take 15–40 seconds"):
        try:
            prompt = f"""Create exactly {num_cards} high-quality flashcards based on the following text.

Style: {style}

Return **only** a valid JSON array. No extra text, no explanation, no markdown.

Each flashcard must be a JSON object with these exact keys:
- "front": the question or prompt (clear and concise)
- "back": the answer or explanation
- "type": one of "Vocabulary", "Cloze", "Q&A", "Sentence", "Definition"

Text to create flashcards from:
{text}"""

            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_key=HUGGINGFACE_API_KEY,
                temperature=0.7,
                max_tokens=3000
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON array
            match = re.search(r'\[.*\]', content, re.DOTALL | re.IGNORECASE)
            json_text = match.group(0) if match else content

            flashcards = json.loads(json_text)

            # Save to session state
            st.session_state.flashcards = flashcards

            st.success(f"✅ Successfully created {len(flashcards)} flashcards!")

        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
            st.info("Tip: Try shorter text or switch to the other model.")

# Show flashcards if generated
if "flashcards" in st.session_state:
    df = pd.DataFrame(st.session_state.flashcards)

    st.subheader("📇 Generated Flashcards")
    st.data_editor(df, use_container_width=True, num_rows="dynamic")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download as CSV (for Anki/Quizlet)",
            data=df.to_csv(index=False),
            file_name="flashcards.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            label="📥 Download as JSON",
            data=json.dumps(st.session_state.flashcards, indent=2),
            file_name="flashcards.json",
            mime="application/json"
        )