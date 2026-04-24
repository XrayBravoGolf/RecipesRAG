import streamlit as st
import pandas as pd
from retrieval import RecipeRetriever
from generator import RecipeGenerator


@st.cache_resource
def get_retriever():
    return RecipeRetriever()


@st.cache_resource
def get_generator():
    return RecipeGenerator()


def initialize_state(force_clear: bool = False) -> None:
    if force_clear or "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask a question to begin. (Note: I cannot answer follow-up questions.)",
            }
        ]


def render_sidebar() -> dict:
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=5)
        show_context = st.toggle("Show retrieved context", value=True)
        st.divider()
        clear = st.button("Clear chat history", use_container_width=True)

    if clear:
        initialize_state(force_clear=True)

    return {
        "top_k": top_k,
        "show_context": show_context,
    }


def render_chat() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


def handle_user_input(config: dict) -> None:
    if prompt := st.chat_input("I want to make pizza without tomato-based sauce"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving recipes..."):
                retriever = get_retriever()
                docs = retriever.search(prompt, k=config["top_k"])
            
            with st.spinner("Generating answer..."):
                generator = get_generator()
                if not docs:
                    response = "No relevant recipes found."
                else:
                    response = generator.generate(prompt, docs)
            
            st.write(response)

            if config["show_context"] and docs:
                with st.expander("Retrieved context (Verifiable Evidence)"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**Result {i} (Score: {doc['score']:.4f})**")
                        st.markdown(f"File: `{doc['parquet_filename']}` | Row: `{doc['row_in_file']}` (Global: {doc['global_row_id']})")
                        st.text_area(f"Recipe {i}", doc['recipe_text'], height=150, disabled=True)
                    
                    csv_data = pd.DataFrame(docs).to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Evidence as CSV",
                        data=csv_data,
                        file_name="evidence.csv",
                        mime="text/csv"
                    )

        st.session_state.messages.append({"role": "assistant", "content": response})


def main() -> None:
    st.set_page_config(page_title="Verifiable RAG Frontend", page_icon="🔎", layout="wide")
    st.title("Let's clear your fridge!")
    st.caption("Verifiable RAG running on Streamlit frontend")

    initialize_state()
    config = render_sidebar()
    render_chat()
    handle_user_input(config)


if __name__ == "__main__":
    main()
