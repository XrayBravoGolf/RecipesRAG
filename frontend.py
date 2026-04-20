import streamlit as st


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask a question to begin."}
        ]


def render_sidebar() -> dict:
    with st.sidebar:
        st.header("Settings")
        collection = st.text_input("Collection", value="default")
        top_k = st.slider("Top-K", min_value=1, max_value=20, value=5)
        show_context = st.toggle("Show retrieved context", value=True)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2)
        st.divider()
        clear = st.button("Clear chat history", use_container_width=True)

    if clear:
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared. Ask a new question."}
        ]

    return {
        "collection": collection,
        "top_k": top_k,
        "show_context": show_context,
        "temperature": temperature,
    }


def render_chat() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


def handle_user_input(config: dict) -> None:
    if prompt := st.chat_input("Ask something about your data"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            response = (
                "This is frontend boilerplate. "
                "Wire this spot to your RAG pipeline (retriever + generator) to return real answers."
            )
            st.write(response)

            if config["show_context"]:
                with st.expander("Retrieved context"):
                    st.write(
                        "Context rendering is enabled. Display your retrieved chunks and metadata here."
                    )

        st.session_state.messages.append({"role": "assistant", "content": response})


def main() -> None:
    st.set_page_config(page_title="Verifiable RAG Frontend", page_icon="🔎", layout="wide")
    st.title("Verifiable RAG")
    st.caption("Experimental Streamlit frontend")

    initialize_state()
    config = render_sidebar()
    render_chat()
    handle_user_input(config)


if __name__ == "__main__":
    main()
