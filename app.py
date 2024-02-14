import argparse
import os
from dataclasses import dataclass
import shutil
import streamlit as st
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

os.environ['OPENAI_API_KEY'] = "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"

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

@dataclass
class Config:
    chroma_path: str
    data_path: str
    prompt_template: str

def main():
    # CLI mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    parser.add_argument("query_text", nargs="?", type=str, help="The query text (only for CLI mode)")
    args = parser.parse_args()

    if args.cli:
        if args.query_text is None:
            print("Please provide a query text.")
            return
        run_cli(args.query_text)
    else:
        run_streamlit()


def run_cli(query_text):
    generate_data_store()
    prepare_and_search(query_text)


def run_streamlit():
    st.title("Summon the Magical Title: 'Inquiries of Hogwarts")

    query_text = st.text_area("Unveil Your Incantation (Query):", "")

    if st.button("Evoke the Response"):
        if query_text.strip() == "":
            st.error("A Query Must Be Forged, Wizard!")
            return
        prepare_and_search(query_text)


def prepare_and_search(query_text):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        st.warning("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    st.subheader("Response:")
    st.write(response_text)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    st.subheader("Sources:")
    st.write(", ".join(sources))


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
