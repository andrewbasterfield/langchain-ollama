#!/usr/bin/env python3

# https://python.langchain.com/docs/use_cases/question_answering/quickstart
# https://python.langchain.com/docs/integrations/llms/ollama
# https://smith.langchain.com/hub/rlm/rag-prompt
# https://python.langchain.com/docs/integrations/vectorstores/chroma
# https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ollama.Ollama.html

import logging
from typing import Any, List, LiteralString

# get_prompt
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
# get_documents
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
# main
from langchain_community.llms import Ollama
# get_retriever
from langchain_community.vectorstores import Chroma, LanceDB, FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore


def get_prompt_hub(llama: bool) -> Any:
    if llama:
        prompt = hub.pull("rlm/rag-prompt-llama")
    else:
        prompt = hub.pull("rlm/rag-prompt")
    return prompt


def get_prompt_local(llama: bool, question: object = "question") -> ChatPromptTemplate:
    if llama:
        template = """[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.<</SYS>> 
Question: {$question} 
Context: {context} 
Answer:
"""
    else:
        template = """You are an assistant for question-answering tasks. Use only the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Question: {$question} 
Context: {context} 
Answer: [/INST]
"""
    from string import Template
    prompt = ChatPromptTemplate.from_template(Template(template).substitute({"question": question}))
    return prompt


def get_documents(path, model_name, embeddings=None) -> List[Document]:
    # Only keep post title, headers, and content from the full HTML.
    # bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header",
    # "post-content")) bs4_strainer = bs4.SoupStrainer()

    if path.startswith("http://") or path.startswith("https://"):
        loader = WebBaseLoader(
            web_paths=(path,),
            # bs_kwargs={"parse_only": bs4_strainer},
        )
    else:
        loader = TextLoader(path, autodetect_encoding=True)

    logging.info("Loading & Splitting %s...", path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=100,
                                                   length_function=len,
                                                   add_start_index=False)
    # text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    # text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name=model_name, chunk_size=1000, chunk_overlap=100)

    # from langchain.text_splitter import NLTKTextSplitter
    # text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)

    # from langchain.text_splitter import SpacyTextSplitter
    # text_splitter = SpacyTextSplitter(max_length=100000)
    # text_splitter = SpacyTextSplitter(max_length=1500000, pipeline="sentencizer")

    # from langchain_experimental.text_splitter import SemanticChunker
    # text_splitter = SemanticChunker(embeddings)

    split_docs = text_splitter.split_documents(docs)
    logging.info("Loaded & Splitted.")
    return split_docs


def get_vectorstore_chroma(embeddings, documents=None,
                           directory=None) -> VectorStore:
    if directory is None:
        directory = "./chroma_db/"

    if documents is None:
        logging.info("Attempting to instantiate chroma vector store from %s...",
                     directory)
        vectorstore = Chroma(embedding_function=embeddings,
                             persist_directory=directory,
                             collection_metadata={"hnsw:space": "cosine"})
    else:
        logging.info(
            "Attempting to instantiate chroma vector store (with documents) from %s...",
            directory)
        vectorstore = Chroma.from_documents(documents=documents,
                                            embedding=embeddings,
                                            persist_directory=directory,
                                            collection_metadata={"hnsw:space": "cosine"})
    logging.info("Instantiated vectorstore.")
    return vectorstore


def get_lancedb_table(db, embeddings) -> Any:
    import pyarrow as pa

    try:
        tbl = db.open_table("vectorstore")
    except:
        schema = pa.schema(
            [
                pa.field(
                    "vector",
                    pa.list_(
                        pa.float32(),
                        len(embeddings.embed_query("test")),  # type: ignore
                    ),
                ),
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
            ]
        )

        tbl = db.create_table("vectorstore", schema=schema, exist_ok=True)
    return tbl


def get_vectorstore_lancedb(embeddings, documents=None,
                            directory=None) -> VectorStore:
    if directory is None:
        directory = "./lancedb/"

    import lancedb
    db = lancedb.connect(directory)
    tbl = get_lancedb_table(db, embeddings)

    if documents is None:
        logging.info("Attempting to instantiate lancedb vector store from %s...",
                     directory)
        vectorstore = LanceDB(embedding=embeddings,
                              connection=tbl)
    else:
        logging.info(
            "Attempting to instantiate lancedb vector store (with documents) from %s...",
            directory)
        vectorstore = LanceDB.from_documents(documents=documents,
                                             embedding=embeddings,
                                             connection=tbl)
    logging.info("Instantiated vectorstore.")
    return vectorstore


def get_vectorstore_faiss(embeddings, documents=None, directory=None) -> FAISS:
    if directory is None:
        directory = "./FAISS"

    try:
        print("FAISS: Loading from disk")
        db = FAISS.load_local(folder_path=directory, embeddings=embeddings)
    except:
        print("FAISS: Loading in memory")
        db = FAISS.from_texts(texts=[""], embedding=embeddings)

    if documents is not None:
        db.add_documents(documents)

    db.save_local(directory)
    return db


def get_vectorstore(embeddings, documents=None, directory=None) -> VectorStore:
    return get_vectorstore_chroma(embeddings=embeddings, documents=documents, directory=directory)


def multi_query_retriever_wrapper(retriever, llm) -> Any:
    from langchain.retrievers.multi_query import MultiQueryRetriever
    return MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm
    )


def embeddings_filter_retriever_wrapper(retriever, embeddings) -> Any:
    from langchain.retrievers.document_compressors import EmbeddingsFilter
    from langchain.retrievers.contextual_compression import \
        ContextualCompressionRetriever
    return ContextualCompressionRetriever(
        base_compressor=EmbeddingsFilter(embeddings=embeddings,
                                         similarity_threshold=0.76),
        base_retriever=retriever
    )


def format_docs(docs) -> LiteralString:
    return "\n\n".join(doc.page_content for doc in docs)


def main(args):
    import re
    ingest = False
    query = False
    temperature: int = 0
    # model = "llama2:7b"
    model = "mistral:7b"
    base_url = "http://localhost:11434"
    log_level = "info"

    for arg in args:
        if arg == "--ingest":
            ingest = True
        elif re.match("--query=(.+)$", arg):
            query = re.findall("--query=(.+)$", arg)[0]
        elif re.match("--temperature=(\\d+)", arg):
            temperature = re.findall("--temperature=(\\d+)", arg)[0]
        elif re.match("--model=(\\w+)", arg):
            model = re.findall("--model=(\\w+)", arg)[0]
        elif re.match("--base-url=(\\S+)", arg):
            base_url = re.findall("--base-url=(\\S+)", arg)[0]
        elif re.match("--log-level=(\\w+)", arg):
            log_level = re.findall("--log-level=(\\w+)", arg)[0]
        else:
            print("Unknown argument")
            exit(1)

    if (not ingest) and (not query):
        print("Need at least one of --ingest or --query")
        exit(1)

    logging.basicConfig(level=getattr(logging, log_level.upper()))
    from langchain.globals import set_debug, set_verbose
    set_verbose(True)
    set_debug(True)

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)

    if ingest:
        logging.warning("Adding documents to vectorstore")
        for web_path in sys.stdin:
            print("Web path: " + web_path.strip())
            docs = get_documents(web_path.strip(), model, embeddings)
            vectorstore = get_vectorstore(embeddings=embeddings, documents=docs)
    else:
        logging.warning("Instantiating vectorstore without documents")
        vectorstore = get_vectorstore(embeddings=embeddings)

    # retriever: VectorStoreRetriever = vectorstore.as_retriever()
    # retriever: VectorStoreRetriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retriever = vectorstore.as_retriever(search_type="mmr")
    # retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.0})

    if query:
        prompt: ChatPromptTemplate = get_prompt_local(llama=False, question="input")
        llm = Ollama(model=model, base_url=base_url, temperature=temperature)

        docs = retriever.get_relevant_documents(query=query)
        for doc in docs:
            logging.info(doc)
        if len(docs) == 0:
            exit(0)

        from langchain.chains.combine_documents import create_stuff_documents_chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        from langchain.chains import create_retrieval_chain
        rag_chain = create_retrieval_chain(retriever, document_chain)

        logging.info("Invoking chain...")
        result = rag_chain.invoke({"input": query})
        logging.info(result["answer"])


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
