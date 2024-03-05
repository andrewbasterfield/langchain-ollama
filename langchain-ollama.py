#!/usr/bin/env python3

# https://python.langchain.com/docs/use_cases/question_answering/quickstart
# https://python.langchain.com/docs/integrations/llms/ollama
# https://smith.langchain.com/hub/rlm/rag-prompt
# https://python.langchain.com/docs/integrations/vectorstores/chroma
# https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ollama.Ollama.html

import logging
from typing import TYPE_CHECKING, Any, Optional, List, LiteralString

# main
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import CharacterTextSplitter


# get_prompt
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

# get_documents
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


# get_retriever
from langchain_community.vectorstores import Chroma, LanceDB


def get_prompt_hub(llama: bool) -> Any:
    if llama:
        prompt = hub.pull("rlm/rag-prompt-llama")
    else:
        prompt = hub.pull("rlm/rag-prompt")
    return prompt


def get_prompt_local(llama: bool) -> ChatPromptTemplate:
    if llama:
        template = """[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.<</SYS>> 
Question: {question} 
Context: {context} 
Answer:
"""
    else:
        template = """You are an assistant for question-answering tasks. Use only the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Question: {question} 
Context: {context} 
Answer: [/INST]
"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def get_documents(web_path,model_name,embeddings=None) -> List[Document]:
    # Only keep post title, headers, and content from the full HTML.
    #bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    #bs4_strainer = bs4.SoupStrainer()

    loader = WebBaseLoader(
        web_paths=(web_path,),
        #bs_kwargs={"parse_only": bs4_strainer},
    )
    logging.info("Loading & Splitting %s...",web_path)
    docs = loader.load()

    #text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=250, add_start_index=False)
    #text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    #text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    #text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    #text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name=model_name, chunk_size=1000, chunk_overlap=100)

    from langchain.text_splitter import NLTKTextSplitter
    text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
    
    #from langchain.text_splitter import SpacyTextSplitter
    #text_splitter = SpacyTextSplitter(max_length=100000)
    #text_splitter = SpacyTextSplitter(max_length=1500000, pipeline="sentencizer")
    
    #from langchain_experimental.text_splitter import SemanticChunker
    #text_splitter = SemanticChunker(embeddings)

    split_docs = text_splitter.split_documents(docs)
    logging.info("Loaded & Splitted.")
    return split_docs

def get_vectorstore_chroma(embeddings, documents=None, directory = "./chroma_db/") -> VectorStore:    
    if documents is None:
        logging.info("Attempting to instantiate chroma vector store from %s...", directory)
        vectorstore = Chroma(embedding_function=embeddings,
                             persist_directory=directory)
    else:
        logging.info("Attempting to instantiate chroma vector store (with documents) from %s...", directory)
        vectorstore = Chroma.from_documents(documents=documents,
                                            embedding=embeddings,
                                            persist_directory=directory)
    logging.info("Instantiated vectorstore.")

def get_lancedb_table(db,embeddings) -> Any:
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

def get_vectorstore_lancedb(embeddings, documents=None, directory = "./lancedb/") -> VectorStore:

    import lancedb
    db = lancedb.connect(directory)
    tbl = get_lancedb_table(db,embeddings)

    if documents is None:
        logging.info("Attempting to instantiate lancedb vector store from %s...", directory)
        vectorstore = LanceDB(embedding=embeddings,
                             connection=tbl)
    else:
        logging.info("Attempting to instantiate lancedb vector store (with documents) from %s...", directory)
        vectorstore = LanceDB.from_documents(documents=documents,
                                            embedding=embeddings,
                                            connection=tbl)
    logging.info("Instantiated vectorstore.")
    return vectorstore

def multi_query_retriever_wrapper(retriever,llm) -> Any:
    from langchain.retrievers.multi_query import MultiQueryRetriever
    return MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm
    )

def embeddings_filter_retriever_wrapper(retriever,embeddings) -> Any:
    from langchain.retrievers.document_compressors import EmbeddingsFilter
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    return ContextualCompressionRetriever(
        base_compressor=EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76), base_retriever=retriever
    )

def format_docs(docs) -> LiteralString:
    return "\n\n".join(doc.page_content for doc in docs)

def main(args):
    import re
    ingest = False
    query = False
    temperature : int = 0
    model = "zephyr"
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

    embeddings = OllamaEmbeddings(model=model, base_url=base_url)

    if ingest:
        logging.warning("Instantiating vectorstore from documents")
        for web_path in sys.stdin:
            print("Web path: "+web_path.strip())
            vectorstore = get_vectorstore_lancedb(embeddings, documents=get_documents(web_path.strip(),model,embeddings))
    else:
        logging.warning("Instantiating vectorstore without documents")
        vectorstore = get_vectorstore_lancedb(embeddings)
    
    #retriever = vectorstore.as_retriever() # similarity
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    #retriever = vectorstore.as_retriever(search_type="mmr")
    #retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}) # not working

    
    #retriever = embeddings_filter_retriever_wrapper(retriever,embeddings)

    if query:


        docs = retriever.get_relevant_documents(query)
        import pprint
        for doc in docs:
            #pprint.pprint(doc.page_content)
            print("Content: %s",doc.page_content)

        prompt = get_prompt_local(True)
        llm = Ollama(model=model, base_url=base_url, temperature=temperature)
        #retriever = multi_query_retriever_wrapper(retriever=retriever,llm=llm)
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        logging.info("Invoking chain...")
        result = rag_chain.invoke(query)
        print(result)


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
