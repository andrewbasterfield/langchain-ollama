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
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.vectorstores import VectorStore
# multi_query_retriever_wrapper
from langchain.retrievers.multi_query import MultiQueryRetriever
# get_prompt_local
from string import Template
# embeddings_filter_retriever_wrapper
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever


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

    split_docs = text_splitter.split_documents(docs)
    logging.info("Loaded & Splitted.")
    return split_docs


def get_vectorstore(embeddings, documents=None, directory="./chroma_db/") -> VectorStore:

    if documents is None:
        logging.debug("Attempting to instantiate chroma vector store from %s...",
                     directory)
        vectorstore = Chroma(embedding_function=embeddings,
                             persist_directory=directory,
                             collection_metadata={"hnsw:space": "cosine"})
    else:
        logging.debug(
            "Attempting to instantiate chroma vector store (with documents) from %s...",
            directory)
        vectorstore = Chroma.from_documents(documents=documents,
                                            embedding=embeddings,
                                            persist_directory=directory,
                                            collection_metadata={"hnsw:space": "cosine"})
    logging.debug("Instantiated vectorstore.")
    return vectorstore


def multi_query_retriever_wrapper(retriever, llm) -> Any:
    return MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm
    )


def embeddings_filter_retriever_wrapper(retriever, embeddings, similarity_threshold=0.6) -> Any:
    return ContextualCompressionRetriever(
        base_compressor=EmbeddingsFilter(embeddings=embeddings,
                                         similarity_threshold=similarity_threshold),
        base_retriever=retriever
    )


def format_docs(docs) -> LiteralString:
    return "\n\n".join(doc.page_content for doc in docs)


def get_metadata_source(docs) -> LiteralString:
    return "\n".join(doc.metadata["source"] for doc in docs)


def main(args):
    import re
    ingest = False
    query = False
    temperature: int = 0
    embeddings_model = "nomic-embed-text"
    model = "llama2:7b"
    base_url = "http://localhost:11434"
    log_level = "info"
    langchain_verbose = False
    langchain_debug = False
    sources = False

    def usage(file=sys.stdout):
        print("Usage: "+(args[0]), file=file)
        print("\t--ingest\t\t\t\tread data locations line by line from STDIN and ingest", file=file)
        print("\t--query=<query>\t\t\t\tquery to ask model", file=file)
        print("\t--temperature=N\t\t\t\tmodel temperature for query (default: 0)", file=file)
        print("\t--model=<model>\t\t\t\tmodel (default: \""+model+"\")", file=file)
        print("\t--base-url=<base-url>\t\t\tURL for Ollama API (default: \"" + base_url + "\")", file=file)
        print("\t--log-level=<debug|info|warning>\tLog level (default: \"" + log_level + "\")", file=file)
        print("\t--langchain-verbose\t\t\tenable langchain verbose mode", file=file)
        print("\t--langchain-debug\t\t\tenable langchain debug mode", file=file)
        print("\t--sources\t\t\t\tprint locations of sources used as context", file=file)
        print("Examples:", file=file)
        print("\tingest:\t`echo https://tldp.org/HOWTO/html_single/8021X-HOWTO/ | "+args[0]+" --ingest`", file=file)
        print("\tquery:\t`" + args[0] + " --query=\"What is 802.1X?\"`", file=file)

    for arg in args[1:]:
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
        elif arg == "--langchain-verbose":
            langchain_verbose = True
        elif arg == "--langchain-debug":
            langchain_debug = True
        elif arg == "--sources":
            sources = True
        elif arg == "--help":
            usage()
            exit(0)
        else:
            print("Unknown argument: "+arg, file=sys.stderr)
            usage(sys.stderr)
            exit(1)

    if (not ingest) and (not query):
        print("Need at least one of --ingest or --query", file=sys.stderr)
        usage(sys.stderr)
        exit(1)

    logging.basicConfig(level=getattr(logging, log_level.upper()))
    from langchain.globals import set_debug, set_verbose

    if langchain_verbose:
        set_verbose(False)

    if langchain_debug:
        set_debug(False)

    embeddings = OllamaEmbeddings(model=embeddings_model, base_url=base_url)

    if ingest:
        logging.debug("Adding documents to vectorstore")
        for path in sys.stdin:
            logging.debug("Document path: " + path.strip())
            docs = get_documents(path.strip(), model, embeddings)
            vectorstore = get_vectorstore(embeddings=embeddings, documents=docs)
    else:
        logging.debug("Instantiating vectorstore without documents")
        vectorstore = get_vectorstore(embeddings=embeddings)

    retriever = vectorstore.as_retriever(search_type="mmr")

    if query:
        prompt: ChatPromptTemplate = get_prompt_local(llama=("llama" in model))
        llm = Ollama(model=model, base_url=base_url, temperature=temperature)
        # retriever = multi_query_retriever_wrapper(retriever, llm)
        # retriever = embeddings_filter_retriever_wrapper(retriever, embeddings, similarity_threshold=0.6)

        rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda d: format_docs(d["context"])))
                | prompt
                | llm
                | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        logging.debug("Invoking chain...")
        result = rag_chain_with_source.invoke(query)
        print(result["answer"])

        if len(result["context"]) == 0:
            logging.error("Context is empty, no matched documents")
            exit(1)

        if sources:
            for doc in result["context"]:
                print("source: "+doc.metadata["source"])


if __name__ == '__main__':
    import sys
    main(sys.argv)
