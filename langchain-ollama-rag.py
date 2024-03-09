#!/usr/bin/env python3
# https://python.langchain.com/docs/use_cases/question_answering/quickstart
# https://python.langchain.com/docs/integrations/llms/ollama
# https://smith.langchain.com/hub/rlm/rag-prompt
# https://python.langchain.com/docs/integrations/vectorstores/chroma
# https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ollama.Ollama.html

import logging
import os
import argparse
import sys

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

os.environ["ANONYMIZED_TELEMETRY"] = "False"


def get_prompt_hub(llama: bool) -> Any:
    if llama:
        prompt = hub.pull("rlm/rag-prompt-llama")
    else:
        prompt = hub.pull("rlm/rag-prompt")
    return prompt


def get_prompt_local(llama: bool, question: object = "question") -> ChatPromptTemplate:
    if llama:
        template = """[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of 
        retrieved context to answer the question. If you don't know the answer, just say that you don't know.<</SYS>> 
        Question: {question} Context: {context} Answer: [/INST]"""
    else:
        template = """You are an assistant for question-answering tasks. Use only the following pieces of
        retrieved context to answer the question. If you don't know the answer, just say that you don't know.
        Question: {question} Context: {context} Answer:"""
    prompt = ChatPromptTemplate.from_template(Template(template).substitute({"question": question}))
    return prompt


def get_documents(path) -> List[Document]:
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
                                                   add_start_index=True)

    split_docs = text_splitter.split_documents(docs)
    logging.info("Loaded & Split.")
    return split_docs


def get_vectorstore(embeddings, documents=None, directory="./chroma_db/") -> VectorStore:
    if documents is None:
        logging.info("Attempting to instantiate chroma vector store from %s...",
                     directory)
        vectorstore = Chroma(embedding_function=embeddings,
                             persist_directory=directory,
                             collection_metadata={"hnsw:space": "cosine"})
    else:
        logging.info(
            "Attempting to instantiate chroma vector store (with %d documents to add) from %s...",
            len(documents), directory)
        vectorstore = Chroma.from_documents(documents=documents,
                                            embedding=embeddings,
                                            persist_directory=directory,
                                            collection_metadata={"hnsw:space": "cosine"})
    logging.info("Instantiated vectorstore.")
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


def main():
    parser = argparse.ArgumentParser(prog=sys.argv[0], formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="""examples:
\tingest:\t`echo https://tldp.org/HOWTO/html_single/8021X-HOWTO/ | %(prog)s --ingest`
\tquery:\t`%(prog)s --query=\"What is 802.1X?\"`"
""")
    parser.add_argument("--ingest", action='store_true', help="read data locations line by line from STDIN and ingest")
    parser.add_argument("--query", help="query to ask model")
    parser.add_argument("--temperature", default=0, help="model temperature for query (default: %(default)s)")
    parser.add_argument("--embeddings-model", default="nomic-embed-text",
                        help="model used for creating the embeddings (default: %(default)s)")
    parser.add_argument("--ollama-embeddings-url", default="http://localhost:11434",
                        help="URL for Ollama API for embeddings (default: %(default)s)")
    parser.add_argument("--generative-model", default="llama2:7b",
                        help="model used for generation (default: %(default)s)")
    parser.add_argument("--ollama-generation-url", default="http://localhost:11434",
                        help="URL for Ollama API for generation (default: %(default)s)")
    parser.add_argument("--log-level", default="warning", help="Log threshold (default: %(default)s)")
    parser.add_argument("--sources", action='store_true', help="Show sources provided in context with query result")
    parser.add_argument("--db-location", default="./chroma_db/", help="Location of the database (default: %(default)s)")
    args = parser.parse_args()

    if (not args.ingest) and (not args.query):
        print("\nNeed at least one of --ingest or --query\n", file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(1)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    embeddings = OllamaEmbeddings(model=args.embeddings_model, base_url=args.ollama_embeddings_url)

    vectorstore = None
    if args.ingest:
        logging.info("Adding documents to vectorstore")
        for path in sys.stdin:
            logging.info("Document path: " + path.strip())
            docs = get_documents(path.strip())
            vectorstore = get_vectorstore(embeddings=embeddings, documents=docs, directory=args.db_location)

    if vectorstore is None:
        logging.info("Instantiating vectorstore without documents")
        vectorstore = get_vectorstore(embeddings=embeddings, directory=args.db_location)

    retriever = vectorstore.as_retriever(search_type="mmr")

    if args.query:
        prompt: ChatPromptTemplate = get_prompt_local(llama=("llama" in args.generative_model))
        llm = Ollama(model=args.generative_model, base_url=args.ollama_generation_url, temperature=args.temperature)

        rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda d: format_docs(d["context"])))
                | prompt
                | llm
                | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        logging.info("Invoking chain...")
        result = rag_chain_with_source.invoke(args.query)
        print(result["answer"])

        if len(result["context"]) == 0:
            logging.error("Context was empty, no relevant documents found")
            exit(1)

        if args.sources:
            for doc in result["context"]:
                print("**** " + str(doc.metadata) + " ****")
                print(doc.page_content)
            print("****")


if __name__ == '__main__':
    main()
