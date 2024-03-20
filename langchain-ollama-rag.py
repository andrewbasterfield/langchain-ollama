#!/usr/bin/env python3
# https://python.langchain.com/docs/use_cases/question_answering/quickstart
# https://python.langchain.com/docs/integrations/llms/ollama
# https://smith.langchain.com/hub/rlm/rag-prompt
# https://python.langchain.com/docs/integrations/vectorstores/chroma
# https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ollama.Ollama.html

import argparse
import logging
import os
import sys
from operator import itemgetter
# get_prompt_local
from string import Template
from typing import Any, List, LiteralString, Optional

import wrapt
from dumper import dumps
# get_prompt
from langchain import hub
from langchain.globals import set_debug, set_verbose
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
# embeddings_filter_retriever_wrapper
from langchain.retrievers.document_compressors import EmbeddingsFilter
# multi_query_retriever_wrapper
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
# get_documents
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
# main
from langchain_community.llms import Ollama
# get_retriever
from langchain_community.vectorstores import Chroma
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSerializable, RunnableConfig, \
    Runnable
from langchain_core.runnables.utils import Input, Output
from langchain_core.vectorstores import VectorStore

os.environ["ANONYMIZED_TELEMETRY"] = "False"


def get_prompt_hub(llama: bool) -> Any:
    if llama:
        prompt = hub.pull("rlm/rag-prompt-llama")
    else:
        prompt = hub.pull("rlm/rag-prompt")
    return prompt


def get_prompt_local(llama: bool, question_token: str = "question") -> ChatPromptTemplate:
    if llama:
        template = """[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of 
        retrieved context to answer the question. If you don't know the answer, just say that you don't know.<</SYS>> 
        Question: {question} Context: {context} Answer: [/INST]"""
    else:
        template = """You are an assistant for question-answering tasks. Use only the following pieces of
        retrieved context to answer the question. If you don't know the answer, just say that you don't know.
        Question: {question} Context: {context} Answer:"""
    return ChatPromptTemplate.from_template(Template(template).substitute({"question": question_token}))


def get_loader(path) -> BaseLoader:
    # Only keep post title, headers, and content from the full HTML.
    # bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header",
    # "post-content")) bs4_strainer = bs4.SoupStrainer()

    logging.info("Loading & Splitting %s...", path)
    if path.upper().endswith(".PDF"):
        loader = PyMuPDFLoader(path)
    elif path.startswith("http://") or path.startswith("https://"):
        loader = WebBaseLoader(
            web_paths=(path,),
            # bs_kwargs={"parse_only": bs4_strainer},
        )
    else:
        loader = TextLoader(path, autodetect_encoding=True)

    return loader


def get_documents(loader: BaseLoader, chunk_size, chunk_overlap) -> List[Document]:
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
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
    """Convert Documents to a single string.:"""
    return "\n\n".join(doc.page_content for doc in docs)


def get_metadata_source(docs) -> LiteralString:
    return "\n".join(doc.metadata["source"] for doc in docs)


class RunnableWrapper(wrapt.ObjectProxy, Runnable[Input, Any]):

    def invoke(self, the_input: Input, config: Optional[RunnableConfig] = None) -> Output:
        logging.warning("wrapped: %s", dumps(self.__wrapped__))
        logging.warning("input: %s", dumps(the_input))
        logging.warning("config: %s", dumps(config))
        res: Output = self.__wrapped__.invoke(input=the_input, config=config)
        logging.warning("result: %s", dumps(res))
        return res


def main():
    parser = argparse.ArgumentParser(prog=sys.argv[0], formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="""examples:
\tingest:\t`echo https://tldp.org/HOWTO/html_single/8021X-HOWTO/ | %(prog)s --ingest`
\tquery:\t`%(prog)s --query=\"What is 802.1X?\"`
""")
    parser.add_argument("--ingest", action='store_true', help="read data locations line by line from STDIN and ingest")
    parser.add_argument("--ingest-chunk-size", default=1000, help="ingestion chunk size (default: %(default)s)")
    parser.add_argument("--ingest-overlap", default=100, help="ingestion chunk overlap size (default: %(default)s)")
    parser.add_argument("--query", help="query to ask model")
    parser.add_argument("--fetch-snippets", default=10, help="num doc snippets to get from database (default: %(default)s)")
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
    parser.add_argument("--enable-debug", action='store_true', help="Enable LangChain debug (default: %(default)s)")
    parser.add_argument("--enable-verbose", action='store_true', help="Enable LangChain verbose (default: %(default)s)")
    parser.add_argument("--sources", action='store_true', help="Show sources provided in context with query result")
    parser.add_argument("--db-location", default="./chroma_db/", help="Location of the database (default: %(default)s)")
    args = parser.parse_args()

    if (not args.ingest) and (not args.query):
        print("\nNeed at least one of --ingest or --query\n", file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(1)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    set_debug(args.enable_verbose or args.enable_debug)
    set_verbose(args.enable_verbose)

    embeddings = OllamaEmbeddings(model=args.embeddings_model, base_url=args.ollama_embeddings_url)

    vectorstore = None
    if args.ingest:
        logging.info("Adding documents to vectorstore")
        for path in sys.stdin:
            logging.info("Document path: " + path.strip())
            loader = get_loader(path.strip())
            docs = get_documents(loader=loader, chunk_size=int(args.ingest_chunk_size), chunk_overlap=int(args.ingest_overlap))
            vectorstore = get_vectorstore(embeddings=embeddings, documents=docs, directory=args.db_location)

    if vectorstore is None:
        logging.info("Instantiating vectorstore without documents")
        vectorstore = get_vectorstore(embeddings=embeddings, directory=args.db_location)

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": int(args.fetch_snippets)})

    if args.query:
        prompt: ChatPromptTemplate = get_prompt_local(llama=("llama" in args.generative_model))
        llm: BaseLLM = Ollama(model=args.generative_model, base_url=args.ollama_generation_url,
                              temperature=args.temperature)

        # First instantiate Runnable with {"question": <invoke arg>, raw_docs: <retriever callback (Runnable)>}
        # Then build context from formatting raw_docs. itemgetter is a python callable and RunnableLambda convert
        # callable to LangChain Runnable. Then feed through a normal RAG chain.
        # assign() on the chain and RunnablePassthrough.assign() in the chain are equivalent
        rag_chain_with_sources: RunnableSerializable = (
            RunnableParallel(question=RunnablePassthrough(), raw_docs=retriever)
            # .assign(context=itemgetter("raw_docs") | RunnableLambda(format_docs))
            .assign(context=lambda obj: format_docs(obj["raw_docs"]))
            .assign(prompt=prompt)
            .assign(answer=itemgetter("prompt") | llm | StrOutputParser()))
        # .assign(answer=(
        #    RunnablePassthrough.assign(context=lambda obj: format_docs(obj["raw_docs"]))
        #    | prompt | llm | StrOutputParser()
        # )))

        logging.info("Invoking chain...")
        result = rag_chain_with_sources.invoke(args.query)
        print(result["answer"])

        if len(result["raw_docs"]) == 0:
            logging.error("Context was empty, no relevant document snippets found")
            exit(1)

        if args.sources:
            for doc in result["raw_docs"]:
                print("**** " + str(doc.metadata) + " ****")
                print(doc.page_content)
            print("****")
        logging.info("Context had %d document snippets", len(result["raw_docs"]))


if __name__ == '__main__':
    main()
