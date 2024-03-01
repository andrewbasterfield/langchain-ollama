#!/usr/bin/env python3

# https://python.langchain.com/docs/use_cases/question_answering/quickstart
# https://python.langchain.com/docs/integrations/llms/ollama
# https://smith.langchain.com/hub/rlm/rag-prompt
# https://python.langchain.com/docs/integrations/vectorstores/chroma
# https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ollama.Ollama.html
from langchain_community.llms import Ollama

import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from ppretty import ppretty

import logging
logging.basicConfig(level=getattr(logging, "info".upper()))

model="llama2:7b"
#model="gemma:7b"
base_url="http://localhost:11434"
temperature=0

def get_prompt_hub():
    from langchain import hub
    prompt = hub.pull("rlm/rag-prompt")
    return prompt

def get_prompt_hub_llama():
    from langchain import hub
    prompt = hub.pull("rlm/rag-prompt-llama")
    return prompt

def get_prompt_local():
    ## Do the prompt
    template = """You are an assistant for question-answering tasks. Use only the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Question: {question} 
Context: {context} 
Answer:
"""
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def get_prompt_local_llama():
    ## Do the prompt
    template = """[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.<</SYS>> 
Question: {question} 
Context: {context} 
Answer: [/INST]
"""
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def get_documents():
    # Only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    logging.debug("loading...")
    docs = loader.load()
    logging.debug("loaded.")

    logging.debug("splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    splits = text_splitter.split_documents(docs)
    logging.debug("splitted.")
    return splits

def get_retriever(documents=None):
    directory="./chroma_db"
    if documents is None:
        logging.info("Attempting to instantiate vector store from %s...",directory)
        vectorstore = Chroma(embedding_function=OllamaEmbeddings(model=model, base_url=base_url), persist_directory="./chroma_db")
    else:
        logging.info("Attempting to instantiate vector store with documents from %s...",directory)
        vectorstore = Chroma.from_documents(documents=documents, embedding=OllamaEmbeddings(model=model, base_url=base_url), persist_directory="./chroma_db")
    logging.info("Instantiated vectorstore.")

    retriever = vectorstore.as_retriever()
    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():

    context = get_retriever(documents=get_documents()) | format_docs
    #context = get_retriever() | format_docs

    prompt = get_prompt_local_llama()

    llm = Ollama(model=model, base_url=base_url, temperature=temperature)

    rag_chain = (
        {"context": context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.debug("invoking chain...")
    result = rag_chain.invoke("What is Task Decomposition?")
    #result = rag_chain.invoke("What is Maximum Inner Product Search?")
    #result = rag_chain.invoke("What is an E46 M3?")

    print(result)

if __name__ == '__main__':
    main()
