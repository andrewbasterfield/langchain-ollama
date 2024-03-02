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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from ppretty import ppretty

import logging
logging.basicConfig(level=getattr(logging, "info".upper()))

#model="llama2:7b"
#model="gemma:7b"
model="zephyr"
base_url="http://localhost:11434"
temperature=5

def get_prompt_hub(llama: bool):
    from langchain import hub
    if llama:
        prompt = hub.pull("rlm/rag-prompt-llama")
    else:
        prompt = hub.pull("rlm/rag-prompt")
    return prompt

def get_prompt_local(llama: bool):
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
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def get_prompt():
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are world class technical documentation writer."),
        ("user", "{question}")
    ])
    return prompt


def get_documents():
    # Only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

    loader = WebBaseLoader(
        #web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        web_paths=("https://tldp.org/HOWTO/html_single/LDAP-HOWTO/",),
        #bs_kwargs={"parse_only": bs4_strainer},
    )
    logging.debug("Loading & Splitting...")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
    splits = text_splitter.split_documents(docs)
    logging.debug("Loaded & Splitted.")
    return splits

def get_retriever_chroma(documents=None):
    from langchain_community.vectorstores import Chroma
    directory="./chroma_db/"+model
    if documents is None:
        logging.info("Attempting to instantiate vector store from %s...",directory)
        vectorstore = Chroma(embedding_function=OllamaEmbeddings(model=model, base_url=base_url), persist_directory=directory)
    else:
        logging.info("Attempting to instantiate vector store with documents from %s...",directory)
        vectorstore = Chroma.from_documents(documents=documents, embedding=OllamaEmbeddings(model=model, base_url=base_url), persist_directory=directory)
    logging.info("Instantiated vectorstore.")

    #retriever = vectorstore.as_retriever(search_type="mmr")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever

def get_retriever_faiss(documents):
    from langchain_community.vectorstores import FAISS
    vector = FAISS.from_documents(documents, OllamaEmbeddings(model=model, base_url=base_url))
    return vector.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():

    #context = get_retriever_faiss(documents=get_documents()) | format_docs
    context = get_retriever_chroma(documents=get_documents()) | format_docs
    #context = get_retriever_chroma() | format_docs

    prompt = get_prompt_local(False)

    llm = Ollama(model=model, base_url=base_url, temperature=temperature)

    rag_chain = (
        {"context": context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Invoking chain...")
    #result = rag_chain.invoke("What is Task Decomposition?")
    #result = rag_chain.invoke("What is Maximum Inner Product Search?")
    #result = rag_chain.invoke("What is an E46 M3?")
    result = rag_chain.invoke("How to create a database?")

    print(result)

if __name__ == '__main__':
    main()
