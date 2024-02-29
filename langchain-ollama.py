#!/usr/bin/env python3

# https://python.langchain.com/docs/use_cases/question_answering/quickstart
# https://python.langchain.com/docs/integrations/llms/ollama
# https://smith.langchain.com/hub/rlm/rag-prompt
# https://python.langchain.com/docs/integrations/vectorstores/chroma
# https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ollama.Ollama.html
from langchain_community.llms import Ollama

import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings

model="llama2"
base_url="http://localhost:11434"


# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
print("loading...")
docs = loader.load()
print("loaded.")

print("splitting...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
splits = text_splitter.split_documents(docs)
print("splitted.")


print("vectorstoring...")
vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model=model, base_url=base_url))
print("vectorstored.")

retriever = vectorstore.as_retriever()
#retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
prompt = hub.pull("rlm/rag-prompt")
#print(prompt)

llm = Ollama(model=model, base_url=base_url)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("invoking chain...")
result = rag_chain.invoke("What is Task Decomposition?")

print(result)
