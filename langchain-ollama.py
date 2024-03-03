#!/usr/bin/env python3

# https://python.langchain.com/docs/use_cases/question_answering/quickstart
# https://python.langchain.com/docs/integrations/llms/ollama
# https://smith.langchain.com/hub/rlm/rag-prompt
# https://python.langchain.com/docs/integrations/vectorstores/chroma
# https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ollama.Ollama.html

import logging

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

# get_retriever
from langchain_community.vectorstores import Chroma

def get_prompt_hub(llama: bool):
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
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

web_paths=[
    "https://tldp.org/HOWTO/text/8021X-HOWTO",
    "https://tldp.org/HOWTO/text/ACPI-HOWTO",
    "https://tldp.org/HOWTO/text/AI-Alife-HOWTO",
    "https://tldp.org/HOWTO/text/Assembly-HOWTO",
    "https://tldp.org/HOWTO/text/Astronomy-HOWTO",
    "https://tldp.org/HOWTO/text/Athlon-Powersaving-HOWTO",
    "https://tldp.org/HOWTO/text/Autodir-HOWTO",
    "https://tldp.org/HOWTO/text/Aviation-HOWTO",
    "https://tldp.org/HOWTO/text/Avr-Microcontrollers-in-Linux-Howto",
    "https://tldp.org/HOWTO/text/Battery-Powered",
    "https://tldp.org/HOWTO/text/Beowulf-HOWTO",
    "https://tldp.org/HOWTO/text/BogoMips",
    "https://tldp.org/HOWTO/text/BTTV",
    "https://tldp.org/HOWTO/text/Cable-Modem",
    "https://tldp.org/HOWTO/text/C++-dlopen",
    "https://tldp.org/HOWTO/text/Cluster-HOWTO",
    "https://tldp.org/HOWTO/text/Coffee",
    "https://tldp.org/HOWTO/text/Compaq-T1500-HOWTO",
    "https://tldp.org/HOWTO/text/Config-HOWTO",
    "https://tldp.org/HOWTO/text/Cryptoloop-HOWTO",
    "https://tldp.org/HOWTO/text/DB2-HOWTO",
    "https://tldp.org/HOWTO/text/Debian-and-Windows-Shared-Printing",
    "https://tldp.org/HOWTO/text/Debian-Binary-Package-Building-HOWTO",
    "https://tldp.org/HOWTO/text/Debian-Jigdo",
    "https://tldp.org/HOWTO/text/Disk-Encryption-HOWTO",
    "https://tldp.org/HOWTO/text/Disk-on-Chip-HOWTO",
    "https://tldp.org/HOWTO/text/DocBook-Demystification-HOWTO",
    "https://tldp.org/HOWTO/text/DPT-Hardware-RAID-HOWTO",
    "https://tldp.org/HOWTO/text/DVD-Playback-HOWTO",
    "https://tldp.org/HOWTO/text/Ecology-HOWTO",
    "https://tldp.org/HOWTO/text/Encrypted-Root-Filesystem-HOWTO",
    "https://tldp.org/HOWTO/text/Ethernet-Bridge-netfilter-HOWTO",
    "https://tldp.org/HOWTO/text/FBB",
    "https://tldp.org/HOWTO/text/Fedora-Multimedia-Installation-HOWTO",
    "https://tldp.org/HOWTO/text/Filesystems-HOWTO",
    "https://tldp.org/HOWTO/text/Finnish-HOWTO",
    "https://tldp.org/HOWTO/text/Flash-Memory-HOWTO",
    "https://tldp.org/HOWTO/text/Font-HOWTO",
    "https://tldp.org/HOWTO/text/Framebuffer-HOWTO",
    "https://tldp.org/HOWTO/text/Glibc-Install-HOWTO",
    "https://tldp.org/HOWTO/text/Hardware-HOWTO",
    "https://tldp.org/HOWTO/text/Home-Electrical-Control",
    "https://tldp.org/HOWTO/text/HOWTO-INDEX",
    "https://tldp.org/HOWTO/text/Howtos-with-LinuxDoc",
    "https://tldp.org/HOWTO/text/Implement-Sys-Call-Linux-2.6-i386",
    "https://tldp.org/HOWTO/text/Infrared-HOWTO",
    "https://tldp.org/HOWTO/text/Installfest-HOWTO",
    "https://tldp.org/HOWTO/text/IP-Masquerade-HOWTO",
    "https://tldp.org/HOWTO/text/IRC",
    "https://tldp.org/HOWTO/text/Italian-HOWTO",
    "https://tldp.org/HOWTO/text/K7s5a-HOWTO",
    "https://tldp.org/HOWTO/text/Kerberos-Infrastructure-HOWTO",
    "https://tldp.org/HOWTO/text/Large-Disk-HOWTO",
    "https://tldp.org/HOWTO/text/LDAP-HOWTO",
    "https://tldp.org/HOWTO/text/LDP-Reviewer-HOWTO",
    "https://tldp.org/HOWTO/text/Leased-Line",
    "https://tldp.org/HOWTO/text/libdc1394-HOWTO",
    "https://tldp.org/HOWTO/text/LILO",
    "https://tldp.org/HOWTO/text/Linksys-Blue-Box-Router-HOWTO",
    "https://tldp.org/HOWTO/text/Linux-Complete-Backup-and-Recovery-HOWTO",
    "https://tldp.org/HOWTO/text/Linux-Gamers-HOWTO",
    "https://tldp.org/HOWTO/text/Linux-i386-Boot-Code-HOWTO",
    "https://tldp.org/HOWTO/text/Linux+IPv6-HOWTO",
    "https://tldp.org/HOWTO/text/Linux+WinNT",
    "https://tldp.org/HOWTO/text/LVM-HOWTO",
    "https://tldp.org/HOWTO/text/Mail-User-HOWTO",
    "https://tldp.org/HOWTO/text/Masquerading-Simple-HOWTO",
    "https://tldp.org/HOWTO/text/Medicine-HOWTO",
    "https://tldp.org/HOWTO/text/MMBase-Inst-HOWTO",
    "https://tldp.org/HOWTO/text/Mobile-IPv6-HOWTO",
    "https://tldp.org/HOWTO/text/Mock-Mainframe",
    "https://tldp.org/HOWTO/text/Modem-HOWTO",
    "https://tldp.org/HOWTO/text/Module-HOWTO",
    "https://tldp.org/HOWTO/text/MP3-CD-Burning",
    "https://tldp.org/HOWTO/text/NC-HOWTO",
    "https://tldp.org/HOWTO/text/NCURSES-Programming-HOWTO",
    "https://tldp.org/HOWTO/text/NET3-4-HOWTO",
    "https://tldp.org/HOWTO/text/Networking-Overview-HOWTO",
    "https://tldp.org/HOWTO/text/NLM-HOWTO",
    "https://tldp.org/HOWTO/text/OLSR-IPv6-HOWTO",
    "https://tldp.org/HOWTO/text/Online-Troubleshooting-HOWTO",
    "https://tldp.org/HOWTO/text/openMosix-HOWTO",
    "https://tldp.org/HOWTO/text/Oracle-9i-Fedora-3-Install-HOWTO",
    "https://tldp.org/HOWTO/text/Outlook-to-Unix-Mailbox",
    "https://tldp.org/HOWTO/text/Parallel-Processing-HOWTO",
    "https://tldp.org/HOWTO/text/PA-RISC-Linux-Boot-HOWTO",
    "https://tldp.org/HOWTO/text/Partition",
    "https://tldp.org/HOWTO/text/Partition-Mass-Storage-Definitions-Naming-HOWTO",
    "https://tldp.org/HOWTO/text/Partition-Mass-Storage-Dummies-Linux-HOWTO",
    "https://tldp.org/HOWTO/text/Partition-Rescue",
    "https://tldp.org/HOWTO/text/PCMCIA-HOWTO",
    "https://tldp.org/HOWTO/text/Plug-and-Play-HOWTO",
    "https://tldp.org/HOWTO/text/Postfix-Cyrus-Web-cyradm-HOWTO",
    "https://tldp.org/HOWTO/text/Qmail-ClamAV-HOWTO",
    "https://tldp.org/HOWTO/text/Quake-HOWTO",
    "https://tldp.org/HOWTO/text/Reading-List-HOWTO",
    "https://tldp.org/HOWTO/text/RedHat-CD-HOWTO",
    "https://tldp.org/HOWTO/text/Reliance-HOWTO",
    "https://tldp.org/HOWTO/text/Samba-Authenticated-Gateway-HOWTO",
    "https://tldp.org/HOWTO/text/Scanner-HOWTO",
    "https://tldp.org/HOWTO/text/Scientific-Computing-with-GNU-Linux",
    "https://tldp.org/HOWTO/text/SCSI-2.4-HOWTO",
    "https://tldp.org/HOWTO/text/Secure-BootCD-VPN-HOWTO",
    "https://tldp.org/HOWTO/text/Security-HOWTO",
    "https://tldp.org/HOWTO/text/Serial-HOWTO",
    "https://tldp.org/HOWTO/text/Software-RAID-HOWTO",
    "https://tldp.org/HOWTO/text/Software-Release-Practice-HOWTO",
    "https://tldp.org/HOWTO/text/Spam-Filtering-for-MX",
    "https://tldp.org/HOWTO/text/SPARC-HOWTO",
    "https://tldp.org/HOWTO/text/SquashFS-HOWTO",
    "https://tldp.org/HOWTO/text/TCP-Keepalive-HOWTO",
    "https://tldp.org/HOWTO/text/Text-Terminal-HOWTO",
    "https://tldp.org/HOWTO/text/TimePrecision-HOWTO",
    "https://tldp.org/HOWTO/text/TimeSys-Linux-Install-HOWTO",
    "https://tldp.org/HOWTO/text/Traffic-Control-HOWTO",
    "https://tldp.org/HOWTO/text/Traffic-Control-tcng-HTB-HOWTO",
    "https://tldp.org/HOWTO/text/TT-XFree86",
    "https://tldp.org/HOWTO/text/Unix-and-Internet-Fundamentals-HOWTO",
    "https://tldp.org/HOWTO/text/Unix-Hardware-Buyer-HOWTO",
    "https://tldp.org/HOWTO/text/UPS-HOWTO",
    "https://tldp.org/HOWTO/text/User-Authentication-HOWTO",
    "https://tldp.org/HOWTO/text/User-Group-HOWTO",
    "https://tldp.org/HOWTO/text/VMS-to-Linux-HOWTO",
    "https://tldp.org/HOWTO/text/Webcam-HOWTO",
    "https://tldp.org/HOWTO/text/WikiText-HOWTO",
    "https://tldp.org/HOWTO/text/Windows-LAN-Server-HOWTO",
    "https://tldp.org/HOWTO/text/Windows-Newsreaders-under-Linux-HOWTO",
    "https://tldp.org/HOWTO/text/Wireless-Link-sys-WPC11",
    "https://tldp.org/HOWTO/text/Wireless-Sync-HOWTO",
    "https://tldp.org/HOWTO/text/XDMCP-HOWTO",
    "https://tldp.org/HOWTO/text/XFree86-Touch-Screen-HOWTO",
    "https://tldp.org/HOWTO/text/XFree86-Video-Timings-HOWTO",
    "https://tldp.org/HOWTO/text/XFree-Local-multi-user-HOWTO",
    "https://tldp.org/HOWTO/text/Xinerama-HOWTO",
    "https://tldp.org/HOWTO/text/XWindow-User-HOWTO"
]
#print(type(web_paths))
#exit(0)

def get_documents(web_path,embeddings=None):
    # Only keep post title, headers, and content from the full HTML.
    #bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    #bs4_strainer = bs4.SoupStrainer()

    loader = WebBaseLoader(
        web_paths=(web_path,),
        #bs_kwargs={"parse_only": bs4_strainer},
    )
    logging.info("Loading & Splitting %s...",web_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100, add_start_index=True)
    #text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    #text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    #text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)

    #from langchain.text_splitter import NLTKTextSplitter
    #text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
    
    #from langchain.text_splitter import SpacyTextSplitter    
    #text_splitter = SpacyTextSplitter(max_length=1500000, pipeline="sentencizer")
    
    #from langchain_experimental.text_splitter import SemanticChunker
    #text_splitter = SemanticChunker(embeddings)

    
    splits = text_splitter.split_documents(docs)
    logging.info("Loaded & Splitted.")
    return splits


def get_retriever_chroma(embeddings, documents=None):
    directory = "./chroma_db/"
    if documents is None:
        logging.info("Attempting to instantiate vector store from %s...", directory)
        vectorstore = Chroma(embedding_function=embeddings,
                             persist_directory=directory)
    else:
        logging.info("Attempting to instantiate vector store (with documents) from %s...", directory)
        vectorstore = Chroma.from_documents(documents=documents,
                                            embedding=embeddings,
                                            persist_directory=directory)
    logging.info("Instantiated vectorstore.")

    #return vectorstore.as_retriever()
    #return vectorstore.as_retriever(search_type="mmr")
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    #return vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}) # not working

def multi_query_retriever_wrapper(retriever,llm):
    from langchain.retrievers.multi_query import MultiQueryRetriever
    return MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm
    )

def embeddings_filter_retriever_wrapper(retriever,embeddings):
    from langchain.retrievers.document_compressors import EmbeddingsFilter
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    return ContextualCompressionRetriever(
        base_compressor=EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76), base_retriever=retriever
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main(args):
    import re
    ingest = False
    query = False
    temperature = 0
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
        logging.warning("Building chroma from documents")
        for web_path in web_paths[0:2]:
            print("Web path: "+web_path)
            retriever = get_retriever_chroma(embeddings, documents=get_documents(web_path,embeddings))
    else:
        logging.warning("Building chroma without documents")
        retriever = get_retriever_chroma(embeddings)
    
    #retriever = embeddings_filter_retriever_wrapper(retriever,embeddings)

    if query:

        docs = retriever.get_relevant_documents(query)
        import pprint
        for doc in docs:
            print("***doc***")
            pprint.pprint(doc)
            print("***end***")

        prompt = get_prompt_local(False)
        llm = Ollama(model=model, base_url=base_url, temperature=temperature)
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        logging.info("Invoking chain...")
        result = rag_chain.invoke(query)
        #pprint.pprint(result)
        print(result)


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
