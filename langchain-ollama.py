#!/usr/bin/env python3

# https://python.langchain.com/docs/use_cases/question_answering/quickstart
# https://python.langchain.com/docs/integrations/llms/ollama
# https://smith.langchain.com/hub/rlm/rag-prompt
# https://python.langchain.com/docs/integrations/vectorstores/chroma
# https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ollama.Ollama.html
from langchain_community.llms import Ollama

from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

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
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import bs4
    # Only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader(
        #web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        web_paths=(
          "https://tldp.org/HOWTO/html_single/8021X-HOWTO/",
          "https://tldp.org/HOWTO/html_single/ACPI-HOWTO/",
          "https://tldp.org/HOWTO/html_single/AI-Alife-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Assembly-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Astronomy-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Athlon-Powersaving-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Autodir-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Aviation-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Avr-Microcontrollers-in-Linux-Howto/",
          "https://tldp.org/HOWTO/html_single/Battery-Powered/",
          "https://tldp.org/HOWTO/html_single/Beowulf-HOWTO/",
          "https://tldp.org/HOWTO/html_single/BogoMips/",
          "https://tldp.org/HOWTO/html_single/BTTV/",
          "https://tldp.org/HOWTO/html_single/Cable-Modem/",
          "https://tldp.org/HOWTO/html_single/C++-dlopen/",
          "https://tldp.org/HOWTO/html_single/Cluster-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Coffee/",
          "https://tldp.org/HOWTO/html_single/Compaq-T1500-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Config-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Cryptoloop-HOWTO/",
          "https://tldp.org/HOWTO/html_single/DB2-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Debian-and-Windows-Shared-Printing/",
          "https://tldp.org/HOWTO/html_single/Debian-Binary-Package-Building-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Debian-Jigdo/",
          "https://tldp.org/HOWTO/html_single/Disk-Encryption-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Disk-on-Chip-HOWTO/",
          "https://tldp.org/HOWTO/html_single/DocBook-Demystification-HOWTO/",
          "https://tldp.org/HOWTO/html_single/DPT-Hardware-RAID-HOWTO/",
          "https://tldp.org/HOWTO/html_single/DVD-Playback-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Ecology-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Encrypted-Root-Filesystem-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Ethernet-Bridge-netfilter-HOWTO/",
          "https://tldp.org/HOWTO/html_single/FBB/",
          "https://tldp.org/HOWTO/html_single/Fedora-Multimedia-Installation-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Filesystems-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Finnish-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Flash-Memory-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Font-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Framebuffer-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Glibc-Install-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Hardware-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Home-Electrical-Control/",
          "https://tldp.org/HOWTO/html_single/HOWTO-INDEX/",
          "https://tldp.org/HOWTO/html_single/Howtos-with-LinuxDoc/",
          "https://tldp.org/HOWTO/html_single/Implement-Sys-Call-Linux-2.6-i386/",
          "https://tldp.org/HOWTO/html_single/Infrared-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Installfest-HOWTO/",
          "https://tldp.org/HOWTO/html_single/IP-Masquerade-HOWTO/",
          "https://tldp.org/HOWTO/html_single/IRC/",
          "https://tldp.org/HOWTO/html_single/Italian-HOWTO/",
          "https://tldp.org/HOWTO/html_single/K7s5a-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Kerberos-Infrastructure-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Large-Disk-HOWTO/",
          "https://tldp.org/HOWTO/html_single/LDAP-HOWTO/",
          "https://tldp.org/HOWTO/html_single/LDP-Reviewer-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Leased-Line/",
          "https://tldp.org/HOWTO/html_single/libdc1394-HOWTO/",
          "https://tldp.org/HOWTO/html_single/LILO/",
          "https://tldp.org/HOWTO/html_single/Linksys-Blue-Box-Router-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Linux-Complete-Backup-and-Recovery-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Linux-Gamers-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Linux-i386-Boot-Code-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Linux+IPv6-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Linux+WinNT/",
          "https://tldp.org/HOWTO/html_single/LVM-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Mail-User-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Masquerading-Simple-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Medicine-HOWTO/",
          "https://tldp.org/HOWTO/html_single/MMBase-Inst-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Mobile-IPv6-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Mock-Mainframe/",
          "https://tldp.org/HOWTO/html_single/Modem-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Module-HOWTO/",
          "https://tldp.org/HOWTO/html_single/MP3-CD-Burning/",
          "https://tldp.org/HOWTO/html_single/NC-HOWTO/",
          "https://tldp.org/HOWTO/html_single/NCURSES-Programming-HOWTO/",
          "https://tldp.org/HOWTO/html_single/NET3-4-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Networking-Overview-HOWTO/",
          "https://tldp.org/HOWTO/html_single/NLM-HOWTO/",
          "https://tldp.org/HOWTO/html_single/OLSR-IPv6-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Online-Troubleshooting-HOWTO/",
          "https://tldp.org/HOWTO/html_single/openMosix-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Oracle-9i-Fedora-3-Install-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Outlook-to-Unix-Mailbox/",
          "https://tldp.org/HOWTO/html_single/Parallel-Processing-HOWTO/",
          "https://tldp.org/HOWTO/html_single/PA-RISC-Linux-Boot-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Partition/",
          "https://tldp.org/HOWTO/html_single/Partition-Mass-Storage-Definitions-Naming-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Partition-Mass-Storage-Dummies-Linux-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Partition-Rescue/",
          "https://tldp.org/HOWTO/html_single/PCMCIA-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Plug-and-Play-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Postfix-Cyrus-Web-cyradm-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Qmail-ClamAV-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Quake-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Reading-List-HOWTO/",
          "https://tldp.org/HOWTO/html_single/RedHat-CD-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Reliance-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Samba-Authenticated-Gateway-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Scanner-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Scientific-Computing-with-GNU-Linux/",
          "https://tldp.org/HOWTO/html_single/SCSI-2.4-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Secure-BootCD-VPN-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Security-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Serial-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Software-RAID-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Software-Release-Practice-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Spam-Filtering-for-MX/",
          "https://tldp.org/HOWTO/html_single/SPARC-HOWTO/",
          "https://tldp.org/HOWTO/html_single/SquashFS-HOWTO/",
          "https://tldp.org/HOWTO/html_single/TCP-Keepalive-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Text-Terminal-HOWTO/",
          "https://tldp.org/HOWTO/html_single/TimePrecision-HOWTO/",
          "https://tldp.org/HOWTO/html_single/TimeSys-Linux-Install-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Traffic-Control-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Traffic-Control-tcng-HTB-HOWTO/",
          "https://tldp.org/HOWTO/html_single/TT-XFree86/",
          "https://tldp.org/HOWTO/html_single/Unix-and-Internet-Fundamentals-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Unix-Hardware-Buyer-HOWTO/",
          "https://tldp.org/HOWTO/html_single/UPS-HOWTO/",
          "https://tldp.org/HOWTO/html_single/User-Authentication-HOWTO/",
          "https://tldp.org/HOWTO/html_single/User-Group-HOWTO/",
          "https://tldp.org/HOWTO/html_single/VMS-to-Linux-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Webcam-HOWTO/",
          "https://tldp.org/HOWTO/html_single/WikiText-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Windows-LAN-Server-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Windows-Newsreaders-under-Linux-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Wireless-Link-sys-WPC11/",
          "https://tldp.org/HOWTO/html_single/Wireless-Sync-HOWTO/",
          "https://tldp.org/HOWTO/html_single/XDMCP-HOWTO/",
          "https://tldp.org/HOWTO/html_single/XFree86-Touch-Screen-HOWTO/",
          "https://tldp.org/HOWTO/html_single/XFree86-Video-Timings-HOWTO/",
          "https://tldp.org/HOWTO/html_single/XFree-Local-multi-user-HOWTO/",
          "https://tldp.org/HOWTO/html_single/Xinerama-HOWTO/",
          "https://tldp.org/HOWTO/html_single/XWindow-User-HOWTO/",
        ),
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

    #retriever = vectorstore.as_retriever()
    #retriever = vectorstore.as_retriever(search_type="mmr")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main(args):

    ingest = False
    query = False

    for arg in args:
        print(arg)
        if arg == "--ingest":
            ingest = True
        else:
            if arg == "--query":
                query = True
            else:
                print("Need one of --ingest or --query")
                exit(1)

    if (not ingest) and (not query):
        print("Need one of --ingest or --query")
        exit(1)


    if ingest:
        logging.warning("Building chroma from documents")
        context = get_retriever_chroma(documents=get_documents()) | format_docs
    else:
        logging.warning("Building chroma without documents")
        context = get_retriever_chroma() | format_docs

    if query:
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
        #result = rag_chain.invoke("How to create an LDAP database?")
        result = rag_chain.invoke("Please tell me about inline assembly")
        print(result)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
