Langchain RAG
-------------
Import HTML or plain text content and build a corpus in chromadb.
Perform RAG (Retrieval Augmented Generation) from the corpus of content.

Prerequisites
-------------
* Tested on Linux with an NVIDIA CUDA-capable card with at least 6GB of RAM
* Tested on M2 Mac 32GB

Mac-specific setup instructions
----------------
```sh
$ brew install ollama
$ /path/to/bin/ollama serve # or: `brew services start ollama` in the background
```
Maybe in another terminal if necessary
```sh
$ ollama pull llama2:7b # get model
$ ollama run llama2:7b # test it runs
$ ollama pull nomic-embed-text # to generate our embeddings
```

Linux-specific instructions
------------------
We need a data directory for ollama persistent state
```sh
$ mkdir data
```
Test we have access to nvidia GPUs in docker
```sh
$ docker run --rm --gpus=all ubuntu nvidia-smi
```
Boot in docker
```sh
$ docker run -d --rm --gpus=all -v ./data:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```
Or Boot in docker-compose
```sh
$ docker-compose up -d ollama
```
Pull a LLM & test it
```sh
$ docker exec -it ollama ollama pull llama2:7b
$ docker exec -it ollama ollama run llama2:7b # test it out
$ docker exec -it ollama ollama pull nomic-embed-text # to generate our embeddings
```

General setup
-------------
```shell
$ cd /path/to/this/repo
$ python3 -m venv venv
$ . ./venv/bin/activate
$ which python3
/path/to/this/repo/venv/bin/python3
$ pip install -r requirements.txt
...
$ ./langchain-ollama-rag.py --help
usage: ./langchain-ollama-rag.py [-h] [--ingest] [--query QUERY] [--temperature TEMPERATURE] [--embeddings-model EMBEDDINGS_MODEL] [--ollama-embeddings-url OLLAMA_EMBEDDINGS_URL] [--generative-model GENERATIVE_MODEL] [--ollama-generation-url OLLAMA_GENERATION_URL] [--log-level LOG_LEVEL] [--sources]
                                 [--db-location DB_LOCATION]

options:
  -h, --help            show this help message and exit
  --ingest              read data locations line by line from STDIN and ingest
  --query QUERY         query to ask model
  --temperature TEMPERATURE
                        model temperature for query (default: 0)
  --embeddings-model EMBEDDINGS_MODEL
                        model used for creating the embeddings (default: nomic-embed-text)
  --ollama-embeddings-url OLLAMA_EMBEDDINGS_URL
                        URL for Ollama API for embeddings (default: http://localhost:11434)
  --generative-model GENERATIVE_MODEL
                        model used for generation (default: llama2:7b)
  --ollama-generation-url OLLAMA_GENERATION_URL
                        URL for Ollama API for generation (default: http://localhost:11434)
  --log-level LOG_LEVEL
                        Log threshold (default: warning)
  --sources             Show sources provided in context with query result
  --db-location DB_LOCATION
                        Location of the database (default: ./chroma_db/)

examples:
	ingest:	`echo https://tldp.org/HOWTO/html_single/8021X-HOWTO/ | ./langchain-ollama-rag.py --ingest`
	query:	`./langchain-ollama-rag.py --query="What is 802.1X?"`

$ cd ingest
$ ./fetch.sh # pulls down the Linux documentation project in plaintext, approx 35M
$ cd ..
$ ls ingest/*.txt | ./langchain-ollama-rag.py --ingest
...
$ ./langchain-ollama-rag.py --query="What is NIS+?" --sources
```
