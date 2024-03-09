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
$ ./langchain-ollama.py --help
Usage: ./langchain-ollama.py
	--ingest				read data locations line by line from STDIN and ingest
	--query=<query>				query to ask model
	--temperature=N				model temperature for query (default: 0)
	--model=<model>				model (default: "llama2:7b")
	--base-url=<base-url>			URL for Ollama API (default: "http://localhost:11434")
	--log-level=<debug|info|warning>	Log level (default: "info")
	--langchain-verbose			enable langchain verbose mode
	--langchain-debug			enable langchain debug mode
	--sources				print locations of sources used as context
Examples:
	ingest:	`echo https://tldp.org/HOWTO/html_single/8021X-HOWTO/ | ./langchain-ollama.py --ingest`
	query:	`./langchain-ollama.py --query="What is 802.1X?"`
$ cd training
$ ./fetch.sh # pulls down the Linux documentation project in plaintext, approx 35M
$ cd ..
$ ls training/*.txt | ./langchain-ollama.py --ingest
...
$ ./langchain-ollama.py --query="What is NIS+?" --sources
```
