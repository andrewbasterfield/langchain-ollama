Langchain RAG
-------------
Import HTML or plain text content and build a corpus in chromadb.
Perform RAG (Retrieval Augmented Generation) from the corpus of content.

Prerequisites
-------------
* Only tested with an NVIDIA CUDA card with at least 6GB of RAM
* [direnv](https://direnv.net/)

Mac-specific setup instructions
----------------
```sh
$ brew install ollama
$ /path/to/bin/ollama serve # or: `brew services start ollama` in the background
```
Maybe in another terminal
```sh
$ ollama pull llama2:7b # get model
$ ollama run llama2:7b # test it runs
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
```
General setup
-------------
```shell
$ cd /path/to/this/repo
direnv: error /path/to/this/repo/.envrc is blocked. Run `direnv allow` to approve its content
$ direnv allow
direnv: loading /path/to/this/repo/.envrc
direnv: export +ANONYMIZED_TELEMETRY +VIRTUAL_ENV ~PATH
$ which python
/path/to/this/repo/.direnv/python-3.X.Y/bin/python
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
Examples:
	ingest:	`echo https://tldp.org/HOWTO/html_single/8021X-HOWTO/ | ./langchain-ollama.py --ingest`
	query:	`./langchain-ollama.py --query="What is 802.1X?"`

```