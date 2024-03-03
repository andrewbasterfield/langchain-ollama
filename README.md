Prerequisites
-------------
* [direnv](https://direnv.net/)

Mac instructions
----------------
```sh
brew install ollama
/path/to/bin/ollama serve # or: `brew services start ollama` in the background
```
Maybe in another terminal
```sh
ollama pull zephyr # get zephyr llm
ollama run zephyr # test it runs
```

Linux instructions
------------------
We need a data directory for ollama persistent state
```sh
mkdir data
```
Test we have access to nvidia GPUs in docker
```sh
docker run --rm --gpus=all ubuntu nvidia-smi
```
Boot in docker
```sh
docker run -d --rm --gpus=all -v ./data:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```
Or Boot in docker-compose
```sh
docker-compose up -d ollama
```
Pull a LLM & test it
```sh
docker exec -it ollama ollama pull zephyr
docker exec -it ollama ollama run zephyr
```
