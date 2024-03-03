Linux instructions
------------------

We need a data directory for ollama persistent state
```sh
mkdir data
```
Test we have access to the GPU in docker
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
