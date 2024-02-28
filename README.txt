docker run --rm --gpus=all ubuntu nvidia-smi
docker run -d --rm --gpus=all -v ./data:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

docker exec -it ollama ollama pull llama2
docker exec -it ollama ollama pull neural-chat
docker exec -it ollama ollama pull mistral


docker exec -it ollama ollama run llama2
docker exec -it ollama ollama run neural-chat
docker exec -it ollama ollama run mistral
