# Korean ShareGPT DeepL Alpaca(KSDA)
After translating the contents of SharGPT with a DeepL translator, a Vicuna model fine-tuned for language translation (Korean-English example)



# Docker

### Build
By building with the following command, the built Docker image can be used with the name KSDV:latest.
```
docker build -t KSDV:latest docker/
```

### Docker Compose

By running the following command, the alpaca-lora service will run as a Docker container, and it can be accessed through the configured port (e.g., 7860).
```
docker-compose -f docker/docker-compose.yml up
```



# License
