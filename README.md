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


## Official weights
The most recent "official" Alpaca-LoRA adapter available at tloen/alpaca-lora-7b was trained on March 26 with the following command:

```
python finetune.py \
    --base_model='decapoda-research/llama-7b-hf' \
    --num_epochs=10 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='./lora-alpaca' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8
```

<details>
<summary> Windows CMD</summary>
```
python finetune.py ^
    --base_model="decapoda-research/llama-7b-hf" ^
    --num_epochs=10 ^
    --cutoff_len=512 ^
    --group_by_length ^
    --output_dir="./lora-alpaca" ^
    --lora_target_modules="[q_proj,k_proj,v_proj,o_proj]" ^
    --lora_r=16 ^
    --micro_batch_size=8

```
</details>

### How can delete cached model weight
```
pip install huggingface_hub["cli"]
```
```
huggingface-cli delete-cache
```

# License
This project adheres to the licenses of the reference code and datasets used. It is the user's responsibility to check the licenses, and the user assumes all responsibilities regarding any licensing restrictions on this code. This repository is provided under the MIT license without any implied or explicit warranties.