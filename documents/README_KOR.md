[![](https://img.shields.io/badge/Language-English-lightgrey)](https://github.com/dsdanielpark/kor-sharegpt-deepl-alpaca) [![](https://img.shields.io/badge/%EC%96%B8%EC%96%B4-%ED%95%9C%EA%B5%AD%EC%96%B4-lightgrey)](https://github.com/dsdanielpark/kor-sharegpt-deepl-alpaca/blob/main/documents/README_KOR.md)



# Korean ShareGPT DeepL Alpaca(KSDA)
After translating the contents of SharGPT with a DeepL translator, a Vicuna model fine-tuned for language translation (Korean-English example)



# [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)

## Docker

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


## Official model weights
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


# [QLoRA](https://github.com/artidoro/qlora)


## Train Using Local Datasets
You can specify the path to your dataset using the `--dataset` argument. If the `--dataset_format` argument is not set, it will default to the Alpaca format. Here are a few examples:
Training with an alpaca format dataset:
```
python qlora.py --dataset="path/to/your/dataset"
```

Training with a self-instruct format dataset:
```
python qlora.py --dataset="path/to/your/dataset" --dataset_format="self-instruct"
```


## Tips 

### How can delete cached model weight
```
pip install huggingface_hub["cli"]
```
```
huggingface-cli delete-cache
```

# License
This project adheres to the licenses of the reference code and datasets used. It is the user's responsibility to check the licenses, and the user assumes all responsibilities regarding any licensing restrictions on this code. This repository is provided under the MIT license without any implied or explicit warranties.