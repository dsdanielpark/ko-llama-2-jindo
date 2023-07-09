[![](https://img.shields.io/badge/Language-English-lightgrey)](https://github.com/dsdanielpark/ko-sharegpt-deepl-alpaca) [![](https://img.shields.io/badge/%EC%96%B8%EC%96%B4-%ED%95%9C%EA%B5%AD%EC%96%B4-lightgrey)](https://github.com/dsdanielpark/ko-sharegpt-deepl-alpaca/blob/main/documents/README_KO.md)

##### Specializes in some task in Korean, aiming to generate natural language models using Alpaca and explore limited GPU fine-tuning.

# Korean Alpaca Lingo

The term `Lingo` in the model name `KoAlpacaLingo` refers to the ability to handle specialized terms and expressions in specific fields. This model represents a model that is proficient in understanding and translating the language characteristics of specific domains in Korean translation tasks.
Through this experiment, we aim to translate and fine-tune the data using Alpaca for the purpose of generating natural Korean language models (LLMs). Additionally, we will explore the possibility of fine-tuning with limited GPU resources.
After using the translator to translate the contents of SharGPT, an alpaca model fine-tuned specifically for language translation tasks.


## Datasets
Please check the [data processing](https://github.com/dsdanielpark/ko-sharegpt-deepl-alpaca/blob/main/documents/DATA_PROCESSING.md) method and the approach for [revisions](https://github.com/dsdanielpark/ko-sharegpt-deepl-alpaca/blob/main/documents/DATA_REVISION.md).


<br><br>

# [Alpaca Lora](https://github.com/tloen/alpaca-lora)

### Docker Build
By building with the following command, the built Docker image can be used with the name `ko-alpaca-lingo:latest`.
```
docker build -t ko-alpaca-lingo:latest docker/
```

### Docker Compose

By running the following command, the alpaca-lora service will run as a Docker container, and it can be accessed through the configured port (e.g., 7860).
```
docker-compose -f docker/docker-compose.yml up
```


### Official Weights Alpaca Lora
The most recent `Official Alpaca LoRA` adapter available at tloen/alpaca-lora-7b was trained on March 26 with the following command:

## Fine-tuning Alpaca Lora
```
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'danielpark/ko_shargpt_deepl_cleaned_v1' \
    --output_dir './lora-alpaca' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length
```

<details>
<summary> Windows CMD</summary>

```
python finetune.py ^
    --base_model 'decapoda-research/llama-7b-hf' ^
    --data_path 'danielpark/ko_shargpt_deepl_cleaned_v1' ^
    --output_dir './lora-alpaca' ^
    --batch_size 128 ^
    --micro_batch_size 4 ^
    --num_epochs 3 ^
    --learning_rate 1e-4 ^
    --cutoff_len 512 ^
    --val_set_size 2000 ^
    --lora_r 8 ^
    --lora_alpha 16 ^
    --lora_dropout 0.05 ^
    --lora_target_modules '[q_proj,v_proj]' ^
    --train_on_inputs ^
    --group_by_length

```

</details>


## Generate (Inference) Alpaca Lora

```
python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```
Or
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
python generate.py ^
    --load_8bit ^
    --base_model 'decapoda-research/llama-7b-hf' ^
    --lora_weights 'tloen/alpaca-lora-7b'
```
Or
```
python finetune.py ^
    --base_model='decapoda-research/llama-7b-hf' ^
    --num_epochs=10 ^
    --cutoff_len=512 ^
    --group_by_length ^
    --output_dir='./lora-alpaca' ^
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' ^
    --lora_r=16 ^
    --micro_batch_size=8
```

</details>


<br><br>

# Alpaca [QLoRA](https://github.com/artidoro/qlora)


## Fine-tuning Alpaca QLoRA
You can specify the path to your dataset using the `--dataset` argument. If the `--dataset_format` argument is not set, it will default to the Alpaca format. Here are a few examples:
Training with an alpaca format dataset:
```
python qlora.py --dataset="./data/ko_shargpt_deepl_cleaned_v1.json"
```
Off-load was used to train with limited resources. Please refer to the following [Link](https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu) and [Git hash](https://github.com/dsdanielpark/ko-sharegpt-alpaca/commit/0c40cacadc724034ed578aaaae06d02c625be8af) for partial revisions. 


Training with a self-instruct format dataset:
```
python qlora.py --dataset="./data/ko_shargpt_deepl_cleaned_v1.json" --dataset_format="self-instruct"
```

<br>

## Appendix 

### How can delete cached model weight
```
pip install huggingface_hub["cli"]
```
```
huggingface-cli delete-cache
```

# [QnA](https://github.com/dsdanielpark/ko-alpaca-lingo/blob/main/documents/QNA.md)

# License
I hold no legal responsibility; <br>
This project adheres to the licenses of the reference code and datasets used. It is the user's responsibility to check the licenses, and the user assumes all responsibilities regarding any licensing restrictions on this code. This repository is provided under the MIT license without any implied or explicit warranties.

```
The MIT License (MIT)

Copyright (c) 2023 Minwoo Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


## Bugs and Issues
Sincerely grateful for any reports on new features or bugs. Your valuable feedback on the code is highly appreciated.

## Contacts
- Core maintainer: [Daniel Park, South Korea](https://github.com/DSDanielPark) <br>
- E-mail: parkminwoo1991@gmail.com <br>

## Reference 


