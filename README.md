Development Status :: 2 - Pre-Alpha <br>
*Copyright (c) 2023 MinWoo Park*


After considering various factors, refactoring and re-documentation will be done before the release.


##### [Jindo](https://github.com/dsdanielpark/ko-guanaco-jindo)(sLLM) is a preprocessing LLM (Language Model) designed to refine various LLM datasets

# Korean LLaMA2 Jindo [![](https://img.shields.io/badge/Language-English-lightgrey)](https://github.com/dsdanielpark/ko-alpaca-jindo) [![](https://img.shields.io/badge/%EC%96%B8%EC%96%B4-%ED%95%9C%EA%B5%AD%EC%96%B4-lightgrey)](https://github.com/dsdanielpark/ko-alpaca-jindo/blob/main/documents/README_KO.md)


[Korean LLaMA2 Jindo](https://github.com/dsdanielpark/ko-llama-2-jindo) is a preprocessing LLM (Language Model) designed to refine various LLM datasets such as alpaca, falcon, guanaco, and wizard. Korean LLaMA2 Jindo is trained based on the [Korean-Open-LLM-Datasets (KOLD) Chain](https://github.com/dsdanielpark/korean-open-llm-datasets-chain), which is a pipeline configured to utilize high-quality Korean datasets. It employs the same training method as Guanaco. <br>
[KOLANI](https://github.com/dsdanielpark/KOLANI) is a fine-tuned LLaMA2 model using the preprocessed data from [Korean LLaMA2 Jindo](https://github.com/dsdanielpark/ko-llama-2-jindo) and various architectures based on [LLaMA2](https://ai.meta.com/llama/) from [Meta AI](https://ai.meta.com/). <br>
The `jindo` in `ko-llama2-jindo` refers to the Korean dog breed, Jindo. The term `jindo` signifies a language model specialized in preprocessing Korean language datasets, similar to `lingo`, and aims to be a lightweight and fast model tailored for processing Korean datasets for LLM(Large Language Model, "LLM") training. 
Through this experiment, we aim to translate and fine-tune the LLM models for the purpose of generating Korean LLMs. Additionally, we will explore the possibility of fine-tuning with limited GPU resources.

## Datasets
Data Pipeline: Please check the [data processing](https://github.com/dsdanielpark/ko-alpaca-jindo/blob/main/documents/DATA_PROCESSING.md) method and the approach for [revisions](https://github.com/dsdanielpark/ko-alpaca-jindo/blob/main/documents/DATA_REVISION.md). Please check the following repository for high-quality large-scale datasets in alpaca format for LLaMA finetuning: [Korean Open LLM Datasets(KOLD) Chain](https://github.com/dsdanielpark/korean-open-llm-datasets-chain).

## Foundation Model
1. [LLaMA](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) from [Meta AI](https://ai.meta.com/)
If you want to download the official model, fill this [official request form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform) and wait.
Delta weights over the original Llama model is released under [CC BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). <br>
2. [LLaMA2](https://ai.meta.com/llama/) from [Meta AI](https://ai.meta.com/)
3. Reference model: 
    [Guanaco](https://huggingface.co/JosephusCheung/Guanaco) is an LLM based on the QLoRA 4-bit fine-tuning method developed by Tim Dettmers et. al. in the UW NLP group. It achieves 99% ChatGPT performance on the Vicuna benchmark.
    - Uses LoRA fine-tuning method
    - Fine-tunes up to a 65B parameter model on a 48GB GPU without performance loss compared to 16-bit models
    - Initial Release: 2023-05-23 <br>



<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>


# Appendix
## Alpaca
Alpaca 7B is a model derived from the [LLaMA 7B model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) of [Meta](https://about.meta.com/), fine-tuned using 52K instruction-following demonstrations. In our initial evaluation, Alpaca demonstrates similar qualitative behavior to OpenAI's text-davinci-003 model, despite being significantly smaller and more affordable to reproduce (<$600). You can find the code release on [LLaMA GitHub repository](https://github.com/facebookresearch/llama).


## LoRA and QLoRA
Applying a method to fine-tune only a subset of LLM weights and optimize GPU utilization in the [Stanford Alpaca.](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- [*LoRA*](https://github.com/microsoft/LoRA)(*Lo*w-*R*ank *A*daptation of Large Language Models) is a method for adapting large-scale pre-trained language models to specific tasks by introducing trainable rank decomposition matrices, significantly reducing the number of trainable parameters and GPU memory requirements, while maintaining or surpassing the performance of traditional fine-tuning approaches on various models.
- [*QLoRA*](https://github.com/artidoro/qlora)(*Q*uantized *Lo*w-*R*ank *A*daptation of Large Language Models) is an efficient finetuning approach that reduces memory usage while maintaining high performance, enabling the finetuning of large language models on a single GPU and achieving state-of-the-art results on various benchmarks.

![](assets/qlora.png)
*[https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)*

<br>

## [Alpaca LoRA](https://github.com/tloen/alpaca-lora)
Includes code for replicating Stanford Alpaca's results using low-rank adaptation (LoRA). They offer an Instruct model of similar quality to `text-davinci-003` that can run on a Raspberry Pi for research purposes. The code can be easily extended to larger models like 13b, 30b, and 65b.
In addition to the training code, which can run within hours on a single RTX 4090, they provide a script for downloading and performing inference on the foundation model and LoRA, along with the LoRA weights. They utilize Hugging Face's PEFT and Tim Dettmers' bitsandbytes for efficient and cost-effective fine-tuning.
The LoRA model produces outputs comparable to the Stanford Alpaca model without hyperparameter tuning (see included outputs). Further tuning may lead to better performance, so tey encourage interested users to try it out and share their results.

### Docker Build
By building with the following command, the built Docker image can be used with the name `ko-alpaca-lingo:latest`.
```shell
docker build -t ko-alpaca-lingo:latest docker/
```

### Docker Compose

By running the following command, the alpaca-lora service will run as a Docker container, and it can be accessed through the configured port (e.g., 7860).
```shell
docker-compose -f docker/docker-compose.yml up
```


### Official Weights Alpaca Lora
The most recent `Official Alpaca LoRA` adapter available at tloen/alpaca-lora-7b was trained on March 26 with the following command:

### Fine-tuning Alpaca Lora
```cmd
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

```cmd
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


### Generate (Inference) Alpaca Lora

```shell
python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'
```
Or
```shell
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

```cmd
python generate.py ^
    --load_8bit ^
    --base_model 'decapoda-research/llama-7b-hf' ^
    --lora_weights 'tloen/alpaca-lora-7b'
```
Or
```cmd
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

## Alpaca [QLoRA](https://github.com/artidoro/qlora)
QLoRA is an efficient finetuning approach that allows for finetuning a 65B parameter model on a single 48GB GPU while maintaining full 16-bit finetuning task performance. It utilizes 4-bit quantization and Low Rank Adapters (LoRA) to reduce memory usage. The best model, Guanaco, surpasses previous models on the Vicuna benchmark, achieving 99.3% of ChatGPT's performance level with just 24 hours of finetuning on a single GPU. QLoRA introduces innovations such as 4-bit NormalFloat (NF4) data type, Double Quantization, and Paged Optimizers to save memory without sacrificing performance. Over 1,000 models were finetuned using QLoRA, demonstrating state-of-the-art results across various datasets and model types. 


### Fine-tuning Alpaca QLoRA
You can specify the path to your dataset using the `--dataset` argument. If the `--dataset_format` argument is not set, it will default to the Alpaca format. Here are a few examples:
Training with an alpaca format dataset:
```shell
python qlora.py --dataset="./data/ko_shargpt_deepl_cleaned_v1.json"
```
Off-load was used to train with limited resources. Please refer to the following [Link](https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu) and [Git hash](https://github.com/dsdanielpark/ko-sharegpt-alpaca/commit/0c40cacadc724034ed578aaaae06d02c625be8af) for partial revisions. 


Training with a self-instruct format dataset:
```shell
python qlora.py --dataset="./data/ko_shargpt_deepl_cleaned_v1.json" --dataset_format="self-instruct"
```

<br>


## [Vicuna](https://huggingface.co/lmsys) using [FastChat](https://github.com/lm-sys/FastChat)
An open source chatbot impressing GPT-4 with 90% Chat-GPT quality. 

<br>

## [Falcon](https://huggingface.co/tiiuae/falcon-7b) from [Hugging Face](https://huggingface.co/)
Falcon-7B and Falcon-40B have been trained on 1.5 trillion and 1 trillion tokens respectively, in line with modern models optimising for inference. The key ingredient for the high quality of the Falcon models is their training data, predominantly based (>80%) on [Falcon RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) — a novel massive web dataset based on CommonCrawl. 

<br>

## [Orca](https://huggingface.co/papers/2306.02707)
Orca presents a novel approach to training large language models, combining progressive learning and teacher assistance to enhance imitation learning.

<br>

## [Long LLaMA](https://github.com/CStanKonrad/long_llama)
LongLLaMA is built upon the foundation of OpenLLaMA and fine-tuned using the Focused Transformer (FoT) method.
<br><br>


## [GPTQ](https://github.com/IST-DASLab/gptq)
GPTQ is the state-of-the-art one-shot weight quantization method. This code is built upon [GPTQ](https://github.com/IST-DASLab/gptq), [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa), [GPTQ-triton](https://github.com/fpgaminer/GPTQ-triton), [Auto-GPTQ](https://github.com/PanQiWei/AutoGPTQ). 

```shell
conda create --name gptq python=3.9 -y
conda activate gptq
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# Via pypi
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

cd src/gptq
pip install -r requirements.txt 
python setup_cuda.py install
```

### Command example for LLaMA
```python
import locale
locale.getpreferredencoding = lambda: "UTF-8"
```
```shell
# Convert LLaMA to hf
python convert_llama_weights_to_hf.py --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir ./llama-hf

# Benchmark language generation with 4-bit LLaMA-7B:

# Save compressed model
CUDA_VISIBLE_DEVICES=0 python llama.py ${MODEL_DIR} c4 --wbits 4 --true-sequential --act-order --groupsize 128 --save llama7b-4bit-128g.pt

# Or save compressed `.safetensors` model
CUDA_VISIBLE_DEVICES=0 python llama.py ${MODEL_DIR} c4 --wbits 4 --true-sequential --act-order --groupsize 128 --save_safetensors llama7b-4bit-128g.safetensors

# Benchmark generating a 2048 token sequence with the saved model
CUDA_VISIBLE_DEVICES=0 python llama.py ${MODEL_DIR} c4 --wbits 4 --groupsize 128 --load llama7b-4bit-128g.pt --benchmark 2048 --check

# Benchmark FP16 baseline, note that the model will be split across all listed GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python llama.py ${MODEL_DIR} c4 --benchmark 2048 --check

# Model inference with the saved model
CUDA_VISIBLE_DEVICES=0 python llama_inference.py ${MODEL_DIR} --wbits 4 --groupsize 128 --load llama7b-4bit-128g.pt --text "this is llama"

# Model inference with the saved model using safetensors loaded direct to gpu
CUDA_VISIBLE_DEVICES=0 python llama_inference.py ${MODEL_DIR} --wbits 4 --groupsize 128 --load llama7b-4bit-128g.safetensors --text "this is llama" --device=0

# Model inference with the saved model with offload(This is very slow).
CUDA_VISIBLE_DEVICES=0 python llama_inference_offload.py ${MODEL_DIR} --wbits 4 --groupsize 128 --load llama7b-4bit-128g.pt --text "this is llama" --pre_layer 16
It takes about 180 seconds to generate 45 tokens(5->50 tokens) on single RTX3090 based on LLaMa-65B. pre_layer is set to 50.
```

### Benchmark performance for FC2 layer of LLaMA-7B
```shell
CUDA_VISIBLE_DEVICES=0 python test_kernel.py
```

### Quantization

Most quantization packages have been developed based on the Linux OS and may not be compatible with Windows.
Basically, 4-bit quantization and 128 groupsize are recommended. You can also export quantization parameters with toml+numpy format.

Command pygptq for llama1
```shell
CUDA_VISIBLE_DEVICES=0 python llama.py ${MODEL_DIR} c4 --wbits 4 --true-sequential --act-order --groupsize 128 --quant-directory ${TOML_DIR}
```
Command gptq for `ko-llama-2-jindo-7b-instruct`
```
python bloom.py danielpark/ko-llama-2-jindo-7b-instruct wikitext2 --wbits 8 --groupsize 128 --save danielpark/ko-llama-2-jindo-7b-instruct-4bit-128g-gptq
```

### alpaca fine-tuning using GPTQ
https://github.com/PanQiWei/AutoGPTQ/blob/main/examples/quantization/quant_with_alpaca.py


<br>


## Quantinization using [llama.cpp](https://github.com/ggerganov/llama.cpp) to [GGML](https://github.com/ggerganov/ggml)
For more details, visit [here](https://huggingface.co/danielpark/ko-llama-2-jindo-7b-instruct-ggml).

```
cd llama.cpp

python3 -m pip install -r requirements.txt
python3 convert.py models/7B/

    # [Optional] for models using BPE tokenizers 
    python convert.py models/7B/ --vocabtype bpe

# quantize the model to 4-bits (using q4_0 method)
./quantize ./models/7B/ggml-model-f16.bin ./models/7B/ggml-model-q4_0.bin q4_0

# run the inference
./main -m ./models/7B/ggml-model-q4_0.bin -n 128
```

Windows-x86
Make ggml formatted weight using Q5_K_M method
```
quantize.exe jindo-7b-instruct.ggmlv3.f16.bin jindo-7b-instruct.ggmlv3.q5_k_m.bin q5_k_m
```

## GGML Formate Quantinization

<details>
<summary> See more... </summary>
  
### Quant Types


#### GGML Quant Type

| Quantization Type | Description                                                                                           | Bits per Weight (bpw) |
|-------------------|-------------------------------------------------------------------------------------------------------|-----------------------|
| GGML_TYPE_Q2_K    | "type-1" 2-bit quantization in super-blocks containing 16 blocks, each block having 16 weights. Block scales and mins are quantized with 4 bits.           | 2.5625                |
| GGML_TYPE_Q3_K    | "type-0" 3-bit quantization in super-blocks containing 16 blocks, each block having 16 weights. Scales are quantized with 6 bits.                           | 3.4375                |
| GGML_TYPE_Q4_K    | "type-1" 4-bit quantization in super-blocks containing 8 blocks, each block having 32 weights. Scales and mins are quantized with 6 bits.                    | 4.5                   |
| GGML_TYPE_Q5_K    | "type-1" 5-bit quantization. Same super-block structure as GGML_TYPE_Q4_K resulting in 5.5 bpw.      | 5.5                   |
| GGML_TYPE_Q6_K    | "type-0" 6-bit quantization. Super-blocks with 16 blocks, each block having 16 weights. Scales are quantized with 8 bits.                                 | 6.5625                |
| GGML_TYPE_Q8_K    | "type-0" 8-bit quantization. Only used for quantizing intermediate results. Block size is 256. All 2-6 bit dot products are implemented for this quantization type. | Not specified         |

#### Method Description

| Model | Description                                      | Recommendation      |
|-------|--------------------------------------------------|---------------------|
| Q4_0  | Small, very high-quality loss                    | Legacy, prefer Q3_K_M |
| Q4_1  | Small, substantial quality loss                  | Legacy, prefer Q3_K_L |
| Q5_0  | Medium, balanced quality                         | Legacy, prefer Q4_K_M |
| Q5_1  | Medium, low quality loss                         | Legacy, prefer Q5_K_M |
| Q2_K  | Smallest, extreme quality loss                   | Not recommended     |
| Q3_K  | Alias for Q3_K_M                                 |                     |
| Q3_K_S| Very small, very high-quality loss               |                     |
| Q3_K_M| Very small, very high-quality loss               |                     |
| Q3_K_L| Small, substantial quality loss                  |                     |
| Q4_K  | Alias for Q4_K_M                                 |                     |
| Q4_K_S| Small, significant quality loss                  |                     |
| Q4_K_M| Medium, balanced quality                         | Recommended        |
| Q5_K  | Alias for Q5_K_M                                 |                     |
| Q5_K_S| Large, low quality loss                          | Recommended        |
| Q5_K_M| Large, very low quality loss                     | Recommended        |
| Q6_K  | Very large, extremely low quality loss           |                     |
| Q8_0  | Very large, extremely low quality loss           | Not recommended     |
| F16   | Extremely large, virtually no quality loss       | Not recommended     |
| F32   | Absolutely huge, lossless                        | Not recommended     |



### Performance
#### LLaMA 2 / 7B
| name  | +ppl   | +ppl 13b to 7b % | size  | size 16bit % | +ppl per -1G |
|-------|--------|------------------|-------|--------------|--------------|
| q2_k  | 0.8698 | 133.344%         | 2.67GB | 20.54%       | 0.084201     |
| q3_ks | 0.5505 | 84.394%          | 2.75GB | 21.15%       | 0.053707     |
| q3_km | 0.2437 | 37.360%          | 3.06GB | 23.54%       | 0.024517     |
| q3_kl | 0.1803 | 27.641%          | 3.35GB | 25.77%       | 0.018684     |
| q4_0  | 0.2499 | 38.311%          | 3.50GB | 26.92%       | 0.026305     |
| q4_1  | 0.1846 | 28.300%          | 3.90GB | 30.00%       | 0.020286     |
| q4_ks | 0.1149 | 17.615%          | 3.56GB | 27.38%       | 0.012172     |
| q4_km | 0.0535 | 8.202%           | 3.80GB | 29.23%       | 0.005815     |
| q5_0  | 0.0796 | 12.203%          | 4.30GB | 33.08%       | 0.009149     |
| q5_1  | 0.0415 | 6.362%           | 4.70GB | 36.15%       | 0.005000     |
| q5_ks | 0.0353 | 5.412%           | 4.33GB | 33.31%       | 0.004072     |
| q5_km | 0.0142 | 2.177%           | 4.45GB | 34.23%       | 0.001661     |
| q6_k  | 0.0044 | 0.675%           | 5.15GB | 39.62%       | 0.000561     |
| q8_0  | 0.0004 | 0.061%           | 6.70GB | 51.54%       | 0.000063     |

#### LLaMA 2 / 13B
| name  | +ppl   | +ppl 13b to 7b % | size  | size 16bit % | +ppl per -1G |
|-------|--------|------------------|-------|--------------|--------------|
| q2_k  | 0.6002 | 92.013%          | 5.13GB | 20.52%       | 0.030206     |
| q3_ks | 0.3490 | 53.503%          | 5.27GB | 21.08%       | 0.017689     |
| q3_km | 0.1955 | 29.971%          | 5.88GB | 23.52%       | 0.010225     |
| q3_kl | 0.1520 | 23.302%          | 6.45GB | 25.80%       | 0.008194     |
| q4_0  | 0.1317 | 20.190%          | 6.80GB | 27.20%       | 0.007236     |
| q4_1  | 0.1065 | 16.327%          | 7.60GB | 30.40%       | 0.006121     |
| q4_ks | 0.0861 | 13.199%          | 6.80GB | 27.20%       | 0.004731     |
| q4_km | 0.0459 | 7.037%           | 7.32GB | 29.28%       | 0.002596     |
| q5_0  | 0.0313 | 4.798%           | 8.30GB | 33.20%       | 0.001874     |
| q5_1  | 0.0163 | 2.499%           | 9.10GB | 36.40%       | 0.001025     |
| q5_ks | 0.0242 | 3.710%           | 8.36GB | 33.44%       | 0.001454     |
| q5_km | 0.0095 | 1.456%           | 8.60GB | 34.40%       | 0.000579     |
| q6_k  | 0.0025 | 0.383%           | 9.95GB | 39.80%       | 0.000166     |
| q8_0  | 0.0005 | 0.077%           | 13.00GB| 52.00%       | 0.000042     |

</details>




# [QnA](https://github.com/dsdanielpark/ko-alpaca-lingo/blob/main/documents/QNA.md)
I have compiled some common and encountered errors, along with their solutions. I hope this will be helpful to many researchers. Before creating an issue, please search for it first. If you find an error along with its solution, I would appreciate it if you could provide a pull request.

# [KOLANI](https://github.com/dsdanielpark/KOLANI)
Most open-source LLM models are derived from the open-source LLM weights of Meta, called LLaMA. The Python implementation of LLaMA and fine-tuning it for the Korean language can be found in the KOLANI(Korean LLM based on LLaMA2 Natural Inference Model,고라니) project.

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
- Core maintainer: [Daniel Park, South Korea](https://github.com/dsdanielpark) <br>
- E-mail: parkminwoo1991@gmail.com <br>


### How can delete cached model weight
```
pip install huggingface_hub["cli"]
```
```
huggingface-cli delete-cache
```

# Reference 
[1] https://github.com/tloen/alpaca-lora <br>
[2] https://github.com/huggingface/peft <br>
[3] https://github.com/artidoro/qlora <br>
[4] https://huggingface.co/timdettmers/qlora-alpaca-7b <br>
[5] https://github.com/artidoro/qlora <br>
[6] https://arxiv.org/abs/2305.14314 <br>
