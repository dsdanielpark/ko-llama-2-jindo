# QnA
[PEFT](https://github.com/huggingface/peft), [Transformers](https://huggingface.co/docs/transformers/index), Lora, and QLora, along with their various settings, have numerous dependencies, including low-level drivers and operating systems that control hardware. As a result, they can cause various errors depending on the development environment. Therefore, it is strongly recommended to use Docker. However, developing automation that handles multiple GPUs and diverse settings can be challenging, and it is highly likely to be infeasible due to the rapid development of libraries. Therefore, it is necessary to refer to the provided helpful error messages and debug accordingly.

### `#1` For debugging
For debug transformers
```
pip install huggingface_hub["cli"]
```
```
transformers-cli env
```
In my case

```
===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes

 and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
bin C:\Users\parkm\.conda\envs\miccai\lib\site-packages\bitsandbytes\libbitsandbytes_cpu.so
C:\Users\parkm\.conda\envs\miccai\lib\site-packages\bitsandbytes\cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
'NoneType' object has no attribute 'cadam32bit_grad_fp32'
CUDA SETUP: Loading binary C:\Users\parkm\.conda\envs\miccai\lib\site-packages\bitsandbytes\libbitsandbytes_cpu.so...
argument of type 'WindowsPath' is not iterable
WARNING:tensorflow:From C:\Users\parkm\.conda\envs\miccai\lib\site-packages\transformers\commands\env.py:63: is_gpu_available (from tensorflow.python.framework.test_util) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.config.list_physical_devices('GPU')` instead.
2023-07-10 02:48:30.125941: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-07-10 02:48:31.226114: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /device:GPU:0 with 5471 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3070 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6

Copy-and-paste the text below in your GitHub issue and FILL OUT the two last points.

- `transformers` version: 4.31.0.dev0
- Platform: Windows-10-10.0.22621-SP0
- Python version: 3.8.5
- Huggingface_hub version: 0.15.1
- Safetensors version: 0.3.1
- PyTorch version (GPU?): 2.0.1+cpu (False)
- Tensorflow version (GPU?): 2.10.1 (True)
- Flax version (CPU?/GPU?/TPU?): not installed (NA)
- Jax version: not installed
- JaxLib version: not installed
- Using GPU in script?: <fill in>
- Using distributed or parallel set-up in script?: <fill in>
```

<br>


### `#2` How can delete cached model weight
```
pip install huggingface_hub["cli"]
```
```
huggingface-cli delete-cache
```

<br>

### `#3` ValueError: gpt_neox.embed_in.weight doesn't have any device set.
Please confirm if you are trying to download and load the config and the appropriate foundation model using the from_pretrained method. Note that the config may vary for different foundation models.

<br>

### `#4` If you can NOT use GPU
Error: You can NOT use GPU
```python
import torch
torch.cuda.is_available()
```
```
raise ValueError(f"{param_name} doesn't have any device set.")
```
https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
```

Off-load was used to train with limited resources. Please refer to the following [Link](https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu) and [Git hash](https://github.com/dsdanielpark/ko-sharegpt-alpaca/commit/0c40cacadc724034ed578aaaae06d02c625be8af) for partial revisions. 

<br>

### `#5` Install bitsandbytes
Error: In addition to CUDA, the bitsandbytes library is required for training with Lora. However, if it fails to install properly on Windows, this issue may occur.
```
RuntimeError:
        CUDA Setup failed despite GPU being available. Please run the following command to get more information:

        python -m bitsandbytes

        Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
        to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
        and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues
```

Solution
https://github.com/TimDettmers/bitsandbytes/issues/175
```
pip install bitsandbytes-windows
```


### `#6` AttributeError: module 'bitsandbytes.nn' has no attribute 'Linear4bit'
https://github.com/oobabooga/text-generation-webui/issues/2228#issuecomment-1556002597

```
python -m pip install git+https://github.com/huggingface/peft@27af2198225cbb9e049f548440f2bd0fba2204aa --force-reinstall --no-deps
```

### `#7` prepare_model_for_kbit_training
https://github.com/huggingface/peft/issues/108
```
from peft import (
ImportError: cannot import name 'prepare_model_for_kbit_training' from 'peft' (C:\Users\parkm\AppData\Roaming\Python\Python39\site-packages\peft\__init__.py) 
```
```
pip install git+https://github.com/younesbelkada/transformers.git@fix-int8-conversion
```