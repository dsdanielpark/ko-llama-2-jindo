# QnA
[PEFT](https://github.com/huggingface/peft), [Transformers](https://huggingface.co/docs/transformers/index), Lora, and QLora, along with their various settings, have numerous dependencies, including low-level drivers and operating systems that control hardware. As a result, they can cause various errors depending on the development environment. Therefore, it is strongly recommended to use Docker. However, developing automation that handles multiple GPUs and diverse settings can be challenging, and it is highly likely to be infeasible due to the rapid development of libraries. Therefore, it is necessary to refer to the provided helpful error messages and debug accordingly.

### `#1` For debugging
For debug transformers
```
transformers-cli env
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