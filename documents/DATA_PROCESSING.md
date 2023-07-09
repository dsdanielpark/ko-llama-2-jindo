# Ko Alpaca Lingo
The term "Lingo" in the model name "KoAlpacaLingo" refers to the ability to handle specialized terms and expressions in specific fields. This model represents a model that is proficient in understanding and translating the language characteristics of specific domains in Korean translation tasks.
Through this experiment, we aim to translate and fine-tune the data using Alpaca for the purpose of generating natural Korean language models (LLMs). Additionally, we will explore the possibility of fine-tuning with limited GPU resources.
After using the translator to translate the contents of SharGPT, an alpaca model fine-tuned specifically for language translation tasks.


## DataSets
### `shargpt_deepl_cleaned_for_en_to_ko.json`

I have extracted translation pairs from the [junelee/sharegpt_deepl_ko](https://huggingface.co/datasets/junelee/sharegpt_deepl_ko) dataset using [src/preprocessing/data_gen.py](https://github.com/dsdanielpark/ko-sharegpt-deepl-alpaca/blob/main/src/preprocessing/data_gen.py) and made some corrections to the parts that were awkwardly interpreted in Korean using the algorithm in [src/preprocessing/data_cleaner.py](https://github.com/dsdanielpark/ko-sharegpt-deepl-alpaca/blob/main/src/preprocessing/data_cleaner.py).

#### Raw data from [junelee/sharegpt_deepl_ko](https://huggingface.co/datasets/junelee/sharegpt_deepl_ko)
- `ko_sharegpt.json`
- `original_shargpt.json`

#### Modified and preprocessed dataset for language translation.
- `ko_shargpt_deepl_cleaned_v1.json`: The dataset extracted and preprocessed using the algorithm to make only the Korean text more natural, as described above.
- `ko_shargpt_deepl_cleaned_v2.json`: Certain portions have been manually deleted or corrected by human inspection from v1.


### `shargpt_google_cleaned_for_en_to_ko.json`
I have extracted translation pairs from the [dbdu/ShareGPT-74k-ko](https://huggingface.co/datasets/dbdu/ShareGPT-74k-ko/tree/main) dataset using 


#### Raw data from [dbdu/ShareGPT-74k-ko](https://huggingface.co/datasets/dbdu/ShareGPT-74k-ko/tree/main)


#### Modified and preprocessed dataset for language translation.
- `shargpt_google_cleaned_v1.json`: The dataset extracted and preprocessed using the algorithm to make only the Korean text more natural, as described above.
- `shargpt_google_cleaned_v2.json`: Certain portions have been manually deleted or corrected by human inspection from v1.