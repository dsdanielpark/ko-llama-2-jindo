# Korean ShareGPT DeepL Alpaca(KSDA)
After translating the contents of SharGPT with a DeepL translator, a Vicuna model fine-tuned for language translation (Korean-English example)

## DataSets
### `shargpt_deepl_cleaned_for_en_to_ko.json`

I have extracted translation pairs from the [junelee/sharegpt_deepl_ko](https://huggingface.co/datasets/junelee/sharegpt_deepl_ko) dataset using src/preprocessing/data_gen.py and made some corrections to the parts that were awkwardly interpreted in Korean using the algorithm in src/preprocessing/data_cleaner.py.

- ko_shargpt_deepl_cleaned_v1: The dataset extracted and preprocessed using the algorithm to make only the Korean text more natural, as described above.
- ko_shargpt_deepl_cleaned_v2: Certain portions have been manually deleted or corrected by human inspection from v1.