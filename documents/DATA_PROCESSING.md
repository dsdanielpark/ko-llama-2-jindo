# Korean ShareGPT Alpaca
After using the translator to translate the contents of SharGPT, an alpaca model fine-tuned specifically for language translation tasks.

## DataSets
### `shargpt_deepl_cleaned_for_en_to_ko.json`

I have extracted translation pairs from the [junelee/sharegpt_deepl_ko](https://huggingface.co/datasets/junelee/sharegpt_deepl_ko) dataset using [src/preprocessing/data_gen.py](https://github.com/dsdanielpark/ko-sharegpt-deepl-alpaca/blob/main/src/preprocessing/data_gen.py) and made some corrections to the parts that were awkwardly interpreted in Korean using the algorithm in [src/preprocessing/data_cleaner.py](https://github.com/dsdanielpark/ko-sharegpt-deepl-alpaca/blob/main/src/preprocessing/data_cleaner.py).

- `ko_shargpt_deepl_cleaned_v1`: The dataset extracted and preprocessed using the algorithm to make only the Korean text more natural, as described above.
- `ko_shargpt_deepl_cleaned_v2`: Certain portions have been manually deleted or corrected by human inspection from v1.