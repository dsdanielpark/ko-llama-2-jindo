preprocessing_param = {
    'target_json_file_path': r'C:\Users\parkm\Desktop\llm\SharGPT-DeepL-Vicuna-Kor\notebooks\output\output_1.json',
    'prefix_set': {"을 ", "를 ", "이 ", "가 ", "은 ", "는 ", "에 ", "으 ", "예, "},
    'dummy_file_path': 'dummy.json',
    'phrase_dictionary': {
        r'\b물론,\b': '물론이죠.',
        r'^(은 |는 )': ''
    },
    'remove_target_word_set' : {'sure!', 'great!'}
}
