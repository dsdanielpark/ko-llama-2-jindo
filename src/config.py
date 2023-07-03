preprocessing_param = {
    'prefix_set': {"을 ", "를 ", "이 ", "가 ", "은 ", "는 ", "에 ", "으 ", "예, "},
    'dummy_file': 'dummy.json',
    'phrase_dictionary': {
        r'\b물론,\b': '물론이죠.',
        r'^(은 |는 )': ''
    }
}
