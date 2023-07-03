import json
import concurrent.futures
import re
import config as CONF




class DataCleaner:
    def __init__(self, target_file, steps):
        self.target_file = target_file
        self.steps = steps

    def remove_specific_words(self, data):
        word_set = {'sure!', 'great!'}
        return [d for d in data if not any(word in d['input'] for word in word_set)]

    def remove_short_fields(self, data):
        return [d for d in data if (input_len := len(d['input'])) > 4 and (output_len := len(d['output'])) > 4]

    def replace_phrase(self, data):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self.replace_sure_translation, data))

    def replace_sure_translation(self, d):
        for pattern, replacement in CONF.PHRASE_DICTIONARY.items():
            d['output'] = re.sub(pattern, replacement, d['output'])
        return d

    def delete_error_korean_prefix(self, d):
        d['output'] = re.sub(r'^(은 |는 )', '', d['output'])
        return d

    def replace_prefix(self, data):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self.delete_error_korean_prefix, data))

    def do_not_translate_code_snippet(self, d):
        input_text = d['input']
        output_text = d['output']

        if '```' in input_text and '```' in output_text:
            start_index = input_text.find('```') + 3
            end_index = input_text.find('```', start_index)
            replace_text = input_text[start_index:end_index]

            d['output'] = output_text.replace('```', f' ```{replace_text}```')

        return d

    def replace_common_phrase(self, data):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self.do_not_translate_code_snippet, data))

    def remove_duplicate_input(self, data):
        unique_inputs = set()
        result = []

        for d in data:
            input_text = d['input']
            if input_text not in unique_inputs:
                unique_inputs.add(input_text)
                result.append(d)

        return result

    def replace_output_prefix(self, data):
        dummy_data = [d['output'] for d in data if d['output'].startswith(tuple(CONF.PREFIX_SET))]
        return dummy_data

    def save_and_remove_prefix(self, data):
        dummy_data = [d['output'] for d in data if d['output'].startswith(tuple(CONF.PREFIX_SET))]

        with open(CONF.DUMMY_FILE, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, ensure_ascii=False, indent=4, separators=(',', ':'))

        return [d for d in data if not any(d['output'].startswith(prefix) for prefix in CONF.PREFIX_SET)]

    def process_json_file(self):
        with open(self.target_file) as f:
            data = json.load(f)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            if 'step1' in self.steps:
                data = list(executor.map(self.remove_specific_words, [data]))

            if 'step2' in self.steps:
                data = list(executor.map(self.remove_short_fields, [data]))

            if 'step3' in self.steps:
                data = list(executor.map(self.replace_phrase, [data]))

            if 'step4' in self.steps:
                data = list(executor.map(self.replace_prefix, [data]))

            if 'step5' in self.steps:
                data = list(executor.map(self.replace_common_phrase, [data]))

            if 'step6' in self.steps:
                data = list(executor.map(self.remove_duplicate_input, [data]))

            if 'step7' in self.steps:
                data = list(executor.map(self.replace_output_prefix, [data]))

            if 'step8' in self.steps:
                data = list(executor.map(self.save_and_remove_prefix, [data]))

        with open(self.target_file, "w", encoding="utf-8") as file:
            json.dump(data[0], file, ensure_ascii=False, indent=4, separators=(',', ':'))


# 예시 사용
cleaner = DataCleaner('target.json', ['step1', 'step2', 'step3'])
cleaner.process_json_file()
