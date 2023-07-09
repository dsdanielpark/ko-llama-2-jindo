# -*- coding: utf-8 -*-
"""
File: data_cleaner.py
Author: Minwoo Park
Date: July 4, 2023
Python Version: Requires Python 3.8 or above
Description: This file should be executed using Python 3.8 or above.
"""

from utilfunction import find_path
import json
import concurrent.futures
import re


class DataProcessor:
    def __init__(self, input_file_path):
        with open(input_file_path) as f:
            self.data = json.load(f)
        self.excluded_data = []

    @staticmethod
    def remove_specific_words(data):
        word_set = {'sure!', 'great!', 'Certainly!', "Sure!", "Great!", "Great, ", "great, ",
                    "sure, ", "Sure, ", "Sure! "}
        return [d for d in data if not any(word in d['input'] for word in word_set)]

    @staticmethod
    def remove_short_fields(data):
        try:
            return [d for d in data if (input_len := len(d['input'])) > 4 and (output_len := len(d['output'])) > 4]
        except Exception as e:
            print(f"Error at remove_short_fields {e} \n Finished unsuccessfully.")
            return data

    @staticmethod
    def replace_sure_translation(data):
        for d in data:
            try:
                d['output'] = re.sub(r'\b물론,\b', '물론이죠.', d['output'])
                d['output'] = re.sub(r'\b확실히,\b', '', d['output'])
                d['output'] = re.sub(r'\b예,\b', '네.', d['output'])
            except Exception as e:
                print(f"Error at replace_sure_translation {e}")
        return data

    @staticmethod
    def delete_error_korean_prefix(data):
        for d in data:
            try:
                d['output'] = re.sub(r'^(은 |는 )', '', d['output'])
            except Exception as e:
                print(f"Error at delete_error_korean_prefix {e}")
        return data

    @staticmethod
    def replace_output_prefix(data):
        prefix_set = {"을 ", "를 ", "이 ", "가 ", "h", "은 ", "는 ", "에 ", "으 ", "의", "예, ", "^[A-Za-z] ", "^[ㄱ-ㅎㅏ-ㅣ가-힣] ", "^[0-9] ", ".", ","}
        exclued_data = []

        for d in data:
            try:
                output_text = d['output']
                if output_text.startswith(tuple(prefix_set)):
                    exclued_data.append(d)
            except Exception as e:
                print(f"Error at replace_output_prefix {e}")
            
        return exclued_data



    @staticmethod
    def do_not_translate_code_snippet(data):
        for d in data:
            try:
                input_text = d['input']
                output_text = d['output']

                if '```' in input_text and '```' in output_text:
                    start_index = input_text.find('```') + 3
                    end_index = input_text.find('```', start_index)
                    replace_text = input_text[start_index:end_index]

                    d['output'] = output_text.replace('```', f' ```{replace_text}```')
            except Exception as e:
                print(f"Error at do_not_translate_code_snippet {e}")

        return data

    @staticmethod
    def remove_duplicates(data):
        unique_data = []
        seen_inputs = set()
        seen_outputs = set()
        for d in data:
            input_value = d["input"]
            output_value = d["output"]
            if (input_value, output_value) not in seen_inputs and \
                    (input_value, output_value) not in seen_outputs and \
                    input_value != output_value:
                seen_inputs.add((input_value, output_value))
                seen_outputs.add((input_value, output_value))
                unique_data.append(d)
        return unique_data

    @staticmethod
    def remove_deletion_and_addition(data):
        for d in data:
            input_value = d["input"]
            output_value = d["output"]

            input_words = input_value.split()
            output_words = output_value.split()

            if len(output_words[0]) > 1 and len(input_words[0]) > 2:
                if len(set(output_words[0].lower()) - set(input_words[0].lower())) < 2 and \
                        input_words[0][1].lower() == output_words[0][0].lower() and \
                        input_words[0][2].lower() == output_words[0][1].lower():
                    output_words[0] = input_words[0]
                    output_value = " ".join(output_words)
                    d["output"] = output_value

            if output_words[0] == "물론,":
                output_words[0] = "물론이죠. "
                output_value = " ".join(output_words)
                d["output"] = output_value

            if len(output_words[0]) == 1 and output_words[0].isalpha() and output_words[0].isascii() and output_words[0].lower() != "a":
                output_words[0] = ""
                output_value = " ".join(output_words)

        return data

    @staticmethod
    def flatten_list(data):
        flattened_list = []
        for sublist in data:
            if isinstance(sublist, list):
                flattened_list.extend(DataProcessor.flatten_list(sublist))
            else:
                flattened_list.append(sublist)
        return flattened_list

    @staticmethod
    def write_to_file(data, file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4, separators=(',', ':'))


    def process_json_file(self, steps, output_file_path, dummy_file_path):
        for step in steps:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                if 'step1' in step:
                    self.data = list(executor.map(DataProcessor.remove_specific_words, [self.data]))
                    self.data = DataProcessor.flatten_list(self.data)

                if 'step2' in step:
                    self.data = list(executor.map(DataProcessor.remove_short_fields, [self.data]))
                    self.data = DataProcessor.flatten_list(self.data)

                if 'step3' in step:
                    self.data = list(executor.map(DataProcessor.replace_sure_translation, [self.data]))
                    self.data = DataProcessor.flatten_list(self.data)

                if 'step4' in step:
                    self.data = list(executor.map(DataProcessor.delete_error_korean_prefix, [self.data]))
                    self.data = DataProcessor.flatten_list(self.data)

                if 'step5' in step:
                    self.data = list(executor.map(DataProcessor.do_not_translate_code_snippet, [self.data]))
                    self.data = DataProcessor.flatten_list(self.data)

                if 'step6' in step:
                    self.data = list(executor.map(DataProcessor.remove_duplicates, [self.data]))
                    self.data = DataProcessor.flatten_list(self.data)

                if 'step7' in step:
                    self.excluded_data = list(executor.map(self.replace_output_prefix, [self.data]))
                    self.excluded_data = DataProcessor.flatten_list(self.excluded_data)
                    self.data = [d for d in self.data if d not in self.excluded_data]

                if 'step8' in step:
                    self.data = list(executor.map(DataProcessor.remove_deletion_and_addition, [self.data]))
                    self.data = DataProcessor.flatten_list(self.data)

        self.data = DataProcessor.flatten_list(self.data)
        self.write_to_file(self.data, output_file_path)
        self.write_to_file(self.excluded_data, dummy_file_path)

        @staticmethod
        def static_process_json_file(steps, input_file_path, output_file_path, dummy_file_path):
            with open(input_file_path) as f:
                data = json.load(f)
            excluded_data = []
            
            for step in steps:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    if 'step1' in step:
                        data = list(executor.map(DataProcessor.remove_specific_words, [data]))
                        data = DataProcessor.flatten_list(data)

                    if 'step2' in step:
                        data = list(executor.map(DataProcessor.remove_short_fields, [data]))
                        data = DataProcessor.flatten_list(data)

                    if 'step3' in step:
                        data = list(executor.map(DataProcessor.replace_sure_translation, [data]))
                        data = DataProcessor.flatten_list(data)

                    if 'step4' in step:
                        data = list(executor.map(DataProcessor.delete_error_korean_prefix, [data]))
                        data = DataProcessor.flatten_list(data)

                    if 'step5' in step:
                        data = list(executor.map(DataProcessor.do_not_translate_code_snippet, [data]))
                        data = DataProcessor.flatten_list(data)

                    if 'step6' in step:
                        data = list(executor.map(DataProcessor.remove_duplicates, [data]))
                        data = DataProcessor.flatten_list(data)

                    if 'step7' in step:
                        excluded_data = list(executor.map(DataProcessor.replace_output_prefix, [data]))
                        excluded_data = DataProcessor.flatten_list(excluded_data)
                        data = [d for d in data if d not in excluded_data]

                    if 'step8' in step:
                        data = list(executor.map(DataProcessor.remove_deletion_and_addition, [data]))
                        data = DataProcessor.flatten_list(data)

            data = DataProcessor.flatten_list(data)
            DataProcessor.write_to_file(data, output_file_path)
            DataProcessor.write_to_file(excluded_data, dummy_file_path)



if __name__ == "__main__":
    json_file_list = find_path('../notebooks/output/', 'file', '.json')
    steps = ['step'+str(i) for i in range(9)]
    dummy_file_path = 'dummy.json'

    [DataProcessor.static_process_json_file(steps, json_file_list[i], json_file_list[i], dummy_file_path) for i in range(len(json_file_list))]
    