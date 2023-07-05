# -*- coding: utf-8 -*-
"""
File: data_cleaner.py
Author: Minwoo Park
Date: July 4, 2023
Python Version: Requires Python 3.8 or above
Description: This file should be executed using Python 3.8 or above.
"""

import json
import concurrent.futures
import re
import src.config as CONF
from typing import List


class DataCleaner:
    """
    DataCleaner class for processing and cleaning JSON data.
    """

    def __init__(self, target_file: str, steps: List[str], preprocessing_param: dict):
        """
        Initialize the DataCleaner object.

        Args:
            target_file (str): Path to the target JSON file.
            steps (List[str]): List of steps to perform during data cleaning.
            preprocessing_param (dict): Preprocessing parameters containing the prefix set,
                                        dummy file path, phrase dictionary, and remove target word set.
        """
        self.target_file = target_file
        self.steps = steps
        self.prefix_set = preprocessing_param['prefix_set']
        self.dummy_file_path = preprocessing_param['dummy_file_path']
        self.phrase_dictionary = preprocessing_param['phrase_dictionary']
        self.remove_target_word_set = preprocessing_param['remove_target_word_set']

    def remove_specific_words(self, data: List[dict]) -> List[dict]:
        """
        Remove specific words from the input data.

        Args:
            data (List[dict]): Input data.

        Returns:
            List[dict]: Processed data with specific words removed.
        """
        return [d for d in data if not any(word in d['input'] for word in self.remove_target_word_set)]

    def remove_short_fields(self, data: List[dict]) -> List[dict]:
        """
        Remove data fields with input and output lengths less than or equal to 4.

        Args:
            data (List[dict]): Input data.

        Returns:
            List[dict]: Processed data with short fields removed.
        """
        return [d for d in data if (input_len := len(d['input'])) > 4 and (output_len := len(d['output'])) > 4]

    def replace_phrase(self, data: List[dict]) -> List[dict]:
        """
        Replace phrases in the output data using the provided phrase dictionary.

        Args:
            data (List[dict]): Input data.

        Returns:
            List[dict]: Processed data with phrases replaced.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self.replace_sure_translation, data))

    def replace_sure_translation(self, d: dict) -> dict:
        """
        Replace "물론," with "물론이죠." and remove Korean prefixes from the output data.

        Args:
            d (dict): Input dictionary.

        Returns:
            dict: Processed dictionary with phrases replaced and prefixes removed.
        """
        for pattern, replacement in self.phrase_dictionary.items():
            d['output'] = re.sub(pattern, replacement, d['output'])
        return d

    def delete_error_korean_prefix(self, d: dict) -> dict:
        """
        Delete error Korean prefixes (은,는) from the output data.

        Args:
            d (dict): Input dictionary.

        Returns:
            dict: Processed dictionary with error Korean prefixes removed.
        """
        d['output'] = re.sub(r'^(은 |는 )', '', d['output'])
        return d

    def replace_prefix(self, data: List[dict]) -> List[dict]:
        """
        Replace prefixes in the output data.

        Args:
            data (List[dict]): Input data.

        Returns:
            List[dict]: Processed data with prefixes replaced.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self.delete_error_korean_prefix, data))

    def do_not_translate_code_snippet(self, d: dict) -> dict:
        """
        Do not translate code snippets in the input and output data.

        Args:
            d (dict): Input dictionary.

        Returns:
            dict: Processed dictionary with code snippets unchanged.
        """
        input_text = d['input']
        output_text = d['output']

        if '```' in input_text and '```' in output_text:
            start_index = input_text.find('```') + 3
            end_index = input_text.find('```', start_index)
            replace_text = input_text[start_index:end_index]

            d['output'] = output_text.replace('```', f' ```{replace_text}```')

        return d

    def replace_common_phrase(self, data: List[dict]) -> List[dict]:
        """
        Replace common phrases in the output data.

        Args:
            data (List[dict]): Input data.

        Returns:
            List[dict]: Processed data with common phrases replaced.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self.do_not_translate_code_snippet, data))

    def remove_duplicate_input(self, data: List[dict]) -> List[dict]:
        """
        Remove duplicate input data.

        Args:
            data (List[dict]): Input data.

        Returns:
            List[dict]: Processed data with duplicate input removed.
        """
        unique_inputs = set()
        result = []

        for d in data:
            input_text = d['input']
            if input_text not in unique_inputs:
                unique_inputs.add(input_text)
                result.append(d)

        return result

    def replace_output_prefix(self, data: List[dict]) -> List[str]:
        """
        Replace output prefixes.

        Args:
            data (List[dict]): Input data.

        Returns:
            List[str]: List of output prefixes.
        """
        dummy_data = [d['output'] for d in data if d['output'].startswith(tuple(self.prefix_set))]
        return dummy_data

    def save_and_remove_prefix(self, data: List[dict]) -> List[dict]:
        """
        Save and remove output prefixes.

        Args:
            data (List[dict]): Input data.

        Returns:
            List[dict]: Processed data with prefixes saved and removed.
        """
        dummy_data = [d['output'] for d in data if d['output'].startswith(tuple(self.prefix_set))]

        with open(self.dummy_file_path, 'w', encoding='utf-8') as f:
            json.dump(dummy_data, f, ensure_ascii=False, indent=4, separators=(',', ':'))

        return [d for d in data if not any(d['output'].startswith(prefix) for prefix in self.prefix_set)]

    def process_json_file(self):
        """
        Process the JSON file based on the specified steps and save the cleaned data.
        """
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


if __name__ == "__main__":
    cleaner = DataCleaner(CONF.preprocessing_param['target_json_file_path'], ['step1', 'step2', 'step3'], preprocessing_param=CONF.preprocessing_param)
    cleaner.process_json_file()
