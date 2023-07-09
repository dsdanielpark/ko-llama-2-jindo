import json
import concurrent.futures

def extract_values(json_data_1, json_data_2, translate_type):
    values_list = []
    
    if translate_type =='ko_to_en':
        for item_1, item_2 in zip(json_data_1, json_data_2):
            
            conversations_1 = item_1.get("conversations", [])
            conversations_2 = item_2.get("conversations", [])
            
            if len(conversations_1) != len(conversations_2):
                continue
            
            for conversation_1, conversation_2 in zip(conversations_1, conversations_2):
                value_1 = conversation_1.get("value")
                value_2 = conversation_2.get("value")
                
                temp_dict = {} 
                
                temp_dict['instruction'] = "다음을 한국어로 번역해."
                temp_dict['input'] = value_1
                temp_dict['output'] = value_2
                
                if value_1 and value_2:
                    values_list.append(temp_dict)
    elif translate_type =='en_to_ko':
        for item_1, item_2 in zip(json_data_1, json_data_2):
            
            conversations_1 = item_1.get("conversations", [])
            conversations_2 = item_2.get("conversations", [])
            
            if len(conversations_1) != len(conversations_2):
                continue
            
            for conversation_1, conversation_2 in zip(conversations_1, conversations_2):
                value_1 = conversation_1.get("value")
                value_2 = conversation_2.get("value")
                
                temp_dict = {} 
                
                temp_dict['instruction'] = "translate the following into English"
                temp_dict['input'] = value_1
                temp_dict['output'] = value_2
                
                if value_1 and value_2:
                    values_list.append(temp_dict)
    
    return values_list
        
    return values_list

def process_json_files(file_path_1, file_path_2, chunk_size, output_filepath, translate_type):
    with open(file_path_1) as file:
        data_1 = json.load(file)[:200]

    with open(file_path_2) as file:
        data_2 = json.load(file)[:200]

    data_1_chunks = [data_1[i:i+chunk_size] for i in range(0, len(data_1), chunk_size)]
    data_2_chunks = [data_2[i:i+chunk_size] for i in range(0, len(data_2), chunk_size)]

    values = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for chunk_1, chunk_2 in zip(data_1_chunks, data_2_chunks):
            future = executor.submit(extract_values, chunk_1, chunk_2, translate_type)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            values.extend(future.result())

    with open(output_filepath, "w", encoding="utf-8") as file:
        json.dump(values, file, ensure_ascii=False,  indent=4, separators=(',', ':'))  



if __name__=="__main__":
    file_path_2 = "../data/original_sharegpt.json"
    file_path_1 = "../data/ko_sharegpt.json"
    chunk_size = 1000
    output_filepath = "shargpt_deepl_cleaned_for_en_to_kor.json"
    process_json_files(file_path_1, file_path_2, chunk_size, output_filepath, "en_to_ko")
