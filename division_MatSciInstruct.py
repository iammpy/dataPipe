
import json5
import os
import json


DataSet="SciQAG"

# SciQAG
#  MatSciInstruct

if "__file__" not in globals():
    __file__ = os.path.abspath(".")
print("Running in file:", __file__)
test_none_path=os.path.join(__file__, 'data',f'{DataSet}_test_none.json')
test_txt_path=os.path.join(__file__,'data', f'{DataSet}_test_txt.json')

with open(test_txt_path, 'r') as f:
    # Load the JSON5 data from the file
    data= json.load(f)
with open(test_none_path, 'r') as f:
    # Load the JSON5 data from the file
    none_data= json.load(f)

print("Total number of data points:", len(data))
print("Total number of data points with None in 'answer':", len(none_data))



from model import call_openai,call_huoshan
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

prompt_path=os.path.join(__file__, 'prompts','division.txt')
with open(prompt_path, 'r') as f:
    raw_prompt_template = f.read()

model_list= ["doubao"]
# model_name = "doubao"
##  "r1"
##  "4o"
##  "doubao"
def process_item(signal_data_item,model_name):
    instruction = signal_data_item['metadata']['instruction']
    input_val = signal_data_item['metadata']['input'] 
    output_val = signal_data_item['metadata']['output']

    prompt = raw_prompt_template.replace("{{instruction}}", str(instruction)) \
                                .replace("{{input}}", str(input_val)) \
                                .replace("{{output}}", str(output_val))
    if model_name == "4o":
        _,content = call_openai(prompt)
    else:
        _, content = call_huoshan(prompt,model_name)
    problem_id = signal_data_item['id']
    return content,problem_id

context_type=["txt,none"]
division_type=["SUBJECTIVE","KNOWLEDGE","REASONING"]

SUBJECTIVE_list=[]
KNOWLEDGE_list=[]
REASONING_list=[]

import json_repair
end=200
def run_use_model(model_name):
    with ThreadPoolExecutor(max_workers=64) as executor:
        future_to_data = {}
        for signal_data_item in data:
            future = executor.submit(process_item, signal_data_item,model_name)
            future_to_data[future] = signal_data_item

        for future in concurrent.futures.as_completed(future_to_data):
            try:
                content ,problem_id= future.result() # 获取 process_item 的返回值
                res_json= json_repair.repair_json(content,return_objects=True)
                
                # 打印结果
                print(content)
                print(f"ID: {problem_id}")
                print("-" * 50)
            except Exception as exc:
                failed_trans_problem = future_to_data[future]
                print(f"Item with raw problem '{failed_trans_problem['raw_problem']}' generated an exception: {exc}")

# for model_name in model_list:
#     print(f"Running model: {model_name}")
#     run_use_model(model_name)
with ThreadPoolExecutor(max_workers=len(model_list)) as outer_executor:
    for model_name in model_list:
        outer_executor.submit(run_use_model, model_name)




