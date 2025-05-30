
import json5
import os
import json


DataSet="MatSciInstruct"

# SciQAG
#  MatSciInstruct

if "__file__" in globals():
    os.chdir(os.path.join(os.path.dirname(__file__)))
    
print("Running in file:", os.getcwd())



from model import call_openai,call_huoshan
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
model_answer_lock = threading.Lock()

prompt_path=os.path.join('.', 'prompts','division.txt')
with open(prompt_path, 'r') as f:
    raw_prompt_template = f.read()

# model_list= ["r1"]
# model_name = "doubao"
##  "r1"
##  "4o"
##  "doubao"

def process_item(signal_data_item,model_name):
    global DataSet
    if DataSet == "MatSciInstruct":
        instruction = signal_data_item['metadata']['instruction']
        input_val = signal_data_item['metadata']['input'] 
        output_val = signal_data_item['metadata']['output']
    elif DataSet == "SciQAG":
        instruction = signal_data_item['metadata']['question']
        input_val = signal_data_item['metadata']['txt'] 
        output_val = signal_data_item['metadata']['answer']
    prompt = raw_prompt_template.replace("{{instruction}}", str(instruction)) \
                                .replace("{{input}}", str(input_val)) \
                                .replace("{{output}}", str(output_val))
    if model_name == "4o":
        _,content = call_openai(prompt)
    else:
        _, content = call_huoshan(prompt,model_name)
    problem_id = signal_data_item['id']
    return content,problem_id

# context_type=["txt,none"]
# division_type=["SUBJECTIVE","KNOWLEDGE","REASONING"]

output_list = []
import json_repair
def run_use_model(model_name,data,task):
    global output_list
    output_list=[]
    counter = 0
    with ThreadPoolExecutor(max_workers=64) as executor:
        future_to_data = {}
        for signal_data_item in data:
            future = executor.submit(process_item, signal_data_item,model_name)
            future_to_data[future] = signal_data_item

        for future in concurrent.futures.as_completed(future_to_data):
            try:
                content ,problem_id= future.result() # 获取 process_item 的返回值
                res_json= json_repair.repair_json(content,return_objects=True)
                model_division = res_json.get("category", "UNKNOWN")
                if model_division == "UNKNOWN":
                    print(f"Warning: Problem ID {problem_id} has an unknown category.")
                    # continue
                # 将结果存储到 output_list 中
                future_to_data[future]["metadata"]["problem_type"] = model_division
                output_list.append(future_to_data[future])
                
                counter += 1
                if counter % 50 == 0:
                    print(f"Processed {counter} items so far .task: {task} model: {model_name}")
                
                # 打印结果
                # print(content)
                # print(f"ID: {problem_id}")
                # print("-" * 50)
            except Exception as exc:
                failed_trans_problem = future_to_data[future]
                print(f"Item with raw problem '{failed_trans_problem['id']}' generated an exception: {exc}")

# for model_name in model_list:
#     print(f"Running model: {model_name}")
#     run_use_model(model_name)
model_name= "r1"
task_list=[
    "MatSciInstruct_train_none",
    "MatSciInstruct_train_txt",
    "SciQAG_train",
    "MatSciInstruct_test_none",
   "MatSciInstruct_test_txt",

    "SciQAG_test", 
]

# DataSet="MatSciInstruct"

# SciQAG
#  MatSciInstruct

for task in task_list:
            
    DataSet= task.split("_")[0]
    print(f"Running task: {task} with model: {model_name} on dataset: {DataSet}")
    data_name=task
    data_path = os.path.join('.', 'data', f"{data_name}.json")
    with open(data_path, 'r') as f:
        data = json.load(f)
    print("Total number of data points:", len(data))
    run_use_model(model_name,data,task)
    output_path = os.path.join('.', 'output', f"{data_name}_type.json")
    with open(output_path, 'w') as f:
        json.dump(output_list, f, indent=4, ensure_ascii=False)
    print(f"Output saved to {output_path}")



