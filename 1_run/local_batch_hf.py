# %%
# 240630

import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from itertools import product
import gc
from format import *
from torch.utils.data import Dataset, DataLoader
import json

# %%
class ModelResponder:
    def __init__(self, model_path, exam_list_local, prompt_func_list, inst_list, path=None, quant=False, llama3=False, qwen=False, yi=False):
        self.batch_size = 350
        self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if yi:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16 if (llama3 or qwen or yi) else torch.float16,
            quantization_config=self.quantization_config if quant else None,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        if path is not None:
            self.path = path
        else:
            self.path = os.path.basename(model_path)

        self.gen_args = {
            "use_cache": True,
            "max_new_tokens": float('inf'),
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }

        if llama3:
            self.terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
            self.gen_args["eos_token_id"] = self.terminators
            self.gen_args["pad_token_id"] = self.tokenizer.eos_token_id
        if yi:
            self.gen_args["eos_token_id"] = self.tokenizer.eos_token_id
            
        indexed_ques_paths = list(enumerate(exam_list_local))
        indexed_prompt_funcs = list(enumerate(prompt_func_list))
        indexed_insts = list(enumerate(inst_list))

        self.combinations = list(product(indexed_ques_paths, indexed_prompt_funcs, indexed_insts))

    def process_files(self):
        for (ques_idx, ques_path), (prompt_idx, prompt_func), (inst_idx, inst) in self.combinations:
            with open(ques_path, encoding='utf-8') as f:
                exam = json.load(f)
            filename=f'output/{self.path} [f{prompt_idx+1}_p{inst_idx+1}_q{ques_idx+1}].json'
            dataset = ExamDataset(exam, inst, prompt_func)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
            with tqdm(total=len(dataloader),desc=filename, leave=True, position=1) as pbar:
                for i, batch in enumerate(dataloader):
                    results=[]
                    with tqdm(total=len(batch), desc=f"Batch {i}", leave=True, position=0) as batch_pbar:
                        for ques_dict in batch:
                            max_retries = 5
                            retries = 0
                            ques=ques_dict['input']
                            while retries < max_retries:
                                try:
                                    inputs = self.tokenizer(ques, return_tensors="pt", add_special_tokens=False).to(self.model.device)
                                    input_len = len(inputs.input_ids[0])
                                    output = self.model.generate(inputs.input_ids, **self.gen_args)
                                    output_text = self.tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
                                    ques_dict['response'] = output_text
                                    results.append(ques_dict)
                                    batch_pbar.update(1)
                                    break
                                except RuntimeError as e:
                                    current = f"{ques_dict['year']}-{ques_dict['session']}-{ques_dict['question_number']}"
                                    if "CUDA out of memory" in str(e):
                                        print(f"CUDA OOM on '{current}'. Trying again after clearing cache.")
                                        torch.cuda.empty_cache()
                                        retries += 1
                                        if retries == max_retries:
                                            print(f"Skipping {current} after repeated OOM errors.")
                                            ques_dict['response'] = None
                                            results.append(ques_dict)
                                            batch_pbar.update(1)
                                    else:
                                        print(f"Error on '{current}': {e}")
                                        raise

                    if os.path.exists(filename):
                        with open(filename, 'r', encoding='utf-8') as file:
                            resp = json.load(file)
                    else:
                        resp = []
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                    resp += results
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(resp, f, indent=4, ensure_ascii=False)
                    pbar.update(1)
                    
    def delete(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

# %%
# model_path = "meta-llama/Meta-Llama-3-70B-Instruct"
# prompt_func_list = [prompt_llama3]
# run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list, llama3=True)
# run.process_files()
# run.delete()

# model_path = "CohereForAI/c4ai-command-r-plus"
# prompt_func_list = [prompt_command]
# run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list)
# run.process_files()
# # run.delete()

# model_path = "Qwen/Qwen2-72B-Instruct"
# prompt_func_list = [prompt_qwen2]
# run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list, qwen=True)
# run.process_files()
# run.delete()

# model_path = "CohereForAI/c4ai-command-r-v01"
# prompt_func_list = [prompt_command]
# run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list)
# run.process_files()
# run.delete()

# model_path = "01-ai/Yi-1.5-34B-Chat"
# prompt_func_list = [prompt_yi]
# run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list, yi=True)
# run.process_files()
# run.delete()

# model_path = "moreh/MoMo-72B-lora-1.8.7-DPO"
# prompt_func_list = [prompt_momo]
# run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list)
# run.process_files()
# run.delete()

# model_path = "upstage/SOLAR-10.7B-Instruct-v1.0"
# prompt_func_list = [prompt_solar]
# run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list)
# run.process_files()
# run.delete()

# model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# prompt_func_list = [prompt_llama3]
# run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list, llama3=True)
# run.process_files()
# run.delete()

# model_path = "meta-llama/Llama-2-7b-chat-hf"
# prompt_func_list = [prompt_llama2]
# run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list)
# run.process_files()
# run.delete()

# model_path = "microsoft/Phi-3-medium-4k-instruct"
# prompt_func_list = [prompt_phi3]
# run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list, qwen=True)
# run.process_files()
# run.delete()

# model_path = "meta-llama/Llama-2-70b-chat-hf"
# prompt_func_list = [prompt_llama2]
# run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list)
# run.process_files()
# run.delete()

# model_path = "upstage/SOLAR-0-70b-16bit"
# prompt_func_list = [prompt_solar]
# run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list)
# run.process_files()
# run.delete()