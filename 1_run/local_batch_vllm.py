# %%
# 240627

import os
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
from itertools import product
import gc
from format import *
from torch.utils.data import Dataset, DataLoader
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# %%
class ModelResponder:
    def __init__(self, model_path, exam_list_local, prompt_func_list, inst_list, path=None, quant=False, llama3=False, qwen=False, yi=False):
        self.sampling_params = SamplingParams(temperature=0)
        self.batch_size = 1024
        self.device_num = torch.cuda.device_count()
        self.model = LLM(model=model_path, tensor_parallel_size=self.device_num, disable_custom_all_reduce=True, gpu_memory_utilization=0.8)
        if path is not None:
            self.path = path
        else:
            self.path = os.path.basename(model_path)

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
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            with tqdm(total=len(dataloader),desc=filename, leave=True, position=1) as pbar:
                for i, batch in enumerate(dataloader):
                    max_retries = 5
                    retries = 0
                    while retries < max_retries:
                        try:
                            outputs = self.model.generate(batch['input'],sampling_params=self.sampling_params)
                            output_text = []
                            for output in outputs:
                                generated_text = output.outputs[0].text
                                output_text.append(generated_text)
                            batch['response'] = output_text
                            results = [{key: batch[key][i].tolist() if isinstance(batch[key][i], torch.Tensor) else batch[key][i] for key in batch} for i in range(len(batch['input']))]

                            break
                        except RuntimeError as e:
                            current = f"Batch {i} in {filename}"
                            if "CUDA out of memory" in str(e):
                                print(f"CUDA OOM on '{current}'. Trying again after clearing cache.")
                                torch.cuda.empty_cache()
                                retries += 1
                                if retries == max_retries:
                                    print(f"Skipping {current} after repeated OOM errors.")
                                    batch['response'] = None
                                    results = [{key: batch[key][i].tolist() if isinstance(batch[key][i], torch.Tensor) else batch[key][i] for key in batch} for i in range(len(batch['input']))]
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
def main():
    model_path = "meta-llama/Meta-Llama-3-70B-Instruct"
    prompt_func_list = [prompt_llama3]
    run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list, path=1024)
    run.process_files()
    run.delete()

# def main():
#     model_path = "meta-llama/Llama-2-70b-chat-hf" 
#     prompt_func_list = [prompt_llama2]
#     run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list)
#     run.process_files()
#     run.delete()

# def main():
#     model_path = "/home/hwjang/project/LLM/data/SOLAR-0-70b-16bit"
#     prompt_func_list = [prompt_solar]
#     run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list)
#     run.process_files()
#     run.delete()


# def main():
#     model_path = "Qwen/Qwen2-72B-Instruct"
#     prompt_func_list = [prompt_qwen2]
#     run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list, qwen=True)
#     run.process_files()
#     run.delete()

# def main():
#     model_path = "CohereForAI/c4ai-command-r-plus"
#     prompt_func_list = [prompt_command]
#     run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list)
#     run.process_files()
#     run.delete()

# def main():
#     model_path = "CohereForAI/c4ai-command-r-v01"
#     prompt_func_list = [prompt_command]
#     run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list)
#     run.process_files()
#     run.delete()

# def main():
#     model_path = "01-ai/Yi-1.5-34B-Chat"
#     prompt_func_list = [prompt_yi]
#     run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list, yi=True)
#     run.process_files()
#     run.delete()

# def main():
#     model_path = "moreh/MoMo-72B-lora-1.8.7-DPO"
#     prompt_func_list = [prompt_momo]
#     run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list)
#     run.process_files()
#     run.delete()

# def main():
#     model_path = "meta-llama/Llama-2-7b-chat-hf"
#     prompt_func_list = [prompt_llama2]
#     run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list)
#     run.process_files()
#     run.delete()

# def main():
#     model_path = "upstage/SOLAR-10.7B-Instruct-v1.0"
#     prompt_func_list = [prompt_solar]
#     run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list)
#     run.process_files()
#     run.delete()

# def main():
#     model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
#     prompt_func_list = [prompt_llama3]
#     run = ModelResponder(model_path, exam_list_local, prompt_func_list, inst_list)
#     run.process_files()
#     run.delete()

if __name__ == '__main__':
    main()