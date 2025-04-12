# 240630

inst = "\
You are taking the Pharmacist Licensing Examination. \
Read the question carefully and use your judgment to select one of the options beginning with '①, ②, ③, ④, ⑤' as an answer. \
Submit only one symbol from '①, ②, ③, ④, ⑤' without any explanation, as this is an OMR exam.\
"

inst_list = [inst]

exam_list = [
    r"../0_data/exam_origin.json",
    r"../0_data/exam_sonnet.json",
    ]

exam_list_local = [
    r'../0_data/exam_origin.json',
    r'../0_data/exam_sonnet.json',
    ]

from torch.utils.data import Dataset, DataLoader

class ExamDataset(Dataset):
    def __init__(self, exam, inst, prompt_func):
        self.data = []
        keys_to_copy = ['year', 'session', 'subject_id', 'question_number', 'common_question']
        for ques_dict in exam:
            ques=ques_dict['question_text']
            ques = prompt_func(inst, ques)
            
            new_dict = {key: ques_dict[key] for key in keys_to_copy}
            new_dict['input'] = ques
            new_dict['answer'] = ques_dict['answer']
            self.data.append(new_dict)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    return batch

def prompt_llama2(inst, ques): # Llama-2
    text = f"""\
<s>[INST] <<SYS>>
{inst}
<</SYS>>

{ques} [/INST] Answer: \
"""
    return text

def prompt_momo(inst, ques): # MoMo
    text = f"""\
[INST] <<SYS>>
{inst}
<</SYS>>

{ques} [/INST]
Answer: \
"""
    return text

def prompt_llama3(inst, ques): # Llama-3
    text = f"""\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{inst}<|eot_id|><|start_header_id|>user<|end_header_id|>

{ques}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Answer: """
    return text

def prompt_solar(inst, ques): # SOLAR
    text = f"""\
### System:
{inst}

### User:
{ques}

### Assistant:
Answer: \
"""
    return text

def prompt_command(inst, ques): # Command R
    text = f"""\
<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{inst}<|END_OF_TURN_TOKEN|>\
<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{ques}<|END_OF_TURN_TOKEN|>\
<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Answer: """
    return text

def prompt_qwen2(inst, ques): # Qwen-2
    text = f"""\
<|im_start|>system
{inst}<|im_end|>
<|im_start|>user
{ques}<|im_end|>
<|im_start|>assistant
Answer: """
    return text

def prompt_yi(inst, ques): # Yi-1.5
    text = f"""\
{inst}<|im_start|>user
{ques}<|im_end|>
<|im_start|>assistant
Answer: """
    return text

def prompt_phi3(inst,ques): # Phi-3
    text = f"""\
<s><|user|>
{inst}\n\n{ques}<|end|>
<|assistant|>
Answer: \
"""
    return text
######################################################################

def prompt_gpt(inst, ques): # GPT
    messages=[
        {"role": "system", "content": inst},
        {"role": "user", "content": ques}
    ],
    return messages

def prompt_gpt(inst, ques): # GPT
    messages=[
        {"role": "system", "content": inst},
        {"role": "user", "content": ques}
    ],
    return messages

def prompt_claude(inst, ques): # Claude
    messages=[{"role":"user","content": ques}]
    return messages

def prompt_unicorn(inst, ques): # PaLM Text
    messages=f"""{inst}

input: {ques}
output:
"""
    return messages