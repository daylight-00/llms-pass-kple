# 240626

def format(ques_dict):
    ques=""
    if 'common_question_text' in ques_dict and ques_dict['common_question_text'] is not None:
        ques=f"###\n{ques_dict['common_question_text']}\n###\n{ques_dict['common_data']}\n"
    ques=ques+f"###\n{ques_dict['question_text']}\n###\n"
    if 'data' in ques_dict and ques_dict['data'] is not None:
        ques=f"{ques_dict['data']}\n\n"
    ques=ques+f"{"\n".join(ques_dict['choices'])}"
    return ques

def extract_substring(s):
    start = s.find('{')
    end = s.rfind('}')
    
    if start != -1 and end != -1 and start < end:
        return s[start:end+1]
    else:
        return "Invalid input: '{}' not found or in incorrect order."

import ast, re
def output_adjust(text):
    text = text.replace(r'"', r'\"')
    text = text.replace(r"\'", r"'")
    text = text.replace(r'\", "', r'", "')
    text = text.replace(r': \"', r': "')
    text = text.replace(r'Note: "10X\"' , r'Note: \"10X\"')
    text = text.replace(r'\", '+r"'", r'", '+"'")
    text = text.replace(r'\", \"⑤', r'", "⑤')
    text = text.replace(r'\", \"④', r'", "④')
    text = text.replace(r'\", \"③', r'", "③')
    text = text.replace(r'\", \"②', r'", "②')
    text = text.replace(r'\"]}', r'"]}')
    text = text.replace(r': [\"', r': ["')
    patterns = [
        r"', '",
        r", '",
        r"': ",
        r": '",
        r"{'",
        r"'\]}",
        r": \['"
    ]
    text=re.sub(r'\n', r'\\n', text)
    for pattern in patterns:
        text = re.sub(pattern, lambda x: x.group(0).replace("'", '"'), text)
    text = ast.literal_eval(text)
    return text

from torch.utils.data import Dataset, DataLoader
class ExamDataset(Dataset):
    def __init__(self, exam, inst, prompt_func):
        self.data = exam
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    return batch

inst = "\
You are a professional translator. \
You will translate a question from the Pharmacist Licensing Examination formatted in a dictionary format. \
Translate the question input in Korean into English, maintaining its original form.\
"

inst_list = [inst]

ques_path_list = [
    r'exams_semi_trans.json',
    ]