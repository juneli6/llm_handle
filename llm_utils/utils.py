import os
import json
import random
import hashlib
import base64
import asyncio
from datetime import datetime, timedelta
try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
except:
    pass


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_jsonl(file_path, errors='replace'):
    data = []
    with open(file_path, 'r', encoding='utf-8', errors=errors) as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                print("Warning: 跳过空白行")
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: 第{line_idx}行JSON解析失败: {e}。\n内容: {line}")
                continue
    return data

def dump_json(obj, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
    return

def dump_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as jsonl_file:
        if isinstance(data, list):
            for item in data:
                jsonl_file.write(json.dumps(item, ensure_ascii=False, indent=None) + '\n')
        else:
            jsonl_file.write(json.dumps(data, ensure_ascii=False, indent=None) + '\n')
    return


def sample_list(lst: list, k: int, mode="uniform"):
    # 从list中采样k个元素并保序返回，uniform模式保证每次返回结果一致
    if not lst:
        return []
    
    if k < 0:
        raise ValueError("采样数量k不能为负数")
    if k > len(lst):
        raise ValueError("采样数量k不能大于列表长度")
    if k == 0:
        return []
    
    if mode == "uniform":
        if k == 1:
            return [lst[0]]
        indices = [i * len(lst) // k for i in range(k)]
    elif mode == "random":
        indices = sorted(random.sample(range(len(lst)), k))
    else:
        raise ValueError("模式必须是 'random' 或 'uniform'")
    
    return [lst[i] for i in indices]


def get_sha256(input_str: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(input_str.encode('utf-8'))
    return hasher.hexdigest()


class DateHandle:

    @staticmethod
    def get_random_date(start: str, end: str) -> str:
        """ 获得指定范围内的随机日期 YYYYMMDD 格式，包含start和end
        """
        start_date = datetime.strptime(start, "%Y%m%d")
        end_date = datetime.strptime(end, "%Y%m%d")
        if start_date > end_date:
            raise ValueError("Start date must be earlier than or equal to end date")
        delta = (end_date - start_date).days
        random_days = random.randint(0, delta)
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime("%Y%m%d")

    @staticmethod
    def shift_date(base_date: str, days: int) -> str:
        """ 根据基准日期计算偏移后的日期 YYYYMMDD
        """
        base_datetime = datetime.strptime(base_date, "%Y%m%d")
        offset_date = base_datetime + timedelta(days=days)
        return offset_date.strftime("%Y%m%d")


class Tokenizer_Handle:

    def __init__(self, tokenizer):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
    
    def count_tokens(
            self, 
            text: str|list[str] = None, 
            add_special_tokens = False, # 对 text 生效
            messages: list|list[list] = None, 
            add_generation_prompt = True, # 对 messages 生效
        ) -> int|list[int]:
        """ str -> int | list -> list
        """
        assert text or messages
        if text:
            encoded = self.tokenizer(text, 
                                    add_special_tokens=add_special_tokens, 
                                    return_length=True)
            token_lengths = encoded["length"]

            if isinstance(text, str):
                assert len(token_lengths) == 1
                return token_lengths[0]
            elif isinstance(text, list):
                return token_lengths
            else:
                raise ValueError(f"type(text) = {type(text)}")
        else:
            encoded = self.tokenizer.apply_chat_template(messages, 
                                                         tokenize=True, 
                                                         add_generation_prompt=add_generation_prompt, 
                                                         padding=False, 
                                                         truncation=False
                                                         )

            if isinstance(messages[0], dict):
                return len(encoded)
            elif isinstance(messages[0], list):
                return [len(i) for i in encoded]
            else:
                raise ValueError(f"type(messages[0]) = {type(messages[0])}")
    
    def cut_by_token_len(self, text: str|list[str], max_tokens: int) -> str|list[str]:
        """ str -> str | list -> list
        """
        text_lst = [text] if isinstance(text, str) else text
        encoded: list[list[int]] = self.tokenizer(text=text_lst, add_special_tokens=False, padding=False, truncation=True, max_length=max_tokens)["input_ids"]
        text_lst_truncated = self.tokenizer.batch_decode(encoded, skip_special_tokens=False)
        if isinstance(text, str):
            assert len(text_lst_truncated) == 1
            return text_lst_truncated[0]
        else:
            return text_lst_truncated


def find_paths_by_prefix_and_suffix(folder_path, max_depth=1, 
                                    prefix = None, suffix = None, target = "file"):
    """ 找到指定文件夹下指定前缀和后缀的文件/文件夹
        max_depth: 1 | "all"
        target: None | file | folder: 为 None 时对 file 和 folder 都匹配
            - 注意 target 不是 file 时不要匹配到了节点文件夹
    """
    matched_paths = []
    for root, dirs, files in os.walk(folder_path):
        if target is None:
            cands = [*dirs, *files]
        elif target == "file":
            cands = files
        elif target == "folder":
            cands = dirs
        else:
            raise

        for i in cands:
            if prefix and not i.startswith(prefix):
                continue
            if suffix and not i.endswith(suffix):
                continue
        
            matched_paths.append(os.path.join(root, i))
        
        if max_depth == 1:
            break
    return matched_paths
 
