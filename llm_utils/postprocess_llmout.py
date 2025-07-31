import re
import json
from json import JSONDecodeError


def remove_think(text):
    pattern = r'<think>.*?</think>'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def find_first_json(text):
    stack = []
    start_index = -1

    for i, char in enumerate(text):
        # 找到可能的JSON开始
        if not stack and char in '{[':
            start_index = i
            stack.append(char)
            continue

        # 处理开括号
        if char in '{[':
            stack.append(char)

        # 处理闭括号
        elif char in '}]':
            if not stack:
                # 闭括号比开括号多
                continue

            # 检查括号匹配
            if (char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '['):
                stack.pop()

                # 如果栈空了，说明找到了完整的JSON
                if not stack:
                    return text[start_index:i + 1]
    return None

def repair_and_parse_json(json_str):
    try:
        # 修复1: 处理类似Python的None/True/False
        repaired = re.sub(r':\s*None\b', ': null', json_str)
        repaired = re.sub(r':\s*True\b', ': true', repaired)
        repaired = re.sub(r':\s*False\b', ': false', repaired)
        
        # 修复2: 处理单引号字符串 (安全替换)
        repaired = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', repaired)
        
        # 修复3: 移除尾随逗号 (避免在字符串内误替换)
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        
        return json.loads(repaired)
    except JSONDecodeError:
        try:
            # 最后尝试：移除注释（常见于JSONC）
            no_comments = re.sub(r'//.*|/\*[\s\S]*?\*/', '', json_str)
            return json.loads(no_comments)
        except:
            # 终极fallback：返回原始文本
            return json_str

def extract_json(output_text):
    json_str = find_first_json(output_text)
    if not json_str:
        return None
    try:
        return json.loads(json_str)
    except JSONDecodeError as e:
        print(f"修复json...")
        return repair_and_parse_json(json_str)


def extract_TFA(output_text: str):
    output_text = remove_think(output_text)
    Thought = output_text.split("Thought:")[-1].split("Final Answer:")[0].strip()
    Final_Answer = output_text.split("Final Answer:")[-1].strip()
    return Thought, Final_Answer


def extract_yesno(output_text: str):
    output_text = remove_think(output_text)
    output_text = output_text.lower()
    if "no" in output_text:
        return "no"
    elif "yes" in output_text:
        return "yes"
    else:
        return "no"
