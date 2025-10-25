import re
import json
import ast
import copy
from json import JSONDecodeError


def remove_tag(text, tag="think"):
    """ 非贪婪匹配
    """
    pattern = rf'<{tag}>.*?</{tag}>'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def extract_tag(text: str, tag: str = "tool_call", output: str = "last"):
    """ 提取指定标签内的内容，非贪婪匹配
    """
    # 转义标签名，防止包含正则特殊字符
    escaped_tag = re.escape(tag)
    pattern = rf'<{escaped_tag}>(.*?)</{escaped_tag}>'
    
    matches: list[str] = re.findall(pattern, text, re.DOTALL)
    
    if not matches:
        return ""
    
    if output == "first":
        return matches[0]
    elif output == "last":
        return matches[-1]
    else:
        raise ValueError(f"output 参数指定错误：{output}")


class JSON_Handle:

    @staticmethod
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

    @staticmethod
    def find_last_json(text):
        stack = []
        end_index = -1  # 记录最后一个完整JSON的结束位置（闭括号位置）
        
        # 从后向前遍历字符串
        for i in range(len(text)-1, -1, -1):
            char = text[i]
            
            # 栈为空时遇到闭括号：记录结束位置并初始化栈
            if not stack and char in ']}':
                # 压入对应的开括号类型（反向匹配时使用）
                stack.append('{' if char == '}' else '[')
                end_index = i  # 记录结束位置
                continue
            
            # 处理栈非空的情况
            if stack:
                if char in ']}':
                    # 遇到闭括号则压入对应的开括号
                    stack.append('{' if char == '}' else '[')
                elif char in '{[':
                    # 遇到开括号时检查是否匹配
                    if char == stack[-1]:
                        stack.pop()  # 括号匹配成功
                        # 栈变空时表示找到完整的JSON
                        if not stack:
                            # 定位实际开始位置（从i到end_index）
                            return text[i:end_index+1]
                    else:
                        # 括号不匹配，重置状态
                        stack = []
                        end_index = -1
        
        return None  # 未找到有效JSON

    @staticmethod
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
                raise
                # return json_str

    @staticmethod
    def extract_json(output_text:str, loc="last"):
        # 尝试直接 loads
        try:
            return json.loads(output_text.strip())
        except:
            pass
        
        # 提取 json span
        if loc == "last":
            json_str = JSON_Handle.find_last_json(output_text)
        elif loc == "first":
            json_str = JSON_Handle.find_first_json(output_text)
        else:
            raise ValueError("loc 错误")
        
        if not json_str:
            raise RuntimeError("未识别到json字符串")
        
        try:
            return json.loads(json_str)
        except JSONDecodeError as e:
            print(f"修复json...")
            return JSON_Handle.repair_and_parse_json(json_str)


def extract_TFA(output_text: str, pattern_1="Thought:", pattern_2="Final Answer:"):
    output_text = remove_tag(output_text)
    pattern_1_start = output_text.rfind(pattern_1)
    pattern_2_start = output_text.rfind(pattern_2)
    if pattern_1_start == -1 or pattern_2_start == -1 or pattern_1_start >= pattern_2_start:
        raise RuntimeError("解析 pattern 位置失败")
    Thought = output_text[pattern_1_start + len(pattern_1) : pattern_2_start].strip()
    Final_Answer = output_text[pattern_2_start + len(pattern_2):].strip()
    return Thought, Final_Answer


def extract_yesno(output_text: str):
    output_text = output_text.split("</think>")[-1]
    output_text = output_text.lower()
    if "no" in output_text:
        return "no"
    elif "yes" in output_text:
        return "yes"
    else:
        return "no"


def extract_toolcall(content: str):
    def extract_code_from_string(text: str):
        pattern = r"""
            (?:["']code["'])    # 键名：双引号或单引号包裹的code
            \s*:\s*             # 键分割符（允许空格）
            (["'])              # 捕获开始引号（第1组）
            ((?:(?!\1).|\\\1)*) # 捕获字符串内容（第2组）：允许转义引号
            \1                  # 匹配结束引号（与开始引号相同）
        """

        match = re.search(pattern, text, re.VERBOSE | re.DOTALL) # 支持多行和任意字符

        if not match:
            raise ValueError("未找到有效的code字段")
        
        return match.group(2)
    
    def _is_valid_toolcall(obj) -> bool:
        return (
            isinstance(obj, dict) and 
            'name' in obj and 
            'arguments' in obj and
            isinstance(obj.get('arguments'), dict)
        )
    
    content = content.strip()
    
    # 1. 标准JSON解析
    try:
        result = json.loads(content)
        if _is_valid_toolcall(result):
            return result
    except:
        pass
    
    # 2. 尝试修复后的JSON解析
    try:
        result = JSON_Handle.repair_and_parse_json(content)
        if _is_valid_toolcall(result):
            return result
    except:
        pass

    # 3. 安全字面值解析
    try:
        repaired = content
        replacements = [
            (r':\s*null\b', ': None'),
            (r':\s*true\b', ': True'), 
            (r':\s*false\b', ': False'),
        ]
        
        for pattern, replacement in replacements:
            repaired = re.sub(pattern, replacement, repaired)
        
        result = ast.literal_eval(repaired)
        if _is_valid_toolcall(result):
            return result
    except:
        pass

    # 4. 特殊工具处理
    if "exe_python" in content:
        try:
            result =  {
                "name": "exe_python", 
                "arguments": {"code": extract_code_from_string(content)}
            }
            if _is_valid_toolcall(result):
                return result
        except:
            pass
    
    raise ValueError(f"提取tool call失败: {content}")


def extract_last_boxed(text: str):
    """ 提取最后一个 "\\boxed{}" 中的内容，必须是双斜杠
    """
    start_key = r'\boxed{'
    start_idx = text.rfind(start_key)

    if start_idx == -1:
        return None
    
    start_idx += len(start_key)
    count = 1  # 括号计数器（已有一个左括号）
    content_start = start_idx
    
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            count += 1
        elif text[i] == '}':
            count -= 1
            if count == 0:
                return text[content_start:i]
    
    return None


def alpaca_to_messages(sample: dict, default_system = None):
    messages = []

    system = sample.get("system")
    if system in ["", None]:
        system = default_system
    if system:
        messages.append({"role": "system", "content": system})
    
    history = sample.get("history")
    if history in ["", None]:
        history = []
    for user_content, assistant_content in history:
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    assert len(instruction + input_text) > 0
    messages.append({"role": "user", "content": instruction + input_text})

    output = sample.get("output", "")
    assert output
    messages.append({"role": "assistant", "content": output})

    return messages

def messages_to_alpaca(messages: list, default_system = None):
    # system
    if messages[0]["role"] == "system":
        system = messages[0]["content"]
        messages = messages[1:]
    else:
        system = default_system
    
    # input output
    assert len(messages) % 2 == 0
    for i, msg in enumerate(messages):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if msg["role"] != expected_role:
            raise ValueError(f"消息角色顺序错误：位置{i}应该是{expected_role}，实际是{msg['role']}")
    input_text = messages[-2]["content"]
    output = messages[-1]["content"]
    messages = messages[:-2]

    # history
    history = []
    for i in range(0, len(messages), 2):
        history.append([messages[i]["content"], messages[i+1]["content"]])
    
    alpaca_data = {
        "system": system if system else "", 
        "history": history, 
        "instruction": "", 
        "input": input_text, 
        "output": output
    }

    return alpaca_data

