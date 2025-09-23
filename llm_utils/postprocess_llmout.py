import re
import json
from json import JSONDecodeError


def remove_tag(text, tag="think"):
    """ 非贪婪匹配
    """
    pattern = rf'<{tag}>.*?</{tag}>'
    return re.sub(pattern, '', text, flags=re.DOTALL)


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
