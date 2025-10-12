import os
import sys
import json
import requests


def send_request_vllm_chat(
        url = None,
        api_key = None,
        model_name = None,
        messages:list[dict] = None,
        other_params = None,
        timeout=(10, 120)
    ):
    # https://docs.vllm.ai/en/latest/getting_started/quickstart.html
    # https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    # https://docs.vllm.ai/en/latest/dev/sampling_params.html

    default_other_params = {
        "seed": None, 
        "temperature": 1.0, 
        "top_p": 1.0, 
        "top_k": -1, 
        "n": 1, # Number of output sequences to return for the given prompt.
        "max_tokens": 16, 

        "stop": None, # List of strings that stop the generation when they are generated. The returned output will not contain the stop strings.
        "stop_token_ids": None, # List of tokens that stop the generation when they are generated. The returned output will contain the stop tokens unless the stop tokens are special tokens.
        "include_stop_str_in_output": False, # Whether to include the stop strings in output text. Defaults to False.
        "frequency_penalty": 0.0, # Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
        "presence_penalty": 0.0, # Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
        "repetition_penalty": 1.0, # Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens.
        "logprobs": None, # Number of log probabilities to return per output token. When set to None, no probability is returned. If set to a non-None value, the result includes the log probabilities of the specified number of most likely tokens, as well as the chosen tokens. Note that the implementation follows the OpenAI API: The API will always return the log probability of the sampled token, so there may be up to logprobs+1 elements in the response.
        "prompt_logprobs": None, # Number of log probabilities to return per prompt token.
    }
    default_other_params.update(other_params)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model_name,
        "messages": messages,
        **default_other_params
    }

    to_return = {
        "status_code": "unknow",
        "content": {}
    }

    # print(json.dumps(payload, ensure_ascii=False, indent=2))
    # 请求
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.Timeout:
        to_return["status_code"] = "timeout"
        to_return["content"] = "timeout"
    except Exception as e:
        to_return["status_code"] = "todo"
        to_return["content"] = str(e)
    else:
        try:
            to_return["status_code"] = str(response.status_code)
            to_return["content"] = response.json()
        except Exception as e:
            to_return["status_code"] = "response.json() error"
            to_return["content"] = str(e)

    return to_return


if __name__ == "__main__":
    url = "http://localhost:8000/v1/chat/completions"
    api_key = "vllm-01"
    model_name="/data/yuguangya/ALLYOUNEED/VL/Qwen2-VL-7B-Instruct"
    other_params={
        "temperature": 0,
        "max_tokens": 100
    }

    messages_t=[
        {"role": "system", "content": "你是一个nlp专家"}, 
        {"role": "user", "content": "你是谁"}
    ]

    res = send_request_vllm_chat(url=url, api_key=api_key, model_name=model_name, messages=messages_t, other_params=other_params)
    print(res)
