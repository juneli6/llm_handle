import os
import sys
import json
import time
import asyncio
import subprocess

work_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_folder)
from llm_utils.llm_logger import llm_logger
from llm_server.utils import dict_to_command_line_args, run_shell_script
from llm_server.vllm_server.call_vllm_chat import send_request_vllm_chat


class VLLM_Server:

    # https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#
    default_load_kwargs = {
        "host": "127.0.0.1", 
        "port": "8000", 
        "model": "", 
        "gpu-memory-utilization": "0.9", 
        "tensor-parallel-size": "1", 
        "pipeline-parallel-size": "1", 

        "dtype": "auto", # Data type for model weights and activations.
        "kv-cache-dtype": "auto", 
        "block-size": "16", 
        "seed": "0", 
        "max-logprobs": "20", 
        "trust-remote-code": True, 

        # "max-model-len": "8192", 
        # "enable-prefix-caching": "<>", 
        # "api-key": "vllm-00"
    }

    def __init__(self, script_root = None):
        if script_root == None:
            script_root = os.path.dirname(os.path.abspath(__file__))
        
        self.deploy_vllm_script_path = os.path.join(script_root, "deploy_vllm.sh")
        self.destroy_vllm_script_path = os.path.join(script_root, "destroy_vllm.sh")

        self.url = None
        self.api_key = None
        self.model_name_or_path = None


    @staticmethod
    def test_if_available(url, api_key, model_name):
        messages = [{"role": "user", "content": "简单介绍一下你自己"}]
        other_params = {
            "max_tokens": 128
        }
        res = send_request_vllm_chat(url=url, api_key=api_key, model_name=model_name, messages=messages, other_params=other_params, timeout=60)
        return res["status_code"], res["content"]


    async def deploy_llm(self, cuda_devices=None, load_kwargs=None):
        # 命令参数
        if load_kwargs:
            self.default_load_kwargs.update(load_kwargs)
        model_name_or_path = self.default_load_kwargs.pop("model")
        if model_name_or_path == "":
            raise ValueError("model 未传入")
        default_load_kwargs_command = model_name_or_path + " " + dict_to_command_line_args(self.default_load_kwargs)

        # 部署
        llm_logger.info(f"deploying {model_name_or_path}, load_kwargs is:")
        llm_logger.info(json.dumps(self.default_load_kwargs, ensure_ascii=False, indent=2))
        service_process = await run_shell_script(self.deploy_vllm_script_path, [default_load_kwargs_command], use_sudo=False, cuda_devices=cuda_devices)
        
        # 检查是否部署成功 - TODO
        while True:
            await asyncio.sleep(30)
            llm_logger.info("测试是否可访问")
            host = self.default_load_kwargs.get("host", "127.0.0.1")
            port = self.default_load_kwargs.get("port", "8000")
            url = f"http://{host}:{port}/v1/chat/completions"
            api_key = self.default_load_kwargs.get("api-key", "EMPTY")
            
            status_code, content = self.test_if_available(url, api_key, model_name_or_path)
            if status_code == "200":
                llm_logger.info(f"{model_name_or_path} 模型已部署")
                llm_logger.info(json.dumps(content, ensure_ascii=False, indent=2))
                break
            else:
                llm_logger.info(json.dumps(str(content), ensure_ascii=False, indent=2))

        self.url = url
        self.api_key = api_key
        self.model_name_or_path = model_name_or_path

        return url, api_key, model_name_or_path


def destroy_llm(mode="kill"):
    """ kill - 终止进程
        list - 仅查看
    """
    destroy_vllm_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "destroy_vllm.sh")

    if mode == "list":
        script_args = ["bash", destroy_vllm_script_path, "--list"]
        llm_logger.info("destroy_llm 仅查看")
    else:
        script_args = ["bash", destroy_vllm_script_path]
        llm_logger.warning("destroy_llm 终止进程")

    try:
        result = subprocess.run(script_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.stdout:
            llm_logger.info(f"脚本输出:\n{result.stdout}")
        if result.stderr:
            llm_logger.error(f"脚本错误信息:\n{result.stderr}")
        
    except Exception as e:
        llm_logger.error(f"调用 destroy_vllm.sh 失败: {e}")



if __name__ == "__main__":
    load_kwargs = {
        "model": ""
    }

    vllm_server = VLLM_Server()
    asyncio.run(vllm_server.deploy_llm(load_kwargs=load_kwargs))

    # time.sleep(30)
    # destroy_llm("list")
    # time.sleep(10)
    # destroy_llm("kill")
