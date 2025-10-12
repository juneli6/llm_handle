import os
import asyncio


def dict_to_command_line_args(input_dict: dict):
    result = []
    
    for key, value in input_dict.items():
        if isinstance(value, bool):
            # 如果值是布尔类型的，转换成命令行标志
            if value:
                result.append(f"--{key}")
        elif value == "<>":
            # 如果值是"<>"，也转换成命令行标志
            result.append(f"--{key}")
        else:
            # 普通的键值对
            result.append(f"--{key} {value}")
    
    # 用空格连接所有的命令行参数
    return ' '.join(result)


async def run_shell_script(
        script_path, 
        args: list[str], 
        use_sudo: bool = False, 
        cuda_devices: str = None # "0,1"
    ):
    command = ['/bin/bash', script_path, *args]
    if use_sudo:
        command.insert(0, 'sudo')
    
    if cuda_devices is not None:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = cuda_devices
    else:
        env = None

    # 使用 asyncio.create_subprocess_exec 启动异步任务
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=None, stdout=None, stderr=None,
        cwd=None, env=env
    )
    return process
