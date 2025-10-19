import os
import sys
import time
import json
import random
import math
import uuid
import itertools
import queue
import threading
import multiprocessing
from datetime import datetime
from multiprocessing.managers import SyncManager
from tqdm import tqdm
from collections import defaultdict
from functools import wraps

from ..llm_utils.llm_logger import llm_logger



def _load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                print("Warning: 跳过空白行")
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: 第{line_num}行JSON解析失败: {e}。\n内容：{line}")
                continue
    return data


def _dump_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as jsonl_file:
        if isinstance(data, list):
            for item in data:
                jsonl_file.write(json.dumps(item, ensure_ascii=False, indent=None) + '\n')
        else:
            jsonl_file.write(json.dumps(data, ensure_ascii=False, indent=None) + '\n')
    return


def load_folder_of_jsonl(file_folder, load_func):
    data = []
    filenames = [i for i in os.listdir(file_folder) if i.endswith(".jsonl")]
    for filename in filenames:
        file_path = os.path.join(file_folder, filename)
        data_i = load_func(file_path)
        data.extend(data_i)
    return data


def dump_folder_of_jsonl(data, file_folder, dump_func, chunk_size=128):
    """ 将数据分割成多个JSONL文件并保存到指定文件夹
    """
    os.makedirs(file_folder, exist_ok=True)
    
    if not isinstance(data, list):
        data = [data]
    
    # 计算需要分割的文件数量
    total_items = len(data)
    num_files = math.ceil(total_items / chunk_size)
    
    print(f"数据总量: {total_items} 条，将分割为 {num_files} 个文件，每个文件最多 {chunk_size} 条")
    
    # 分割数据并保存到多个文件
    for i in range(num_files):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_items)
        chunk_data = data[start_idx:end_idx]
        
        filename = f"{i}.jsonl"
        file_path = os.path.join(file_folder, filename)
        
        dump_func(chunk_data, file_path)
        
        print(f"已生成文件 {filename}，包含 {len(chunk_data)} 条数据")
    
    print(f"已将数据分割为 {num_files} 个文件保存到 {file_folder}")
    return


def find_latest_file(directory, prefix=None, suffix=None):
    """ 找到目录下最新的文件或文件夹
    """
    all_files: list[str] = os.listdir(directory)
    
    filtered_files = []
    for filename in all_files:
        if prefix and not filename.startswith(prefix):
            continue
        if suffix and not filename.endswith(suffix):
            continue
        filtered_files.append(filename)
    
    if not filtered_files:
        return None
    
    # 找到最新的文件
    latest_file = None
    latest_time = None
    
    for filename in filtered_files:
        time_part = filename
        if prefix:
            time_part = time_part.removeprefix(prefix)
        if suffix:
            time_part = time_part.removesuffix(suffix)
        
        file_time = datetime.strptime(time_part, "%Y%m%d_%H%M%S")
        
        if latest_time is None or file_time > latest_time:
            latest_time = file_time
            latest_file = filename
    
    return os.path.join(directory, latest_file)


def retry_on_error(num_retry=6, delay=1, backoff=2):
    """ 重试装饰器 总共执行 1 + num_retry 次
    Args:
        num_retry: 重试次数
        delay: 初始延迟时间(秒)
        backoff: 延迟倍数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(num_retry + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < num_retry:
                        llm_logger.warning(f"Function {func.__name__} failed: {str(e)}. \nRetrying in {current_delay} seconds... ({attempt + 1}/{num_retry})")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        llm_logger.error(f"Function {func.__name__} failed after {num_retry} retries. \nLast error: {str(e)}")
            raise last_exception
        
        return wrapper
    return decorator


class ResourceIterator:
    def __init__(self, resources: dict[str, list] = None, parallel_mode: str = "process"):
        self.resources = resources or {}
        self.parallel_mode = parallel_mode
        
        for key, values in self.resources.items():
            if not values:
                raise ValueError(f"Resource list for '{key}' cannot be empty")
        assert parallel_mode in ["process", "thread"]
        
        # 全局锁和计数器
        if parallel_mode == "process":
            manager = SyncManager()
            manager.start()
            self.global_lock = manager.Lock()
            self.counters = manager.dict()
        else:
            self.global_lock = threading.Lock()
            self.counters = {}
        
        for key in self.resources:
            self.counters[key] = 0
    
    def get(self) -> dict[str, object]:
        with self.global_lock:
            result = {}
            for key, values in self.resources.items():
                idx = self.counters[key] % len(values)
                result[key] = values[idx]
                self.counters[key] += 1
            return result


class RunTask(object):
    def __init__(
            self, 
            work_func = None, 
            f_input_data: list[dict] = None, 
            resource: dict[str, list] = None, 
            num_parallel: int = 1, 
            save_path: str = None, # .jsonl 或者 folder
            parallel_mode: str = "process", # "process" | "thread"
            io_retry: bool = False, 
            buffer_size: int = 1, # 缓冲区大小（条）
            flush_interval: int = 30, # 强制刷新缓冲区的最大时间间隔（秒）
        ):
        """ work_func: 该函数在多个进程中执行，不能有全局变量
            f_input_data: list[dict]; dict:
                {
                    "headers": {"data_id": "xxx"}, # data_id 每个样本不同即可
                    "payload": {...}
                }
            输出的文件: 
                {
                    "headers": {...}, 
                    "payload": {...}
                    "status_code": "200", # status_code 以 "2" 开头代表已完成
                    "res": {...}
                }
            work_func 得到的输入:
                {
                    "headers": {
                        "data_id": "xxx", 
                        "worker_id": "xxx", 
                        "resource": {...}
                    }, 
                    "payload": {...}
                }
            work_func 必须的输出: status_code: str, res

            buffer_size > 1 时，保存的结果注意按照 data_id 去重，防止极端情况有重复保存的数据
        """
        # 参数
        assert work_func is not None and f_input_data is not None
        for item in f_input_data:
            assert item.get("headers") is not None and item["headers"].get("data_id") is not None
        self.save_mode = "file" if save_path.endswith(".jsonl") else "folder"
        assert not save_path.endswith("/")
        assert parallel_mode in ["process", "thread"]

        if io_retry:
            load_jsonl = retry_on_error(num_retry=6)(_load_jsonl)
            dump_jsonl = retry_on_error(num_retry=6)(_dump_jsonl)
        else:
            load_jsonl = _load_jsonl
            dump_jsonl = _dump_jsonl

        self.load_jsonl = load_jsonl
        self.dump_jsonl = dump_jsonl

        self.work_func = work_func
        self.resource_iter = ResourceIterator(resources=resource)
        self.previous_save_path = find_latest_file(
            directory = os.path.dirname(save_path), 
            prefix = os.path.basename(save_path).removesuffix(".jsonl") + "@", 
            suffix = None if self.save_mode == "folder" else ".jsonl"
        )
        self.save_path = save_path + "@" + datetime.now().strftime("%Y%m%d_%H%M%S") if self.save_mode == "folder" else save_path
        self.parallel_mode = parallel_mode
        self.worker_name = "进程" if parallel_mode == "process" else "线程"
        self.output_queue = multiprocessing.Queue() if parallel_mode == "process" else queue.Queue()
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval


        # self.f_input_data_grouped
        try:
            llm_logger.info("开始加载已完成的数据")
            if self.save_mode == "file":
                save_path_data = load_jsonl(save_path)
            else:
                save_path_data = load_folder_of_jsonl(self.previous_save_path, load_func=load_jsonl)
            save_path_data = [i for i in save_path_data if i.get("status_code", "").startswith("2")]

            llm_logger.info(f"加载完成，已完成个数：{len(save_path_data)}")
            llm_logger.info(f"准备更新已完成的数据到：{self.save_path}")
            if self.save_mode == "file":
                dump_jsonl(save_path_data, save_path)
            else:
                dump_folder_of_jsonl(save_path_data, self.save_path, dump_func=dump_jsonl, chunk_size=buffer_size)
            llm_logger.info("已更新")

            save_path_data_ids = set([i["headers"]["data_id"] for i in save_path_data])
            f_input_data = [d for d in tqdm(f_input_data, desc="过滤中") if d["headers"]["data_id"] not in save_path_data_ids]
            llm_logger.info(f"过滤已完成的数据后剩余：{len(f_input_data)}")
        except Exception as e:
            llm_logger.warning(e)

        worker_id_iter = itertools.cycle(list(range(num_parallel)))
        for idx in tqdm(range(len(f_input_data)), desc="分配中"):
            f_input_data[idx]["headers"]["worker_id"] = str(next(worker_id_iter))

        grouped_data = defaultdict(list)
        for fid_item in f_input_data:
            worker_id = fid_item["headers"]["worker_id"]
            grouped_data[worker_id].append(fid_item)
        self.f_input_data_grouped = list(grouped_data.values())


        self.num_input_data = len(f_input_data)
        self.num_parallel = len(self.f_input_data_grouped)
        llm_logger.info(f"任务总长度：{self.num_input_data}")
        llm_logger.info(f"{self.worker_name}个数：{self.num_parallel}")
        llm_logger.info(f"每个{self.worker_name}的任务数：{[len(i) for i in self.f_input_data_grouped]}")


    def process_worker(self, chunk_data):
        for data in chunk_data:
            data["headers"]["resource"] = self.resource_iter.get()
            status_code, res = self.work_func(data)
            output_item = {**data, "res": res, "status_code": status_code}
            self.output_queue.put(json.dumps(output_item, ensure_ascii=False))


    def save_results_to_file(self):
        save_folder = os.path.dirname(self.save_path)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        bar = tqdm(total=self.num_input_data, desc=f"总体进度({self.worker_name})：")
        with open(self.save_path, "a", encoding="utf-8") as f:
            while True:
                json_line = self.output_queue.get()
                if json_line is None:
                    break
                f.write(str(json_line) + "\n")
                f.flush()

                bar.update(1)
                bar.refresh()


    def save_results_to_folder(self):
        @retry_on_error(num_retry=6, delay=1, backoff=2)
        def ensure_directory():
            save_folder = os.path.dirname(self.save_path)
            os.makedirs(save_folder, exist_ok=True)
        
        bar = tqdm(total=self.num_input_data, desc=f"总体进度({self.worker_name})：")
        
        try:
            ensure_directory()
            
            buffer = []
            
            while True:
                json_line = self.output_queue.get()
                json_line = json.loads(json_line)
                
                if json_line is None:
                    # 结束信号
                    if buffer:
                        self.dump_jsonl(buffer, os.path.join(self.save_path, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"))
                    break
                
                buffer.append(json_line)
                
                # 缓冲区达到指定大小时，写入文件
                if len(buffer) >= self.buffer_size:
                    self.dump_jsonl(buffer, os.path.join(self.save_path, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"))
                    time.sleep(1.1)
                    buffer = []

                bar.update(1)
                bar.refresh()
        
        except Exception as e:
            llm_logger.error(f"save 进程失败: {e}")
            raise

        finally:
            bar.close()


    def start_process(self):
        llm_logger.info("启动多进程任务")
        processes = []
        for chunk_data in self.f_input_data_grouped:
            p = multiprocessing.Process(target=self.process_worker, args=(chunk_data,))
            p.start()
            processes.append(p)
        
        save_results = self.save_results_to_file if self.save_mode == "file" else self.save_results_to_folder
        save_process = multiprocessing.Process(target=save_results, args=())
        save_process.start()

        # 等待所有进程完成
        for p in processes:
            p.join()
        time.sleep(3)
        self.output_queue.put(None)
        save_process.join()
        llm_logger.info("所有进程已完成")


    def start_thread(self):
        llm_logger.info("启动多线程任务")
        threads = []
        for chunk_data in self.f_input_data_grouped:
            t = threading.Thread(target=self.process_worker, args=(chunk_data,))
            t.start()
            threads.append(t)
        
        save_results = self.save_results_to_file if self.save_mode == "file" else self.save_results_to_folder
        save_thread = threading.Thread(target=save_results, args=())
        save_thread.start()

        for t in threads:
            t.join()
        time.sleep(3)
        self.output_queue.put(None)
        save_thread.join()
        llm_logger.info("所有线程已完成")


    def start(self):
        if self.parallel_mode == "process":
            self.start_process()
        elif self.parallel_mode == "thread":
            self.start_thread()
        else:
            raise


def hello_world(input_data: dict, prompt="提示词"):
    time.sleep(random.choice([1]))
    resource = input_data["headers"]["resource"]
    payload = input_data["payload"]
    res = f"{prompt}:{resource}:{payload}"
    return "200", res



if __name__ == '__main__':
    f_input_data = [
        {"headers": {"data_id": "1"}, "payload": {"content": "value1"}},
        {"headers": {"data_id": "2"}, "payload": {"content": "value2"}},
        {"headers": {"data_id": "3"}, "payload": {"content": "value3"}},
        {"headers": {"data_id": "4"}, "payload": {"content": "value4"}},
        {"headers": {"data_id": "5"}, "payload": {"content": "value5"}},
        {"headers": {"data_id": "6"}, "payload": {"content": "value6"}},
        {"headers": {"data_id": "7"}, "payload": {"content": "value7"}},
        {"headers": {"data_id": "8"}, "payload": {"content": "value8"}}
    ]
    resource = {
        "url_apikey": [{"url": "1", "key": 1}, {"url": "2", "key": 2}], 
        "temp": ["1", "2", "3", "4", "5"], 
    }

    task = RunTask(
        work_func=hello_world, 
        f_input_data=f_input_data,
        resource=resource,
        num_parallel=3,
        save_path="output/output_test.jsonl",
        parallel_mode="process", # process | thread
        io_retry=True
    )
    task.start()
