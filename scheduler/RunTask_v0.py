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

from ..llm_utils.llm_logger import llm_logger



class IO_Handle:

    num_retry = 0

    def __init__(self, num_retry=0):
        self.num_retry = num_retry
    
    def _retry(self, func, *args, **kwargs):
        last_exception = None
        current_delay = 1
        backoff = 2
        
        for attempt in range(self.num_retry + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.num_retry:
                    print(f"func failed: {str(e)}. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff
                else:
                    print(f"func failed after {self.num_retry} retries.")
        raise last_exception

    def _load_jsonl(self, file_path, errors='replace'):
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
    
    def _dump_jsonl(self, data, file_path):
        with open(file_path, 'w', encoding='utf-8') as jsonl_file:
            if isinstance(data, list):
                for item in data:
                    jsonl_file.write(json.dumps(item, ensure_ascii=False, indent=None) + '\n')
            else:
                jsonl_file.write(json.dumps(data, ensure_ascii=False, indent=None) + '\n')
        return
    
    
    def ensure_directory(self, name):
        self._retry(os.makedirs, name=name, exist_ok=True)
        return

    def load_jsonls_from_folder(self, folder):
        data = []
        filenames = [i for i in self._retry(os.listdir, folder) if i.endswith(".jsonl")]
        for filename in filenames:
            file_path = os.path.join(folder, filename)
            data_i = self._retry(self._load_jsonl, file_path)
            data.extend(data_i)
        return data
    
    def dumps_to_jsonls_folder(self, data, folder, chunk_size=128):
        """ 将数据分割成多个JSONL文件并保存到指定文件夹
        """
        self.ensure_directory(folder)

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
            
            file_path = os.path.join(folder, f"{i}.jsonl")
            
            self._retry(self._dump_jsonl, chunk_data, file_path)
            print(f"生成文件 {file_path}，包含数据：{start_idx}-{end_idx}")
        return
    
    def find_latest_path(self, directory, prefix=None, suffix=None, time_format = "%Y%m%d_%H%M%S"):
        """ 找到目录下最新的文件或文件夹
        """
        all_files: list[str] = self._retry(os.listdir, directory)
        
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
            # 假设删除 prefix 和 suffix 之后刚好是 time 部分
            time_part = filename
            if prefix:
                time_part = time_part.removeprefix(prefix)
            if suffix:
                time_part = time_part.removesuffix(suffix)
            
            file_time = datetime.strptime(time_part, time_format)
            
            if latest_time is None or file_time > latest_time:
                latest_time = file_time
                latest_file = filename
        
        return os.path.join(directory, latest_file)


class ResourceIterator:
    def __init__(self, resources: dict[str, list] = None, parallel_mode: str = "process"):
        """ 以 "@LOCK" 结尾是锁资源
        """
        self.resources = resources or {}
        self.parallel_mode = parallel_mode
        
        # 锁资源和普通资源
        self.lock_resources = {}
        self.normal_resources = {}
        
        for key, values in self.resources.items():
            if not values:
                raise ValueError(f"Resource list for '{key}' cannot be empty")
            
            if key.endswith("@LOCK"):
                lock_key = key[:-5]
                self.lock_resources[lock_key] = values
            else:
                self.normal_resources[key] = values
        
        assert parallel_mode in ["process", "thread"]
        
        # 全局锁和计数器
        if parallel_mode == "process":
            self.manager = SyncManager()
            self.manager.start()
            self.global_lock = self.manager.Lock()
            self.counters = self.manager.dict()
            self.lock_states = self.manager.dict()
        else:
            self.global_lock = threading.Lock()
            self.counters = {}
            self.lock_states = {}
        
        for key in self.normal_resources:
            self.counters[key] = 0
        
        # 初始化锁资源状态
        for key, values in self.lock_resources.items():
            self.lock_states[key] = self.manager.list([False] * len(values)) if parallel_mode == "process" else [False] * len(values)
            self.counters[key] = 0
    
    def _get_lock_resource(self, key: str, timeout: float = 60.00) -> object:
        """获取锁资源，如果没有可用资源则等待直到超时, 注意外部需要加锁"""
        values = self.lock_resources[key]
        lock_states = self.lock_states[key]
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # 查找第一个未被占用的资源
            for i, (value, is_locked) in enumerate(zip(values, lock_states)):
                if not is_locked:
                    # 标记为已占用并返回
                    lock_states[i] = True
                    return value
            
            # 如果没有找到可用资源，等待一小段时间后重试
            self.global_lock.release()
            try:
                time.sleep(1)
            finally:
                self.global_lock.acquire()
        
        raise RuntimeError(f"获取锁资源超时：{key}")
        # 超时后按普通逻辑获取下一个资源，并标记为占用
        idx = self.counters[key] % len(values)
        result = values[idx]
        lock_states[idx] = True
        self.counters[key] += 1
        print(f"Warning: 锁资源 '{key}' 获取超时，使用轮询方式分配: {result}")
        return result
    
    def _release_lock_resource(self, key: str, resource: object):
        """释放锁资源"""
        values = self.lock_resources[key]
        lock_states = self.lock_states[key]
        
        with self.global_lock:
            for i, value in enumerate(values):
                if value == resource and lock_states[i]:
                    lock_states[i] = False
                    return
    
    def get(self, timeout: float = 3600.00) -> dict[str, object]:
        with self.global_lock:
            result = {}
            
            # 获取锁资源
            for key in self.lock_resources:
                lock_resource = self._get_lock_resource(key, timeout)
                result[key] = lock_resource
            
            # 获取普通资源
            for key, values in self.normal_resources.items():
                idx = self.counters[key] % len(values)
                result[key] = values[idx]
                self.counters[key] += 1

            return result
    
    def release(self, used_resources: dict[str, object]):
        """释放已使用的锁资源"""
        for key, resource in used_resources.items():
            # 处理原始键名
            if key in self.lock_resources:
                self._release_lock_resource(key, resource)
            # 处理带@LOCK后缀的键名
            elif key.endswith("@LOCK") and key[:-5] in self.lock_resources:
                self._release_lock_resource(key[:-5], resource)
    
    def __del__(self):
        if hasattr(self, 'manager'):
            self.manager.shutdown()


class RunTask(object):
    def __init__(
            self, 
            work_func = None, 
            f_input_data: list[dict] = None, 
            resource: dict[str, list] = None, 
            num_parallel: int = 1, 
            parallel_mode: str = "process", # "process" | "thread"
            save_path: str = None, # .jsonl 或者 folder

            # 这两个参数只对 save 为文件夹时生效
            io_num_retry: int = 0, 
            buffer_size: int = 1, # save 程序的缓冲区大小（条）
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
        """
        # 参数
        for item in f_input_data:
            assert item.get("headers") is not None and item["headers"].get("data_id") is not None
        
        assert parallel_mode in ["process", "thread"]

        self.save_mode = "file" if save_path.endswith(".jsonl") else "folder"
        save_path = save_path.removesuffix("/")
        assert not save_path.endswith("/")

        self.io_handle = IO_Handle(num_retry=io_num_retry)

        self.work_func = work_func
        self.resource_iter = ResourceIterator(resources=resource)
        self.parallel_mode = parallel_mode
        self.worker_name = "进程" if parallel_mode == "process" else "线程"
        self.output_queue = multiprocessing.Queue() if parallel_mode == "process" else queue.Queue()

        self.io_handle.ensure_directory(os.path.dirname(save_path))
        if self.save_mode == "file":
            self.previous_save_path = save_path
        else:
            self.previous_save_path = self.io_handle.find_latest_path(
                directory = os.path.dirname(save_path), 
                prefix = os.path.basename(save_path).removesuffix(".jsonl") + "@", 
                suffix = None if self.save_mode == "folder" else ".jsonl"
            )
        self.save_path = save_path + "@" + datetime.now().strftime("%Y%m%d_%H%M%S") if self.save_mode == "folder" else save_path
        self.buffer_size = buffer_size

        # self.f_input_data_grouped
        try:
            llm_logger.info("开始加载已完成的数据")
            if self.save_mode == "file":
                save_path_data = self.io_handle._load_jsonl(save_path)
            else:
                save_path_data = self.io_handle.load_jsonls_from_folder(self.previous_save_path)
            save_path_data = [i for i in save_path_data if i.get("status_code", "").startswith("2")]

            llm_logger.info(f"加载完成，已完成个数：{len(save_path_data)}")
            llm_logger.info(f"准备更新已完成的数据到：{self.save_path}")
            if self.save_mode == "file":
                self.io_handle._dump_jsonl(save_path_data, save_path)
            else:
                self.io_handle.dumps_to_jsonls_folder(save_path_data, self.save_path, chunk_size=buffer_size)
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
            resources = self.resource_iter.get()
            data["headers"]["resource"] = resources
            
            try:
                status_code, res = self.work_func(data)
                output_item = {**data, "res": res, "status_code": status_code}
                self.output_queue.put(json.dumps(output_item, ensure_ascii=False))
            finally:
                self.resource_iter.release(resources)


    def save_results_to_file(self):
        save_folder = os.path.dirname(self.save_path)
        os.makedirs(save_folder, exist_ok=True)
        
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
        bar = tqdm(total=self.num_input_data, desc=f"总体进度({self.worker_name})：")
        try:
            self.io_handle.ensure_directory(os.path.dirname(self.save_path))
            
            buffer = []
            while True:
                json_line = self.output_queue.get()
                
                if json_line is None:
                    # 结束信号
                    if buffer:
                        self.io_handle._retry(
                            self.io_handle._dump_jsonl, 
                            buffer, 
                            os.path.join(self.save_path, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl")
                        )
                    break
                
                json_line = json.loads(json_line)
                buffer.append(json_line)
                
                # 缓冲区达到指定大小时，写入文件
                if len(buffer) >= self.buffer_size:
                    self.io_handle._retry(
                        self.io_handle._dump_jsonl, 
                        buffer, 
                        os.path.join(self.save_path, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl")
                    )
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
        parallel_mode="process", # process | thread
        save_path="output/output_test.jsonl",
        io_num_retry=1, 
        buffer_size=2
    )
    task.start()
