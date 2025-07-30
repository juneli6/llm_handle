import os
import sys
import time
import json
import random
import itertools
import queue
import threading
import multiprocessing
from multiprocessing.managers import SyncManager
from tqdm import tqdm
from collections import defaultdict

from ..llm_utils.llm_logger import llm_logger
from ..llm_utils.utils import load_jsonl, dump_jsonl


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
            save_path: str = None, 
            parallel_mode: str = "process" # "process" | "thread"
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
        assert work_func is not None and f_input_data is not None
        for item in f_input_data:
            assert item.get("headers") is not None and item["headers"].get("data_id") is not None
        assert save_path is not None and save_path.endswith(".jsonl")
        assert parallel_mode in ["process", "thread"]


        self.work_func = work_func
        self.resource_iter = ResourceIterator(resources=resource)
        self.save_path = save_path
        self.parallel_mode = parallel_mode
        self.worker_name = "进程" if parallel_mode == "process" else "线程"
        self.output_queue = multiprocessing.Queue() if parallel_mode == "process" else queue.Queue()


        # self.f_input_data_grouped
        try:
            llm_logger.info("开始加载已完成的数据")
            save_path_data = [i for i in load_jsonl(save_path) if i.get("status_code", "").startswith("2")]
            llm_logger.info(f"加载完成，已完成个数：{len(save_path_data)}")
            llm_logger.info(f"准备更新已完成的数据到：{save_path}")
            dump_jsonl(save_path_data, save_path)
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


    def save_results(self):
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


    def start_process(self):
        llm_logger.info("启动多进程任务")
        processes = []
        for chunk_data in self.f_input_data_grouped:
            p = multiprocessing.Process(target=self.process_worker, args=(chunk_data,))
            p.start()
            processes.append(p)
        
        save_process = multiprocessing.Process(target=self.save_results, args=())
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
        
        save_thread = threading.Thread(target=self.save_results, args=())
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
        parallel_mode="process" # process | thread
    )
    task.start()
