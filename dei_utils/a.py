import os
import torch
import time
from datetime import datetime
import pynvml
import requests
import sys

def barkbark(title="Done", body="Done", num=1):
    body_with_bark = f"{body}?sound=healthnotification"
    bark(title, body_with_bark, num)

def bark(title="Done", body=None, num=1):
    key = os.getenv('BARK_KEY')
    if not key:
        print("Error: BARK_KEY not found", file=sys.stderr)
        return
    num = int(num)
    file_path = os.path.abspath(__file__)
    for i in range(num):
        if num > 1:
            title_with_num = f"{str(title)}_#{i}_{num-1}#"
        else:
            title_with_num = title
        if body:
            url = f"https://api.day.app/{key}/{title_with_num}/{body}"
        else:
            url = f"https://api.day.app/{key}/{title_with_num}"
        try:
            print(f'\n### BARK FROM {file_path} ###\n------\n{url}\n------\n### BARK FROM {file_path} ###\n')
            requests.get(url, timeout=10)
        except Exception as e:
            print(f"Notification failed: {e}", file=sys.stderr)

def compute_bits(a,b,c,d):
    n = min(a,c)
    x = max(a,c)
    k = ((32-x)/2+(x-n)+n*2)/32
    n = min(b,d)
    x = max(b,d)
    v = ((32-x)/2+(x-n)+n*2)/32
    return (k+v)/2

def debug(x):
    def _test(b,n,s,anno):
        s+=('\n'+n*'\t'+anno+': '+str(type(b))[8:][:-2])
        if isinstance(b,torch.Tensor):
            s+=f'{str(b.shape)[11:][:-1]},dtype={b.dtype}'
        elif isinstance(b,list) or isinstance(b,tuple):
            s+=f'[{len(b)}]'
            k = len(b)
            if k >10:
                k=2
            for i in range(k):
                s = _test(b[i],n+1,s,str(i))
        return s
    print(_test(x,0,'',''))

def save(file, path):
    os.makedirs(os.path.expanduser(f'~/data'), exist_ok=True)
    path = f'~/data/{path}.pt'
    path = os.path.expanduser(path)
    try:
        print(f'Saving tensor to {path}')
        torch.save(file, path)
    except Exception as e:
        print(f"Error saving tensor: {e}")

def store(file, path):
    os.makedirs(os.path.expanduser(f'~/data'), exist_ok=True)
    path = f'~/data/{path}.pt'
    path = os.path.expanduser(path)
    if os.path.exists(path):
        l = torch.load(path)
    else:
        l = []
    if isinstance(file,list):
        l+=file
    else:
        l.append(file)
    try:
        # print(f'Saving tensor to {path}')
        torch.save(l, path)
    except Exception as e:
        print(f"Error saving tensor: {e}")

def load(path):
    path = f'~/data/{path}.pt'
    path = os.path.expanduser(path)
    try:
        tensor = torch.load(path)
        return tensor
    except Exception as e:
        print(f"Error loading tensor: {e}")
        return None

class Conqueror:
    def __init__(self, interval_sec=5*60, mercy = 5, max_num = 10, sleep_sec=5,):
        self.gpu_list = []
        self.memory_threshold = 0.15
        # self.total_memory = 24
        self.allocated_tensors = {}
        self.interval = interval_sec
        self.gpu_detection_count = {}
        self.mercy = mercy
        self.max_num = max_num
        self.sleep_sec = sleep_sec
        self.cnt=0
        pynvml.nvmlInit()

    def get_available_gpus(self):
        available_gpus = []
        for i in range(torch.cuda.device_count()):
            # allocated_memory = torch.cuda.memory_allocated(f'cuda:{i}')
            # total_memory = torch.cuda.get_device_properties(f'cuda:{i}').total_memory
            # used_percentage = allocated_memory / total_memory
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_percentage = mem_info.used / mem_info.total
            if i not in self.gpu_detection_count:
                self.gpu_detection_count[i] = 0
            if used_percentage < self.memory_threshold:
                self.gpu_detection_count[i] += 1
                # thre = 1 if self.mercy else 0
                if self.gpu_detection_count[i] > self.mercy:
                    available_gpus.append(i)
                else:
                    print(f'{i}: skipped')
            else:
                self.gpu_detection_count[i]=0
        if len(available_gpus) > self.max_num:
            available_gpus = available_gpus[:self.max_num]
        return available_gpus

    def allocate_memory_on_gpus(self, gpus):
        a = 4*torch.rand(len(gpus))//2+9
        b = (64*torch.rand(len(gpus))).round()+512-16
        c = (64*torch.rand(len(gpus))).round()+1024-16
        d = (64*torch.rand(len(gpus))).round()+1024-16
        for i in range(len(gpus)):
            tensor = torch.zeros(int((a[i]*b[i]*c[i]*d[i]).item()), device=f'cuda:{gpus[i]}')
            self.allocated_tensors[i] = tensor
            # print(f"GPU {gpu} 已分配 20GB 的显存。")
        print(f'conquered: {gpus}')
    
    def release_memory(self):
        for gpu, tensor in self.allocated_tensors.items():
            del tensor
            torch.cuda.empty_cache()
            # print(f"GPU {gpu} 的 20GB 显存已释放。")
        self.allocated_tensors.clear()
        torch.cuda.empty_cache()

    def conquer(self):
        while True:
            try:
                self.cnt+=1
                print(f'\n### {self.cnt}')
                print(datetime.now().strftime('%Y-%m-%d, %A, %H:%M:%S'))
                self.release_memory()
                time.sleep(self.sleep_sec)
                available_gpus = self.get_available_gpus()
                if available_gpus:
                    if len(available_gpus) >=10:
                        s=''
                        for g in available_gpus:
                            s=f'{s}{g} '
                        return s
                    print(f'find: {self.gpu_detection_count}')
                    self.allocate_memory_on_gpus(available_gpus)
                else:
                    # print("没有可用的GPU满足要求。")
                    print(f'do nothing: {self.gpu_detection_count}')
                print(f'sleep for {self.interval} seconds')
                time.sleep(self.interval)
            except Exception as e:
                print(f"异常捕获: {e}")
                return '0'
            
    def detect(self):
        while True:
            try:
                self.cnt+=1
                print(f'\n### {self.cnt}')
                print(datetime.now().strftime('%Y-%m-%d, %A, %H:%M:%S'))
                time.sleep(self.sleep_sec)
                available_gpus = self.get_available_gpus()
                if available_gpus:
                    num_available = len(available_gpus)
                    print(f'[FIND] {num_available:02d}')
                    if num_available >= 2:
                        pass
            except Exception as e:
                print(f"异常捕获: {e}")
                return '0'

class Alternator:
    def __init__(self,max_num):
        self.max_num = max_num
        self.value = 0
    
    def next(self):
        self.value = (self.value+1)%self.max_num
        return self.value