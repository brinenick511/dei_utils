import os
import torch
import time
from datetime import datetime

def dei_print(x):
    print(f'\n\n--\n{x}\n--\n\n')

def dei_save(path, file):
    os.makedirs(os.path.expanduser(f'~/data'), exist_ok=True)
    path = f'~/data/{path}.pt'
    path = os.path.expanduser(path)
    try:
        print(f'Saving tensor to {path}')
        torch.save(file, path)
    except Exception as e:
        print(f"Error saving tensor: {e}")

def dei_load(path):
    path = f'~/data/{path}.pt'
    path = os.path.expanduser(path)
    try:
        tensor = torch.load(path)
        return tensor
    except Exception as e:
        print(f"Error loading tensor: {e}")
        return None

class Dei_Conqueror:
    def __init__(self, interval_sec=5*60, mercy = 5, sleep_sec=5):
        self.gpu_list = []
        self.memory_threshold = 0.2
        self.total_memory = 24
        self.allocated_tensors = {}
        self.interval = interval_sec
        self.gpu_detection_count = {}
        self.mercy = mercy
        self.sleep_sec = sleep_sec
        self.cnt=0

    def get_available_gpus(self):
        available_gpus = []
        for i in range(torch.cuda.device_count()):
            allocated_memory = torch.cuda.memory_allocated(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory
            used_percentage = allocated_memory / total_memory
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
        
        return available_gpus

    def allocate_memory_on_gpus(self, gpus):
        a = 4*torch.rand(len(gpus))//2+9
        b = (64*torch.rand(len(gpus))).round()+512-64
        c = (64*torch.rand(len(gpus))).round()+1024-64
        d = (64*torch.rand(len(gpus))).round()+1024-64
        for i in gpus:
            tensor = torch.zeros(int((a[i]*b[i]*c[i]*d[i]).item()), device=f'cuda:{i}')
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
            self.cnt+=1
            print(f'\n### {self.cnt}')
            print(datetime.now().strftime('%Y-%m-%d, %A, %H:%M:%S'))
            self.release_memory()
            time.sleep(self.sleep_sec)
            available_gpus = self.get_available_gpus()
            if available_gpus:
                # print(f"找到可用的GPU: {available_gpus}")
                print(f'find: {self.gpu_detection_count}')
                self.allocate_memory_on_gpus(available_gpus)
            else:
                # print("没有可用的GPU满足要求。")
                print(f'do nothing: {self.gpu_detection_count}')
            time.sleep(self.interval)