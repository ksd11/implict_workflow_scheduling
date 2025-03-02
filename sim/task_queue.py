import heapq
from dataclasses import dataclass
from typing import Any, Optional
from .cluster2 import Task
@dataclass
class Task:
    task_id: str
    timestamp: float
    data: Any
    
    # 用于优先队列比较
    def __lt__(self, other):
        return self.timestamp < other.timestamp

class TaskQueue:
    def __init__(self):
        self.tasks = []  # 优先队列
        self.task_counter = 0
        
    def add_task(self, timestamp: float, data: Any) -> None:
        """添加任务到队列"""
        task = Task(
            task_id=self.task_counter,
            timestamp=timestamp,
            data=data
        )
        self.task_counter += 1
        heapq.heappush(self.tasks, task)
        
    def get_task(self) -> Optional[Task]:
        """获取最早的任务"""
        if not self.tasks:
            return None
        return heapq.heappop(self.tasks)
    
    def peek(self) -> Optional[Task]:
        """查看最早的任务但不移除"""
        if not self.tasks:
            return None
        return self.tasks[0]
    
    def is_empty(self) -> bool:
        return len(self.tasks) == 0
