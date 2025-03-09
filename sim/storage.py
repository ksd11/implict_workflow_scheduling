from collections import OrderedDict
from typing import Tuple, Any

class Storage:
    def __init__(self, size):
        self.capacity = size  # 缓存容量
        self.used = 0        # 已使用空间
        # key, value => layer_id : (layer_size, download_finish_time)
    
    '''
        往缓存里添加layer_id表示的缓存块，大小为layer_size
        1. 若缓存存在,则啥事也没有
        2. 若缓存不存在,则添加缓存。需要保证缓存总大小不超过size, 若超出，则先驱逐旧的缓存块，再添加新的缓存块。驱逐的算法为LRU
    '''
    def add(self, layer_id, layer_size, download_finish_time=0):
        # 如果已存在，直接返回
        if self.contain(layer_id):
            return
        
        # 检查是否需要腾出空间
        while self.used + layer_size > self.capacity and not self.empty():
            # 移除最久未使用的缓存
            _, info = self.expel()
            removed_size, _ = info
            self.used -= removed_size
            
        # 如果单个layer太大，则不缓存
        if layer_size > self.capacity:
            return
            
        # 添加新缓存
        self.cache(layer_id, (layer_size, download_finish_time))
        self.used += layer_size

    def has_layer(self, L):
        res = []
        for i in range(L):
            if self.contain(i):
                res.append(1)
            else:
                res.append(0)
        return res
    
    '''
        获取层的下载完成时间（用户保证层存在）
    '''
    def get_download_finish_time(self, layer_id):
        self.get(layer_id)[1]

    '''
        获取层的大小（用户保证层存在）
    '''
    def get_layer_size(self, layer_id):
        self.get(layer_id)[0]

    '''
        判断缓存里是否还有layer_id表示的缓存块
    '''
    def contain(self, layer_id):
        pass
    
    '''
        标记layer_id对应的缓存块命中一次
    '''
    def hit(self, layer_id):
        pass
    
    def get_all_layers(self):
        pass
    
    def clear(self):
        pass

    # 返回剩余空间
    def remain(self):
        return self.capacity - self.used
    
    # 从缓存中驱逐一块
    def expel(self):
        pass
    
    # 添加key valeu到缓存
    def cache(self, key, value):
        pass
    
    # 从缓存读key
    def get(self, key):
        pass
    
    # 缓存是否为空
    def empty(self):
        pass

# 先进先出的存储
class FCFSStorage(Storage):
    def __init__(self, size):
        super(FCFSStorage, self).__init__(size)
        self.buffer = OrderedDict()
    
    def contain(self, key):
        return key in self.buffer

    def hit(self, key):
        return

    def get_all_layers(self):
        return set(self.buffer.keys())
    
    def clear(self):
        self.used = 0
        self.buffer = OrderedDict()

    def expel(self):
        return self.buffer.popitem(last=False)
    
    def cache(self, key, value):
        self.buffer[key] = value
    
    def get(self, key):
        return self.buffer.get(key)

    def empty(self):
        return not self.buffer


class LRUStorage(FCFSStorage):
    def __init__(self, size):
        super(LRUStorage, self).__init__(size)

    def hit(self, key):
        if key in self.buffer:
            size = self.buffer.pop(key)
            self.buffer[key] = size


import heapq
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class CacheItem:
    priority: int
    timestamp: int
    key: Any
    value: Any
    
    def __lt__(self, other):
        # 先比较优先级，再比较时间戳
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


class PriorityStorage(Storage):
    def __init__(self, size):
        super().__init__(size)
        self.buffer = {}           # key -> CacheItem
        self.heap = []            # 最小堆，存储CacheItem
        self.timestamp = 0
        self.cacheItem = CacheItem
    
    def contain(self, key):
        return key in self.buffer
        
    def get_all_layers(self):
        return set(self.buffer.keys())
    
    def clear(self):
        self.used = 0
        self.buffer.clear()
        self.heap = []
        self.timestamp = 0
        
    def hit(self, key):
        if key in self.buffer:
            item = self.buffer[key]
            # 创建新项并更新
            new_item = self.cacheItem(
                priority=item.priority + 1,
                timestamp=self.timestamp,
                key=key,
                value=item.value
            )
            self.buffer[key] = new_item
            heapq.heappush(self.heap, new_item) # 里面会有老的记录，在expel处理
            self.timestamp += 1
    
    def expel(self) -> Optional[tuple]:
        while self.heap:
            item = heapq.heappop(self.heap)
            # 检查是否是最新的项
            if self.buffer[item.key] is item:
                del self.buffer[item.key]
                return item.key, item.value
        return None
    
    def cache(self, key, value):
        item = self.cacheItem(1, self.timestamp, key, value)
        self.buffer[key] = item
        heapq.heappush(self.heap, item)
        self.timestamp += 1
        
    def get(self, key):
        return self.buffer[key].value if key in self.buffer else None
        
    def empty(self):
        return not self.buffer


@dataclass
class CacheItemPlus:
    priority: int
    timestamp: int
    key: Any
    value: Any
    
    def __lt__(self, other):
        # 优先级： 命中次数 * layer_size
        selfPriority = self.priority * self.value[0]
        otherPrioruty = other.priority * other.value[0]

        if selfPriority != otherPrioruty:
            return selfPriority < otherPrioruty
        
        # 优先级一样则先进先出
        return self.timestamp < other.timestamp

class PriorityPlusStorage(PriorityStorage):
    def __init__(self, size):
        super(PriorityPlusStorage, self).__init__(size)
        self.cacheItem = CacheItemPlus


######### Test ##########

import unittest

class TestLRUStorage(unittest.TestCase):
    def setUp(self):
        self.capacity = 10
        self.storage = LRUStorage(self.capacity)

    def test_basic_operations(self):
        # 测试添加和获取
        self.storage.add(1, 2)
        self.assertEqual(self.storage.get(1), (2,0))
        
        # 测试不存在的键
        self.assertIsNone(self.storage.get(2))

    def test_lru_feature(self):
        # 添加三个项目
        self.storage.add('A', 3)
        self.storage.add('B', 4)
        self.storage.add('C', 3) ### 存储已满 A -> B -> C
        
        # 访问A，使其成为最近使用
        self.storage.hit('A') ### B -> C -> A
        
        # 添加新项目触发逐出
        self.storage.add('D', 4) ### C -> A -> B
        
        # B应该被逐出（最少使用）
        self.assertIsNone(self.storage.get('B'))
        self.assertIsNotNone(self.storage.get('A'))

    def test_capacity_limit(self):
        # 添加超过容量的项目
        items = [(i, i) for i in range(self.capacity + 2)]
        for key, value in items:
            self.storage.add(key, value)
            
        # 检查容量
        count = len(self.storage.buffer)
        self.assertLessEqual(count, self.capacity)
        
        # 最早的项应该被逐出
        self.assertIsNone(self.storage.get(0))
        self.assertIsNone(self.storage.get(1))


class TestPriorityStorage(unittest.TestCase):
    def setUp(self):
        self.size = 5
        self.storage = PriorityStorage(self.size)
        
    def test_basic_operations(self):
        # 测试添加和获取
        self.storage.add("k1", 2)
        self.assertTrue(self.storage.contain("k1"))
        self.assertEqual(self.storage.get("k1"), (2,0))
        
    def test_priority_order(self):
        # 添加项目
        self.storage.add("k1", 2)
        self.storage.add("k2", 3)
        
        # k1命中增加优先级
        self.storage.hit("k1")
        
        # 添加超出容量的项目
        self.storage.add("k3", 1)
        
        # k2应该被驱逐(优先级低)
        self.assertFalse(self.storage.contain("k2"))
        self.assertTrue(self.storage.contain("k1"))
        
    def test_fcfs_with_same_priority(self):
        # 添加相同优先级的项目
        self.storage.add("k1", 1)
        self.storage.add("k2", 1)
        self.storage.add("k3", 1)
        
        # 添加第四个触发驱逐
        self.storage.add("k4", 3)
        
        # k1应该被驱逐(最早添加)
        self.assertFalse(self.storage.contain("k1"))
        
    # def test_capacity_limit(self):
    #     # 添加超过容量的项目
    #     for i in range(self.size + 2):
    #         self.storage.cache(f"k{i}", f"v{i}")
            
    #     # 检查容量未超限
    #     self.assertEqual(len(self.storage.buffer), self.size)

if __name__ == '__main__':
    unittest.main()
