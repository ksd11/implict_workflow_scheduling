


from intervaltree import IntervalTree, Interval
import math

class Core:
    def __init__(self, idx):
        self.idx = idx
        # [(start, end), ...] 按start排序的不相交区间。其实每个区间都是不重叠的
        self.intervals = [(0, float('inf'))]
        
    def _binary_search(self, target):
        # 二分查找第一个end > target的区间
        left, right = 0, len(self.intervals)
        while left < right:
            mid = (left + right) // 2
            if self.intervals[mid][1] <= target:
                left = mid + 1
            else:
                right = mid
        return left if left < len(self.intervals) else -1

    def occupy(self, start, end, timestamp=None):
        # O(log n)找到被占用区间的位置
        idx = self._binary_search(start)
        if idx == -1:
            return
            
        s, e = self.intervals[idx]
        if start < s or end > e:
            assert False, "occupy error"
            
        # 分割区间
        new_intervals = []
        if s < start:
            new_intervals.append((s, start))
        if end < e:
            new_intervals.append((end, e))
            
        # O(n)更新区间列表
        self.intervals[idx:idx+1] = new_intervals
        
        # 移除过期区间 O(n)
        if timestamp is not None:
            idx = self._binary_search(timestamp) # 小于idx对应的区间，区间end <= timestamp
            if idx > 0:
                self.intervals = self.intervals[idx:]
                
    def find_est(self, start, size) -> float:
        # 遍历所有可用区间，找到第一个满足大小要求的
        idx = self._binary_search(start) # idx对应的区间，区间end > start
        while idx < len(self.intervals):
            s, e = self.intervals[idx]
            real_start = max(s, start)
            # 检查区间大小是否满足要求
            if e - real_start >= size:
                return real_start
            idx += 1
                
        assert False, "no available slot"


import unittest

class TestCore(unittest.TestCase):
    def setUp(self):
        self.core = Core(idx=0)
        
    def test_basic_operations(self):
        # 测试基本占用
        self.core.occupy(1, 3)
        self.assertEqual(self.core.intervals, [(0, 1), (3, float('inf'))])
        
        # 测试中间占用
        self.core.occupy(4, 6)
        self.assertEqual(self.core.intervals, [(0, 1), (3, 4), (6, float('inf'))])
        
    def test_find_est(self):
        # 占用区间后查找
        self.core.occupy(1, 3) # [0, 1), [3,inf)
        
        # 测试不同起始位置
        self.assertEqual(self.core.find_est(0, 1), 0)  # 在开头找到空间
        self.assertEqual(self.core.find_est(2, 1), 3)  # 在已占用区间后找到空间
        self.assertEqual(self.core.find_est(3, 1), 3)  # 刚好在边界

        self.core.occupy(10, 100) # [0, 1), [3, 10), [100,inf)
        self.assertEqual(self.core.find_est(2, 8), 100)
        self.assertEqual(self.core.find_est(2, 7), 3)

        
    def test_timestamp_cleanup(self):
        # 设置多个区间
        self.core.occupy(1, 2)
        self.core.occupy(3, 4)
        self.core.occupy(5, 6) # [0,1), [2,3), [4,5), [6, inf)
        self.assertEqual(len(self.core.intervals), 4)
        
        # 清理时间戳之前的区间
        self.core.occupy(7, 8, timestamp=4) # [4,5), [6, 7), [8, inf)
        # 应该只保留时间戳4之后的区间
        self.assertEqual(len(self.core.intervals), 3)
        
    def test_error_cases(self):
        # 测试区间不存在的情况
        self.core.occupy(1, 3) # [0, 1), [3, inf)
        
        # 尝试占用已占用区间
        with self.assertRaises(AssertionError):
            self.core.occupy(2, 4)
            
        # 尝试查找过大的区间
        # with self.assertRaises(AssertionError):
        #     self.core.find_est(0, float('inf'))

if __name__ == '__main__':
    unittest.main()