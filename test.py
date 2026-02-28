class Node:
    __slots__ = 'prev', 'next', 'key', 'value'

    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.dummy = Node()
        self.dummy.pre = self.dummy
        self.dummy.next = self.dummy
        self.key2node = {}

    # 获取节点，把节点移到链表头部
    def get_node(self, key):
        if key not in self.key2node:
            return None
        node = self.key2node[key]
        # 先抽
        self.remove(node)
        # 再推到头部
        self.push_front(node)
        return node
    
    def get(self, key):
        node = self.get_node(key)
        return node.value if node else -1

    def put(self, key, value):
        # 如果有就先移
        node = self.get_node(key)
        if node:
            node.value = value
            return
        # 没有就新建
        self.key2node[key] = node = Node(key, value)
        self.push_front(node)
        # 超额 就删除最后一个
        if len(self.key2node) > self.capacity:
            back = self.dummy.prev
            del self.key2node[back.key]
            self.remove(back)

    # 删除
    def remove(self, x):
        # 先断前
        x.prev.next = x.next
        # 断后
        x.next.prev = x.prev

    # 推到最前
    def push_front(self, x):
        # 先管自己
        x.prev = self.dummy
        x.next = self.dummy.next
        # 再管插入后的前后
        x.prev.next = x
        x.next.prev = x

