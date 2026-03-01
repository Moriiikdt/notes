# 01 背包 capacity是容量 w是物品体积 v是价值
def zero_one_bag(capacity:int, w: List[int], v: List[int]) -> int:
      n = len(w)

      def dfs(i, c):
            if i < 0:
                  return 0
            # 剩余空间小于当前选择的体积 只能不选
            if c < w[i]:
                  return dfs(i-1, c)
            return max(dfs(i-1, c), dfs(i-1, c-w[i])+v[i])
      return dfs(n-1, capacity)