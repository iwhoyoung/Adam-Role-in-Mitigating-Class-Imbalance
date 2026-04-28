# BY2406211 胡阳

## pseudo code
# // 初始化
# 输入：权重矩阵 D（n x n），其中 n 是顶点的数量
# 输出：最小权重三角剖分的和
 
# X[n][n] = 初始化为一个大值（如 float('inf')）
 
# // 创建一个二维数组，用于记录最佳分割点的位置
# k_best[n][n] = 初始化为 -1
 
# // 遍历所有可能的顶点对 (i, j)，其中 i < j 且 j > i + 1（确保不相邻）
# for i from 0 to n-1 do
#     for j from i+2 to n-1 do
#         // 尝试所有可能的分割点 k，其中 i < k < j
#         for k from i+1 to j-1 do
#             // 计算当前三角剖分的权重和
#             W = D[i][k] + D[k][j] + D[i][j]
#          
#             // 如果当前权重和小于 X[i][j] 中存储的值，则更新 X[i][j] 和 k_best[i][j]
#             if W < X[i][j] then
#                 X[i][j] = W
#                 k_best[i][j] = k
 
# // 输出： X[0][n-1]
# 最小权重和: 277
# 三角形顶点：v1, v8 和 v2
# 三角形顶点：v2, v8 和 v3
# 三角形顶点：v3, v8 和 v4
# 三角形顶点：v4, v8 和 v7
# 三角形顶点：v4, v7 和 v5
# 三角形顶点：v5, v7 和 v6
##

def min_weight_triangulation(D):
    # D 是一个 n x n 的矩阵，其中 n 是顶点的数量
    # D[i][j] 表示顶点 vi 和 vj 之间的权重
    n = len(D)
    
    # X 数组用于存储最小权重和
    X = [[float('inf')] * n for _ in range(n)]
    
    # 用于记录最佳分割点的数组
    k_best = [[-1] * n for _ in range(n)]
    for i in range(n): 
        X[i][i] = 0
    for i in range(n-1): 
        X[i][i+1] = 0
    for r in range(2,n): # 按顶点数遍历
        for i in range(1,n-r+1): 
            j = i+r-1
            # 对于包含顶点(i-1)~j的多边形先计算默认分割点K为i的数值替换最小权重和矩阵X的inf值与分割点矩阵k_best的-1
            X[i-1][j]=X[i][j] + D[i-1][i] + D[i-1][j] + D[i][j]
            k_best[i-1][j] = i
            for k in range(i + 1, j):  # 尝试剩余所有可能的分割点 k
                W = D[i-1][k] + D[k][j] + D[i-1][j] 
                if X[i-1][k]+X[k][j]+W < X[i-1][j]:
                    X[i-1][j] = X[i-1][k]+X[k][j]+W
                    k_best[i-1][j] = k
    
    # 最终的答案是 X[0][n-1]
    
    # 输出最小权重和
    print("最小权重和:", X[0][7])
    
    # 输出弦
    def print_triangles(i, j):
        if i+1 < j:
            k = k_best[i][j]
            print(f"三角形顶点：v{i+1}, v{j+1} 和 v{k+1}")  # 输出三角形
            print_triangles(i, k)
            print_triangles(k, j)
    
    # 从 X[0][7] 开始回溯
    print_triangles(0, 7)
        


if __name__ == '__main__':
    D = [
    [0, 14, 25, 27, 10, 11, 24, 16],
    [0, 0, 18, 15, 27, 28, 16, 14],
    [0, 0, 0, 19, 14, 19, 16, 10],
    [0, 0, 0, 0, 22, 23, 15, 14],
    [0, 0, 0, 0, 0, 14, 13, 20],
    [0, 0, 0, 0, 0, 0, 15, 18],
    [0, 0, 0, 0, 0, 0, 0, 27],
    [0, 0, 0, 0, 0, 0, 0, 0]]
    min_weight_triangulation(D)
