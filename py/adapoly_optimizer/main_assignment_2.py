# BY2406211 胡阳

## pseudo code
##
# // 初始化
# 输入：M1 (道路长度矩阵), M2 (道路费用矩阵), max_cost (费用约束1500)
# 输出：最短路径长度, 对应路径

# 1. 定义堆 min_heap
# 2. 初始化：
#     将城市编号，当前长度，当前花费 (0, 0，0) 加入 min_heap
#     设置数组prev全为-1与distance全为inf
#     设置到达0号城市距离distance[0]为0
# 3. while min_heap 非空:
#     从 min_heap 中取出一个更新的存储信息 城市编号，当前长度，当前花费
#       for i in 所有城市:
#           if 当前编号城市可到达编号i城市
#               计算长度new_length和花费new_cost
#           if new_cost <= max_cost and (distance[i] == float('inf') or new_length < distance[i]):
#               将城市编号，当前长度，当前花费 (i, new_length, new_cost) 加入 min_heap
#               更新distance[i] = new_length与prev[i] = city_index          
#             
# 4. 返回到达乙城市最短距离 distance[-1]与数组prev
# 5. 通过数组prev迭代追溯最短路径
# 输出：The shortest distance is: 469
#      The path is: 0 2 7 11 13 20 22 25 31 33 39 42 45 49
##

import heapq

length_file = "/lichenghao/huY/ada_optimizer/submit/m1.txt"
cost_file = "/lichenghao/huY/ada_optimizer/submit/m2.txt"


unreachable = 9999
city_num = 50
max_cost = 1500

class NodeInfo:
    def __init__(self, city_index, current_length, current_cost):
        self.city_index = city_index
        self.current_length = current_length
        self.current_cost = current_cost

    def __lt__(self, other):
        return self.current_length < other.current_length

##读取以行分隔的矩阵，转化为2维数组
def read_file(filename):
    with open(filename, 'r') as file:
        data = []
        for line in file:
            row = list(map(int, line.split()))
            data.append(row)
    return data

#函数查找最短路径
def shortest_path(node_dist, road_cost):
    min_heap = []
    prev = [-1] * city_num
    distance = [float('inf')] * city_num
    distance[0] = 0
    heapq.heappush(min_heap, NodeInfo(0, 0, 0))

    while min_heap:
        current_node = heapq.heappop(min_heap)
        city_index = current_node.city_index
        current_length = current_node.current_length
        current_cost = current_node.current_cost

        # 采用单源最短路径遍历所有城市
        for i in range(city_num):
            if node_dist[city_index][i] != unreachable:
                new_length = current_length + node_dist[city_index][i]
                new_cost = current_cost + road_cost[city_index][i]
                if new_cost <= max_cost and (distance[i] == float('inf') or new_length < distance[i]):
                    heapq.heappush(min_heap, NodeInfo(i, new_length, new_cost))
                    distance[i] = new_length
                    prev[i] = city_index

    return distance[-1] if distance[-1] != float('inf') else None, prev

def print_path(prev, location):
    if prev[location] == -1:
        print(location, end=' ')
        return
    print_path(prev, prev[location])
    print(location, end=' ')

def main():
    #读取不同城市间的距离矩阵
    node_map = read_file(length_file)
    #读取不同城市间的收费矩阵
    cost_map = read_file(cost_file)
    path_length, prev = shortest_path(node_map, cost_map)
    if path_length is not None:
        print(f"The shortest distance is: {path_length}")
        print("The path is: ", end='')
        print_path(prev, city_num - 1)
        print()
    else:
        print("There is no path supported.")

if __name__ == "__main__":
    main()

