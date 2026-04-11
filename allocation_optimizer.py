import math
import itertools

def calc_system_reliability(r, x):
    """计算当前分配方案下的系统总可靠度"""
    Rs = 1.0
    for i in range(len(r)):
        Ri = 1 - (1 - r[i]) ** x[i]
        Rs *= Ri
    return Rs

def calc_total_cost(c, x):
    """计算当前分配方案的总成本"""
    return sum(c[i] * x[i] for i in range(len(c)))

# ==========================================
# 1. 贪心算法（快速得到一个近似最优解和成本上限）
# ==========================================
def greedy_spare_allocation(r, c, w, R_target):
    n = len(r)
    x = [1] * n  
    iterations = 0
    Rs = calc_system_reliability(r, x)
    
    while Rs < R_target:
        best_i = -1
        max_gi = -1.0
        for i in range(n):
            Ri_curr = 1 - (1 - r[i]) ** x[i]
            Ri_next = 1 - (1 - r[i]) ** (x[i] + 1)
            delta_Ri = Ri_next - Ri_curr 
            
            # 综合贪心指数
            gi = (delta_Ri / c[i]) * w[i] * (1 - Ri_curr)
            
            if gi > max_gi:
                max_gi = gi
                best_i = i
                
        x[best_i] += 1
        iterations += 1
        Rs = calc_system_reliability(r, x)
        
    cost = calc_total_cost(c, x)
    return x, Rs, cost, iterations

# ==========================================
# 2. 回溯法 + 剪枝（利用贪心结果作为初始边界，寻找全局最优）
# ==========================================
def backtracking_optimal_allocation(r, c, R_target, initial_best_cost, initial_best_x):
    n = len(r)
    
    # ���录全局最优解
    best_cost = initial_best_cost
    best_x = list(initial_best_x)
    best_Rs = calc_system_reliability(r, best_x)
    
    # 回溯统计参数
    search_nodes = 0

    def dfs(idx, current_cost, current_Rs, current_x):
        nonlocal best_cost, best_x, best_Rs, search_nodes
        search_nodes += 1
        
        # 【剪枝 1：成本超限剪枝】
        if current_cost >= best_cost:
            return
            
        # 【剪枝 2：可靠度理论上限剪枝】
        if current_Rs < R_target:
            return

        # 终止条件：所有子系统都分配完毕
        if idx == n:
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = list(current_x)
                best_Rs = current_Rs
            return

        # 对当前第 idx 个子系统尝试不同的备件数量 xi
        xi = 1
        while True:
            added_cost = c[idx] * xi
            if current_cost + added_cost >= best_cost:
                break
                
            Ri = 1 - (1 - r[idx]) ** xi
            next_Rs = current_Rs * Ri
            
            current_x.append(xi)
            dfs(idx + 1, current_cost + added_cost, next_Rs, current_x)
            current_x.pop() 
            
            xi += 1 

    dfs(0, 0, 1.0, [])
    
    return best_x, best_Rs, best_cost, search_nodes

# ==========================================
# 3. 主程序测试
# ==========================================
if __name__ == "__main__":
    r_components = [0.85, 0.90, 0.70, 0.88] 
    c_components = [5, 4, 3, 6]              
    w_importance = [1.0, 1.0, 1.2, 1.0]      
    R_target = 0.95                          
    
    print("=== 1. 执行贪心算法 ===")
    x_greedy, Rs_greedy, cost_greedy, iters = greedy_spare_allocation(r_components, c_components, w_importance, R_target)
    print(f"分配方案: x = {x_greedy}")
    print(f"可靠度: Rs = {Rs_greedy:.12f}")
    print(f"总成本: Cost = {cost_greedy}")
    print(f"迭代次数: {iters}\n")
    
    print("=== 2. 执行回溯+剪枝 (以贪心Cost为上限) ===")
    x_opt, Rs_opt, cost_opt, search_nodes = backtracking_optimal_allocation(
        r_components, c_components, R_target, 
        initial_best_cost=cost_greedy, 
        initial_best_x=x_greedy
    )
    
    print(f"全局最优方案: x* = {x_opt}")
    print(f"最高性价比可靠度: Rs* = {Rs_opt:.12f}")
    print(f"最低总成本: Cost* = {cost_opt}")
    print(f"回溯搜索节点数: {search_nodes} (得益于剪枝，计算量极小)")
    
    if cost_greedy > cost_opt:
        print(f"\n=> 结论：回溯法成功纠正了贪心算法的局部最优，节省了成本 {cost_greedy - cost_opt}！")
    else:
        print("\n=> 结论：贪心算法初始给出的即为全局最优解！")
