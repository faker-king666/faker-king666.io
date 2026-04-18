import matplotlib.pyplot as plt
import math

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==============================
# 1) 10子系统参数
# ==============================
r = [0.60, 0.85, 0.90, 0.75, 0.88, 0.92, 0.82, 0.78, 0.86, 0.89]
c = [80,   10,   5,    60,   8,    3,    12,   40,   15,   7]
w = [1.0,  1.0,  1.0,  1.5,  1.0,  1.0,  1.0,  1.2,  1.0,  1.0]
R_target = 0.98
n = len(r)

# 回溯搜索控制（防止“跑不出来”）
XI_CAP_GLOBAL = 20   # 每个子系统最大备件数硬上限，可按需要调小到12~16
EPS = 1e-12

# ==============================
# 2) 基础函数
# ==============================
def Ri(i, x_i):
    return 1 - (1 - r[i]) ** x_i

def calc_Rs(x):
    Rs = 1.0
    for i in range(n):
        Rs *= Ri(i, x[i])
    return Rs

def calc_cost(x):
    return sum(c[i] * x[i] for i in range(n))

# ==============================
# 3) 贪心策略（传统 + 三个改进）
# ==============================
def choose_traditional(x):
    # 传统：只看绝对可靠度增量
    return max(
        range(n),
        key=lambda i: Ri(i, x[i] + 1) - Ri(i, x[i])
    )

def choose_improve_1(x):
    # 改进1：单位性价比
    return max(
        range(n),
        key=lambda i: (Ri(i, x[i] + 1) - Ri(i, x[i])) / c[i]
    )

def choose_improve_2(x):
    # 改进2：单位性价比 * 最不稳定优先
    def score(i):
        delta = Ri(i, x[i] + 1) - Ri(i, x[i])
        instability = 1 - Ri(i, x[i])
        return (delta / c[i]) * instability
    return max(range(n), key=score)

def choose_improve_3(x):
    # 改进3：单位性价比 * 最不稳定优先 * 重要度
    def score(i):
        delta = Ri(i, x[i] + 1) - Ri(i, x[i])
        instability = 1 - Ri(i, x[i])
        return (delta / c[i]) * instability * w[i]
    return max(range(n), key=score)

def run_and_track(choose_func):
    x = [1] * n
    step_hist = [0]
    cost_hist = [calc_cost(x)]
    rs_hist = [calc_Rs(x)]
    chosen_hist = [-1]

    step = 0
    MAX_STEPS = 5000
    while rs_hist[-1] + EPS < R_target and step < MAX_STEPS:
        i = choose_func(x)
        x[i] += 1
        step += 1
        step_hist.append(step)
        cost_hist.append(calc_cost(x))
        rs_hist.append(calc_Rs(x))
        chosen_hist.append(i)

    return {
        "x": x,
        "step": step_hist,
        "cost": cost_hist,
        "rs": rs_hist,
        "chosen": chosen_hist
    }

# ==============================
# 4) 回溯精修（关键子系统优先）
# ==============================
def backtracking_refine_critical(initial_best_cost, critical_mode="critical"):
    """
    critical_mode:
      - "critical":         score = w_i * (1-r_i)
      - "critical_per_cost":score = w_i * (1-r_i) / c_i
    """
    best_cost = initial_best_cost
    best_x = None

    # ---- 关键顺序：最关键子系统优先 ----
    if critical_mode == "critical_per_cost":
        critical_score = [w[i] * (1 - r[i]) / c[i] for i in range(n)]
    else:
        critical_score = [w[i] * (1 - r[i]) for i in range(n)]

    order = sorted(range(n), key=lambda i: critical_score[i], reverse=True)

    # ---- xi上限：由当前best_cost推导 + 全局硬上限 ----
    # 最低成本基线（每个子系统至少1个）
    min_base_cost = sum(c)
    xi_max = []
    for i in range(n):
        # 在“其它子系统先取1个”前提下，i最多还能加多少
        # c[i] * xi + (min_base_cost - c[i]) < best_cost
        # xi < (best_cost - (min_base_cost - c[i])) / c[i]
        raw = int((best_cost - (min_base_cost - c[i]) - 1) // c[i])
        raw = max(1, raw)
        xi_max.append(min(raw, XI_CAP_GLOBAL))

    # ---- 可达性上界剪枝准备 ----
    # 给定每个节点最大可取xi_max，后缀最大可达可靠度乘积
    Ri_max = [Ri(i, xi_max[i]) for i in range(n)]
    suffix_max = [1.0] * (n + 1)
    for k in range(n - 1, -1, -1):
        i = order[k]
        suffix_max[k] = suffix_max[k + 1] * Ri_max[i]

    # 预先计算前缀最小成本（用于更强成本剪枝）
    # rem_min_cost[k] = 从k到末尾每个至少取1时的最小附加成本
    rem_min_cost = [0] * (n + 1)
    for k in range(n - 1, -1, -1):
        i = order[k]
        rem_min_cost[k] = rem_min_cost[k + 1] + c[i] * 1

    def dfs(k, cur_cost, cur_Rs, x_work):
        nonlocal best_cost, best_x

        # 剪枝1：当前成本 + 剩余最小成本都不可能优于best
        if cur_cost + rem_min_cost[k] >= best_cost - EPS:
            return

        # 剪枝2：即使后面全取上限也达不到目标可靠度
        if cur_Rs * suffix_max[k] + EPS < R_target:
            return

        # 到叶子
        if k == n:
            if cur_Rs + EPS >= R_target and cur_cost < best_cost - EPS:
                best_cost = cur_cost
                candidate = [1] * n
                for idx, val in x_work.items():
                    candidate[idx] = val
                best_x = candidate
            return

        i = order[k]

        # xi从1开始，先找低成本可行解，加速收紧best_cost
        for xi in range(1, xi_max[i] + 1):
            next_cost = cur_cost + c[i] * xi
            if next_cost + rem_min_cost[k + 1] >= best_cost - EPS:
                break

            next_Rs = cur_Rs * Ri(i, xi)
            x_work[i] = xi
            dfs(k + 1, next_cost, next_Rs, x_work)

        x_work.pop(i, None)

    dfs(0, 0.0, 1.0, {})

    if best_x is None:
        # 没找到更优则返回初始上界（通常不会）
        return None, initial_best_cost, None, order, xi_max
    return best_x, best_cost, calc_Rs(best_x), order, xi_max

# ==============================
# 5) 运行：传统、改进1、改进2、改进3、最终集成（改进3+回溯）
# ==============================
res_t = run_and_track(choose_traditional)
res_1 = run_and_track(choose_improve_1)
res_2 = run_and_track(choose_improve_2)
res_3 = run_and_track(choose_improve_3)

# 回溯精修：以改进3终成本作上界
x_bt, cost_bt, rs_bt, order_used, xi_max_used = backtracking_refine_critical(
    initial_best_cost=res_3["cost"][-1],
    critical_mode="critical"   # 可改为 "critical_per_cost"
)

# 构造“最终集成方案”真实轨迹：
# 改进3轨迹 + 回溯精修一步（如果更优）
integrated_step = res_3["step"][:]
integrated_cost = res_3["cost"][:]
integrated_rs = res_3["rs"][:]

if x_bt is not None and cost_bt < res_3["cost"][-1] - EPS:
    integrated_step.append(integrated_step[-1] + 1)
    integrated_cost.append(cost_bt)
    integrated_rs.append(rs_bt if rs_bt is not None else integrated_rs[-1])

# ==============================
# 6) 结果打印
# ==============================
print("======== 结果汇总 ========")
print(f"传统贪心:    cost={res_t['cost'][-1]}, Rs={res_t['rs'][-1]:.6f}, steps={res_t['step'][-1]}")
print(f"改进1(性价比): cost={res_1['cost'][-1]}, Rs={res_1['rs'][-1]:.6f}, steps={res_1['step'][-1]}")
print(f"改进2(+不稳定): cost={res_2['cost'][-1]}, Rs={res_2['rs'][-1]:.6f}, steps={res_2['step'][-1]}")
print(f"改进3(+权重):   cost={res_3['cost'][-1]}, Rs={res_3['rs'][-1]:.6f}, steps={res_3['step'][-1]}")

if x_bt is not None:
    print(f"回溯最优:     cost={cost_bt}, Rs={rs_bt:.6f}, x*={x_bt}")
else:
    print(f"回溯最优:     未找到更优解，沿用改进3成本 {res_3['cost'][-1]}")

print("回溯搜索顺序(子系统下标从0开始):", order_used)
print("各子系统xi上限:", xi_max_used)

# ==============================
# 7) 绘图函数
# ==============================
def plot_step_cost_compare(res_base, res_new, title, new_label):
    plt.figure(figsize=(10, 6))
    plt.plot(res_base["step"], res_base["cost"], marker='o', markersize=3, linewidth=2.0, label='传统贪心')
    plt.plot(res_new["step"], res_new["cost"], marker='o', markersize=3, linewidth=2.0, label=new_label)
    plt.title(title)
    plt.xlabel("迭代步数")
    plt.ylabel("累计成本")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_rs_cost_compare(res_base, res_new, title, new_label):
    plt.figure(figsize=(10, 6))
    plt.plot(res_base["rs"], res_base["cost"], marker='.', markersize=6, linewidth=1.8, label='传统贪心')
    plt.plot(res_new["rs"], res_new["cost"], marker='.', markersize=6, linewidth=1.8, label=new_label)
    plt.axvline(x=R_target, color='red', linestyle='--', linewidth=1.8, label=f'目标可靠度={R_target}')
    plt.title(title)
    plt.xlabel("系统可靠度 Rs")
    plt.ylabel("累计成本")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==============================
# 8) 每加入一次改进，生成一张“与传统对比”图
# ==============================
# 图1：传统 vs 改进1
plot_step_cost_compare(
    res_t, res_1,
    "图1：传统贪心 vs 改进1（单位性价比）- 步数-成本",
    "改进1：单位性价比"
)

# 图2：传统 vs 改进2
plot_step_cost_compare(
    res_t, res_2,
    "图2：传统贪心 vs 改进2（+最不稳定优先）- 步数-成本",
    "改进2：性价比×不稳定优先"
)

# 图3：传统 vs 改进3
plot_step_cost_compare(
    res_t, res_3,
    "图3：传统贪心 vs 改进3（+重要度）- 步数-成本",
    "改进3：性价比×不稳定×重要度"
)

# 图4：传统 vs 最终集成（改进3 + 回溯）
plt.figure(figsize=(10, 6))
plt.plot(res_t["step"], res_t["cost"], marker='o', markersize=3, linewidth=2.0, label='传统贪心')
plt.plot(integrated_step, integrated_cost, marker='o', markersize=3, linewidth=2.0,
         label='最终集成方案（多维贪心+回溯）')

# 回溯最优线
final_bt_cost = cost_bt if x_bt is not None else res_3["cost"][-1]
plt.axhline(y=final_bt_cost, color='purple', linestyle='--', linewidth=2.0,
            label=f'回溯最优成本={final_bt_cost}')

plt.title("图4：传统贪心 vs 最终集成方案（真实回溯并入）- 步数-成本")
plt.xlabel("迭代步数")
plt.ylabel("累计成本")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# （可选）如果你也想保留可靠度-成本对比图，取消下方注释
# plot_rs_cost_compare(res_t, res_1, "图1'：传统 vs 改进1 - 可靠度-成本", "改进1")
# plot_rs_cost_compare(res_t, res_2, "图2'：传统 vs 改进2 - 可靠度-成本", "改进2")
# plot_rs_cost_compare(res_t, res_3, "图3'：传统 vs 改进3 - 可靠度-成本", "改进3")
