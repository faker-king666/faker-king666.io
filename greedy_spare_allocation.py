"""基于贪心算法的串联系统备件分配（Python源码）.

功能:
1) 在满足系统可靠度约束 Rs >= R0 的前提下，使用贪心法最小化备件成本;
2) 提供小规模穷举验证函数，用于对比贪心结果;
3) 内置示例参数，可直接运行.
"""

import argparse
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import product
from math import log
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Problem:
    """串联系统备件分配问题参数."""

    reliabilities: Sequence[float]  # r_i
    costs: Sequence[float]  # c_i
    target_reliability: float  # R0
    upper_bounds: Optional[Sequence[int]] = None  # 每个子系统最大备件数 U_i

    def __post_init__(self) -> None:
        if len(self.reliabilities) == 0:
            raise ValueError("reliabilities 不能为空")
        if len(self.reliabilities) != len(self.costs):
            raise ValueError("reliabilities 与 costs 长度必须一致")
        if not (0 < self.target_reliability <= 1):
            raise ValueError("target_reliability 必须在 (0, 1] 范围内")
        for r in self.reliabilities:
            if not (0 < r < 1):
                raise ValueError("每个 r_i 必须在 (0, 1) 范围内")
        for c in self.costs:
            if c <= 0:
                raise ValueError("每个 c_i 必须为正")
        if self.upper_bounds is not None:
            if len(self.upper_bounds) != len(self.reliabilities):
                raise ValueError("upper_bounds 长度必须与 reliabilities 一致")
            if any(u < 0 for u in self.upper_bounds):
                raise ValueError("upper_bounds 中每个 U_i 必须 >= 0")


@dataclass
class Solution:
    x: List[int]
    system_reliability: float
    total_cost: float
    iterations: int


def subsystem_reliability(r: float, x_i: int) -> float:
    """R_i(x_i) = 1 - (1-r_i)^(x_i+1)."""
    return 1.0 - (1.0 - r) ** (x_i + 1)


def system_reliability(reliabilities: Sequence[float], x: Sequence[int]) -> float:
    rs = 1.0
    for r, x_i in zip(reliabilities, x):
        rs *= subsystem_reliability(r, x_i)
    return rs


def greedy_allocate(problem: Problem, use_heap: bool = True) -> Solution:
    """贪心求解.

    指标: G_i = Δln(R_i) / c_i
    每轮给 G_i 最大的子系统增加 1 个备件，直到 Rs >= R0.

    Args:
        problem: 问题参数
        use_heap: True 使用最大堆优化; False 使用朴素全扫描

    Raises:
        RuntimeError: 在给定上界下无法达到目标可靠度
    """

    r = list(problem.reliabilities)
    c = list(problem.costs)
    m = len(r)
    ub = list(problem.upper_bounds) if problem.upper_bounds is not None else [10**9] * m

    x = [0] * m
    rs = system_reliability(r, x)
    if rs >= problem.target_reliability:
        return Solution(x=x, system_reliability=rs, total_cost=0.0, iterations=0)

    def gain(i: int) -> float:
        if x[i] >= ub[i]:
            return float("-inf")
        ri_now = subsystem_reliability(r[i], x[i])
        ri_next = subsystem_reliability(r[i], x[i] + 1)
        delta_l = log(ri_next) - log(ri_now)
        return delta_l / c[i]

    iterations = 0

    if use_heap:
        heap: List[Tuple[float, int]] = []
        for i in range(m):
            g = gain(i)
            heappush(heap, (-g, i))  # 最大堆: 存负值

        while rs < problem.target_reliability:
            if not heap:
                raise RuntimeError("堆为空，无法继续分配")

            neg_g, i = heappop(heap)
            current_g = gain(i)

            # 惰性删除: 若键值过期，压回最新值并继续
            if abs((-neg_g) - current_g) > 1e-15:
                heappush(heap, (-current_g, i))
                continue

            if current_g == float("-inf"):
                raise RuntimeError("在给定 upper_bounds 下无法达到目标可靠度")

            x[i] += 1
            iterations += 1
            rs = system_reliability(r, x)

            # 仅该 i 的增益会变化，更新其堆键即可
            heappush(heap, (-gain(i), i))
    else:
        while rs < problem.target_reliability:
            g_list = [gain(i) for i in range(m)]
            i = max(range(m), key=lambda k: g_list[k])
            if g_list[i] == float("-inf"):
                raise RuntimeError("在给定 upper_bounds 下无法达到目标可靠度")

            x[i] += 1
            iterations += 1
            rs = system_reliability(r, x)

    total_cost = sum(ci * xi for ci, xi in zip(c, x))
    return Solution(x=x, system_reliability=rs, total_cost=total_cost, iterations=iterations)


def brute_force_optimal(problem: Problem) -> Solution:
    """小规模穷举最优解（用于验证）.

    要求提供 upper_bounds；否则穷举空间无界。
    """
    if problem.upper_bounds is None:
        raise ValueError("穷举验证必须提供 upper_bounds")

    r = list(problem.reliabilities)
    c = list(problem.costs)
    ub = list(problem.upper_bounds)

    best_x: Optional[List[int]] = None
    best_cost = float("inf")
    best_rs = 0.0

    for x_tuple in product(*(range(u + 1) for u in ub)):
        rs = system_reliability(r, x_tuple)
        if rs >= problem.target_reliability:
            cost = sum(ci * xi for ci, xi in zip(c, x_tuple))
            if cost < best_cost:
                best_cost = cost
                best_rs = rs
                best_x = list(x_tuple)

    if best_x is None:
        raise RuntimeError("穷举未找到可行解，请检查约束或提高 upper_bounds")

    return Solution(x=best_x, system_reliability=best_rs, total_cost=best_cost, iterations=0)


def demo() -> None:
    """示例运行."""
    problem = Problem(
        reliabilities=[0.90, 0.85, 0.80, 0.95],
        costs=[5, 4, 3, 8],
        target_reliability=0.95,
        upper_bounds=[7, 7, 7, 7],
    )

    greedy = greedy_allocate(problem, use_heap=True)
    optimal = brute_force_optimal(problem)

    print("=== 贪心结果 ===")
    print(f"x = {greedy.x}")
    print(f"Rs = {greedy.system_reliability:.12f}")
    print(f"Cost = {greedy.total_cost}")
    print(f"Iterations = {greedy.iterations}")

    print("\n=== 穷举最优结果 ===")
    print(f"x* = {optimal.x}")
    print(f"Rs* = {optimal.system_reliability:.12f}")
    print(f"Cost* = {optimal.total_cost}")

    gap = greedy.total_cost - optimal.total_cost
    print(f"\n成本差距 greedy-optimal = {gap}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="贪心备件分配求解器")
    parser.add_argument(
        "--mode",
        choices=["demo", "greedy", "bruteforce"],
        default="demo",
        help="运行模式: demo(默认), greedy(仅贪心), bruteforce(仅穷举)",
    )
    parser.add_argument(
        "--naive",
        action="store_true",
        help="greedy 模式下改为朴素扫描（默认是堆优化）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problem = Problem(
        reliabilities=[0.90, 0.85, 0.80, 0.95],
        costs=[5, 4, 3, 8],
        target_reliability=0.95,
        upper_bounds=[7, 7, 7, 7],
    )

    if args.mode == "demo":
        demo()
    elif args.mode == "greedy":
        sol = greedy_allocate(problem, use_heap=not args.naive)
        print(sol)
    else:
        sol = brute_force_optimal(problem)
        print(sol)


if __name__ == "__main__":
    main()
