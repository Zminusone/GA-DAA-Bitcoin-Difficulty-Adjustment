"""
ga_daa.py
NSGA-II 多目标遗传算法 + GA-DAA 仿真器
复现自：基于基因算法的比特币区块链难度调整算法研究与改进（张俊林，2022102899）
"""

import numpy as np
from blockchain_simulator import (
    BitcoinDAA, INITIAL_DIFF, TARGET_BLOCK_TIME, BITCOIN_INTERVAL,
    sample_block_time
)


# ─────────────────────────────────────────────
# NSGA-II 实现（论文第3.3节）
# ─────────────────────────────────────────────
class NSGAII:
    """
    NSGA-II 多目标遗传算法（双变量、双目标）
    变量：x = [block_interval ∈ [60,600], retarget_interval ∈ [100,4032]]
    目标：f1 = std(block_times), f2 = std(difficulties)
    超参数（论文表3-1）：
        pop_size=100, max_gen=30, Pc=0.9, ηc=20, Pm=0.5, ηm=20
    """

    # 变量边界（论文公式3-1）
    LB = np.array([60.0,  100.0])
    UB = np.array([600.0, 4032.0])

    def __init__(self, evaluate_fn,
                 pop_size: int = 100,
                 max_gen:  int = 30,
                 pc: float = 0.9,  eta_c: float = 20.0,
                 pm: float = 0.5,  eta_m: float = 20.0,
                 seed: int = 0):
        self.evaluate = evaluate_fn
        self.pop_size = pop_size
        self.max_gen  = max_gen
        self.pc, self.eta_c = pc, eta_c
        self.pm, self.eta_m = pm, eta_m
        self.rng = np.random.default_rng(seed)

    # ── 初始化种群 ──────────────────────────────
    def _init_pop(self) -> np.ndarray:
        return self.rng.uniform(self.LB, self.UB,
                                size=(self.pop_size, 2))

    # ── 评估目标函数 ──────────────────────────────
    def _eval_pop(self, pop: np.ndarray) -> np.ndarray:
        return np.array([self.evaluate(x) for x in pop])

    # ── 快速非支配排序 O(MN²) ─────────────────────
    def _fast_non_dominated_sort(self, F: np.ndarray) -> list[list[int]]:
        N = len(F)
        n_dom = np.zeros(N, dtype=int)          # 被支配数
        S_dom = [[] for _ in range(N)]          # 支配集
        rank  = np.full(N, -1, dtype=int)
        fronts = [[]]

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S_dom[i].append(j)
                elif self._dominates(F[j], F[i]):
                    n_dom[i] += 1
            if n_dom[i] == 0:
                rank[i] = 0
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S_dom[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        rank[j] = k + 1
                        next_front.append(j)
            k += 1
            fronts.append(next_front)

        return [f for f in fronts if f]

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.all(a <= b) and np.any(a < b))

    # ── 拥挤距离 ─────────────────────────────────
    def _crowding_distance(self, F: np.ndarray, front: list[int]) -> np.ndarray:
        n = len(front)
        dist = np.zeros(n)
        Ff = F[front]
        for m in range(F.shape[1]):
            order = np.argsort(Ff[:, m])
            dist[order[0]] = dist[order[-1]] = np.inf
            f_range = Ff[order[-1], m] - Ff[order[0], m]
            if f_range == 0:
                continue
            for k in range(1, n - 1):
                dist[order[k]] += (Ff[order[k+1], m] - Ff[order[k-1], m]) / f_range
        return dist

    # ── SBX 模拟二进制交叉 ────────────────────────
    def _sbx_crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c1, c2 = p1.copy(), p2.copy()
        if self.rng.random() > self.pc:
            return c1, c2
        for i in range(2):
            if self.rng.random() <= 0.5:
                u = self.rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (self.eta_c + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1))
                c1[i] = 0.5 * ((p1[i] + p2[i]) - beta * abs(p2[i] - p1[i]))
                c2[i] = 0.5 * ((p1[i] + p2[i]) + beta * abs(p2[i] - p1[i]))
        c1 = np.clip(c1, self.LB, self.UB)
        c2 = np.clip(c2, self.LB, self.UB)
        return c1, c2

    # ── 多项式变异 ────────────────────────────────
    def _polynomial_mutation(self, x: np.ndarray) -> np.ndarray:
        child = x.copy()
        for i in range(2):
            if self.rng.random() <= self.pm:
                u = self.rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (self.eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))
                child[i] += delta * (self.UB[i] - self.LB[i])
        return np.clip(child, self.LB, self.UB)

    # ── 锦标赛选择 ────────────────────────────────
    def _tournament(self, ranks: np.ndarray, crowd: np.ndarray) -> int:
        a, b = self.rng.choice(len(ranks), 2, replace=False)
        if ranks[a] < ranks[b]:
            return a
        if ranks[b] < ranks[a]:
            return b
        return a if crowd[a] >= crowd[b] else b

    # ── 主循环 ────────────────────────────────────
    def run(self) -> tuple[np.ndarray, np.ndarray]:
        """
        运行 NSGA-II，返回 (pareto_X, pareto_F)
        pareto_X: shape (k,2)  最终Pareto前沿的参数
        pareto_F: shape (k,2)  对应目标值
        """
        pop = self._init_pop()
        F   = self._eval_pop(pop)

        for _ in range(self.max_gen):
            # 生成子代
            ranks_arr, crowd_arr = self._rank_and_crowd(pop, F)
            children = []
            while len(children) < self.pop_size:
                i1 = self._tournament(ranks_arr, crowd_arr)
                i2 = self._tournament(ranks_arr, crowd_arr)
                c1, c2 = self._sbx_crossover(pop[i1], pop[i2])
                children.append(self._polynomial_mutation(c1))
                children.append(self._polynomial_mutation(c2))
            children = np.array(children[:self.pop_size])
            F_child  = self._eval_pop(children)

            # 精英合并
            combined   = np.vstack([pop, children])
            F_combined = np.vstack([F, F_child])
            pop, F = self._select_next(combined, F_combined)

        # 提取Pareto第一层
        fronts = self._fast_non_dominated_sort(F)
        pf_idx = fronts[0]
        return pop[pf_idx], F[pf_idx]

    def _rank_and_crowd(self, pop, F):
        fronts    = self._fast_non_dominated_sort(F)
        rank_arr  = np.empty(len(pop), dtype=int)
        crowd_arr = np.zeros(len(pop))
        for r, front in enumerate(fronts):
            for idx in front:
                rank_arr[idx] = r
            cd = self._crowding_distance(F, front)
            for k, idx in enumerate(front):
                crowd_arr[idx] = cd[k]
        return rank_arr, crowd_arr

    def _select_next(self, combined, F_combined):
        fronts    = self._fast_non_dominated_sort(F_combined)
        selected  = []
        for front in fronts:
            if len(selected) + len(front) <= self.pop_size:
                selected.extend(front)
            else:
                cd   = self._crowding_distance(F_combined, front)
                order = np.argsort(-cd)
                need  = self.pop_size - len(selected)
                selected.extend([front[o] for o in order[:need]])
                break
        idx = np.array(selected)
        return combined[idx], F_combined[idx]


# ─────────────────────────────────────────────
# GA-DAA 仿真器（论文第3.4节 + 第4章）
# ─────────────────────────────────────────────
class GADAASim:
    """
    GA-DAA 区块链仿真器
    在每个调整周期末检测偏差，若 deviation > threshold 则触发 NSGA-II 优化。
    """

    def __init__(self,
                 hashrate_seq:      np.ndarray,
                 trigger_threshold: float = 0.25,
                 sub_sim_blocks:    int   = 300,
                 ga_pop_size:       int   = 100,
                 ga_max_gen:        int   = 30,
                 seed:              int   = 42):
        self.hashrate_seq      = hashrate_seq
        self.threshold         = trigger_threshold
        self.sub_sim_blocks    = sub_sim_blocks
        self.ga_pop_size       = ga_pop_size
        self.ga_max_gen        = ga_max_gen
        self.seed              = seed
        self.rng               = np.random.default_rng(seed)

        # 当前 DAA 参数（初始为比特币默认值）
        self.block_interval    = TARGET_BLOCK_TIME
        self.retarget_interval = BITCOIN_INTERVAL

        self.difficulty        = INITIAL_DIFF
        self._window_times:    list[float] = []
        self.ga_trigger_count: int = 0
        self.ga_trigger_blocks: list[int] = []

    def _make_evaluate_fn(self, local_hashrate: np.ndarray):
        """创建适应度评估函数（子仿真 300 块）"""
        rng_eval = np.random.default_rng(self.rng.integers(1 << 31))
        diff0    = self.difficulty

        def evaluate(x: np.ndarray):
            bi   = float(x[0])
            ri   = max(1, int(round(x[1])))
            diff = diff0
            win: list[float] = []
            bts: list[float] = []
            diffs: list[float] = []

            n = min(self.sub_sim_blocks, len(local_hashrate))
            for i in range(n):
                hr = local_hashrate[i % len(local_hashrate)]
                bt = rng_eval.exponential(diff / hr)
                win.append(bt)
                bts.append(bt)
                diffs.append(diff)
                if len(win) >= ri:
                    act    = sum(win)
                    tgt    = ri * bi
                    factor = max(0.25, min(4.0, act / tgt))
                    diff   = diff * factor
                    win.clear()

            f1 = float(np.std(bts))   if len(bts)   > 1 else 1e9
            f2 = float(np.std(diffs)) if len(diffs) > 1 else 1e9
            return np.array([f1, f2])

        return evaluate

    def run(self) -> dict:
        n = len(self.hashrate_seq)
        block_times  = np.empty(n)
        difficulties = np.empty(n)

        for i in range(n):
            hr = self.hashrate_seq[i]
            bt = self.rng.exponential(self.difficulty / hr)
            self._window_times.append(bt)
            block_times[i]  = bt
            difficulties[i] = self.difficulty

            # 到达调整周期末
            if (i + 1) % self.retarget_interval == 0:
                mean_bt = np.mean(self._window_times)
                deviation = abs(mean_bt - self.block_interval) / self.block_interval

                # ── GA 触发检测（论文公式3-5）
                if deviation > self.threshold:
                    local_hr = self.hashrate_seq[max(0, i - self.sub_sim_blocks + 1): i + 1]
                    eval_fn  = self._make_evaluate_fn(local_hr)
                    ga       = NSGAII(eval_fn,
                                      pop_size=self.ga_pop_size,
                                      max_gen =self.ga_max_gen,
                                      seed    =self.rng.integers(1 << 31))
                    pf_X, pf_F = ga.run()
                    # 从Pareto前沿选 f1 最小的解（论文3.4.2）
                    best_idx = np.argmin(pf_F[:, 0])
                    best_x   = pf_X[best_idx]
                    self.block_interval    = float(best_x[0])
                    self.retarget_interval = max(100, int(round(best_x[1])))
                    self.ga_trigger_count += 1
                    self.ga_trigger_blocks.append(i)

                # 常规难度调整
                actual = sum(self._window_times)
                target = len(self._window_times) * self.block_interval
                factor = max(0.25, min(4.0, actual / target))
                self.difficulty = self.difficulty * factor
                self._window_times.clear()

        return {
            "block_times":     block_times,
            "difficulties":    difficulties,
            "ga_trigger_count": self.ga_trigger_count,
            "ga_trigger_blocks": self.ga_trigger_blocks,
        }
