"""
run_experiments_final.py
GA-DAA 全套实验（复现 + 导师建议统计补充）
GA超参数：pop=40, gen=12, sub=150（精简版，保证可运行性）
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats
import warnings, os
warnings.filterwarnings("ignore")

from blockchain_simulator import (
    HashRateScenario, simulate_bitcoin_daa,
    TARGET_BLOCK_TIME, BITCOIN_INTERVAL
)
from ga_daa import GADAASim

OUT = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)

# ── 字体 & 风格
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.unicode_minus": False,
    "font.size":          10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

C_BTC  = "#F7931A"
C_GA   = "#1976D2"
C_TGT  = "#9E9E9E"
C_TRIG = "#E91E63"
C_GRID = "#EEEEEE"

GA_KW = dict(ga_pop_size=40, ga_max_gen=12, sub_sim_blocks=150, trigger_threshold=0.25)
SEEDS = [42, 123, 256, 789, 1024, 2048, 3141, 5926, 9999, 7777]
SIM_DAYS_5000 = 5000 * TARGET_BLOCK_TIME / 86400   # ≈34.7天


# ══════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════

def moving_avg(arr, w=100):
    if len(arr) <= w:
        return arr
    return np.convolve(arr, np.ones(w)/w, mode="valid")

def run_pair(hr_seq, seed):
    btc = simulate_bitcoin_daa(hr_seq, seed=seed)
    sim = GADAASim(hr_seq, seed=seed, **GA_KW)
    ga  = sim.run()
    return btc, ga, sim

def pct_dev(arr):
    return (np.mean(arr) - TARGET_BLOCK_TIME) / TARGET_BLOCK_TIME * 100

def impr_pct(btc_val, ga_val):
    if btc_val == 0: return 0.
    return (btc_val - ga_val) / abs(btc_val) * 100

def styled_table(ax, rows, col_labels, row_labels, title=""):
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=col_labels, rowLabels=row_labels,
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.65)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#0D47A1"); cell.set_text_props(color="white", fontweight="bold")
        elif c == -1:
            cell.set_facecolor("#E3F2FD"); cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#FAFAFA" if r % 2 == 0 else "white")
        cell.set_edgecolor("#CCCCCC")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

def timeseries_axes(ax_bt, ax_d, btc_r, ga_r, title, trig_blocks=None, n=None):
    bt_b = btc_r["block_times"][:n] if n else btc_r["block_times"]
    bt_g = ga_r["block_times"][:n]  if n else ga_r["block_times"]
    d_b  = btc_r["difficulties"][:n] if n else btc_r["difficulties"]
    d_g  = ga_r["difficulties"][:n]  if n else ga_r["difficulties"]
    xs   = np.arange(len(bt_b))
    w    = min(100, len(xs)//4)

    # 出块时间
    ax_bt.fill_between(xs, bt_b, alpha=0.08, color=C_BTC)
    ax_bt.fill_between(xs, bt_g, alpha=0.08, color=C_GA)
    if len(xs) > w:
        mx = xs[w-1:]
        ax_bt.plot(mx, moving_avg(bt_b,w), color=C_BTC, lw=1.8, label="Bitcoin DAA")
        ax_bt.plot(mx, moving_avg(bt_g,w), color=C_GA,  lw=1.8, label="GA-DAA")
    ax_bt.axhline(TARGET_BLOCK_TIME, color=C_TGT, ls="--", lw=1.2, label=f"目标 {TARGET_BLOCK_TIME}s")
    if trig_blocks:
        for tb in trig_blocks:
            if n is None or tb < n:
                ax_bt.axvline(tb, color=C_TRIG, lw=0.9, alpha=0.75)
    ax_bt.set_ylabel("出块时间 (s)"); ax_bt.set_title(title, fontweight="bold")
    ax_bt.legend(fontsize=8, loc="upper right")
    ax_bt.set_facecolor("#FAFAFA"); ax_bt.grid(color=C_GRID, lw=0.5)

    # 归一化难度
    d_b_n = d_b / d_b[0]; d_g_n = d_g / d_g[0]
    ax_d.plot(np.arange(len(d_b_n)), d_b_n, color=C_BTC, lw=1.4, label="Bitcoin DAA")
    ax_d.plot(np.arange(len(d_g_n)), d_g_n, color=C_GA,  lw=1.4, label="GA-DAA")
    ax_d.set_ylabel("归一化难度"); ax_d.set_xlabel("区块编号")
    ax_d.legend(fontsize=8); ax_d.set_facecolor("#FAFAFA"); ax_d.grid(color=C_GRID, lw=0.5)


# ══════════════════════════════════════════════
# 实验1：恒定哈希率
# ══════════════════════════════════════════════
def exp1():
    print("▶ 实验1：恒定哈希率 n=4000 …")
    hr = HashRateScenario.constant(4000)
    btc_r, ga_r, sim = run_pair(hr, 42)

    fig = plt.figure(figsize=(14,8), facecolor="white")
    gs  = gridspec.GridSpec(2,2,figure=fig,height_ratios=[1.6,1],hspace=0.42,wspace=0.38)
    ax_bt = fig.add_subplot(gs[0,0]); ax_d = fig.add_subplot(gs[1,0])
    ax_t  = fig.add_subplot(gs[:,1])
    timeseries_axes(ax_bt, ax_d, btc_r, ga_r, "实验1：恒定哈希率基准")

    rows = [
        [f"{np.mean(btc_r['block_times']):.1f}", f"{np.mean(ga_r['block_times']):.1f}", "—"],
        [f"{np.std(btc_r['block_times']):.1f}",  f"{np.std(ga_r['block_times']):.1f}", "—"],
        [f"{pct_dev(btc_r['block_times']):+.2f}%", f"{pct_dev(ga_r['block_times']):+.2f}%","—"],
        ["—", f"{ga_r['ga_trigger_count']}次", "—"],
    ]
    styled_table(ax_t, rows,
        col_labels=["Bitcoin DAA","GA-DAA","说明"],
        row_labels=["平均出块时间(s)","出块时间标准差(s)","相对目标偏差","GA触发次数"],
        title="表4-4  实验1性能对比（n=4000块）")
    fig.suptitle("实验1：恒定哈希率基准测试", fontsize=12, fontweight="bold")
    fig.savefig(f"{OUT}/exp1_constant.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   完成 Bitcoin={np.mean(btc_r['block_times']):.1f}s GA={np.mean(ga_r['block_times']):.1f}s 触发={ga_r['ga_trigger_count']}")


# ══════════════════════════════════════════════
# 实验2：突发激增
# ══════════════════════════════════════════════
def exp2():
    print("▶ 实验2：突发激增 n=4000 …")
    hr = HashRateScenario.surge(4000, 1500)
    btc_r, ga_r, sim = run_pair(hr, 42)

    fig = plt.figure(figsize=(14,8), facecolor="white")
    gs  = gridspec.GridSpec(2,2,figure=fig,height_ratios=[1.6,1],hspace=0.42,wspace=0.38)
    ax_bt = fig.add_subplot(gs[0,0]); ax_d = fig.add_subplot(gs[1,0])
    ax_t  = fig.add_subplot(gs[:,1])
    timeseries_axes(ax_bt, ax_d, btc_r, ga_r, "实验2：突发激增（第1500块后×7.5）",
                    trig_blocks=ga_r["ga_trigger_blocks"])
    ax_bt.axvline(1500, color="red", lw=1.8, ls=":", label="激增点")
    ax_bt.legend(fontsize=8)

    pre  = btc_r["block_times"][:1500]
    post_b = btc_r["block_times"][1500:]
    post_g = ga_r["block_times"][1500:]
    rows = [
        [f"{np.mean(pre):.1f}", f"{np.mean(post_b):.1f}", f"{np.mean(post_g):.1f}"],
        [f"{np.std(pre):.1f}",  f"{np.std(post_b):.1f}",  f"{np.std(post_g):.1f}"],
        ["—","—",f"{ga_r['ga_trigger_count']}次"],
        [f"{pct_dev(pre):+.2f}%",f"{pct_dev(post_b):+.2f}%",f"{pct_dev(post_g):+.2f}%"],
    ]
    styled_table(ax_t, rows,
        col_labels=["BTC(激增前)","BTC(激增后)","GA-DAA(激增后)"],
        row_labels=["平均出块时间(s)","出块时间标准差(s)","GA触发次数","相对目标偏差"],
        title="表4-5  实验2性能对比（n=4000块）")
    fig.suptitle("实验2：突发哈希率激增测试", fontsize=12, fontweight="bold")
    fig.savefig(f"{OUT}/exp2_surge.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   完成 激增后BTC={np.mean(post_b):.1f}s GA={np.mean(post_g):.1f}s 触发={ga_r['ga_trigger_count']}")


# ══════════════════════════════════════════════
# 实验3：阶梯式增长（导师建议重点）
# ══════════════════════════════════════════════
def exp3():
    print("▶ 实验3：阶梯式增长 n=5000 + 10次重复统计 …")

    # 单次（seed=42）详细
    hr = HashRateScenario.staircase(5000)
    btc_r, ga_r, sim = run_pair(hr, 42)

    # 10次重复
    print("   运行10次重复实验 …")
    btc_means=[]; btc_stds=[]; ga_means=[]; ga_stds=[]
    ga_trigs=[]; btc_devs=[]; ga_devs=[]
    for s in SEEDS:
        hr_i = HashRateScenario.staircase(5000)
        b, g, _ = run_pair(hr_i, s)
        btc_means.append(np.mean(b["block_times"]))
        btc_stds.append(np.std(b["block_times"]))
        ga_means.append(np.mean(g["block_times"]))
        ga_stds.append(np.std(g["block_times"]))
        ga_trigs.append(g["ga_trigger_count"])
        btc_devs.append(pct_dev(b["block_times"]))
        ga_devs.append(pct_dev(g["block_times"]))
        print(f"   seed={s}: BTC={np.mean(b['block_times']):.0f}s GA={np.mean(g['block_times']):.0f}s 触发={g['ga_trigger_count']}")

    t_stat, p_val = stats.ttest_rel(btc_means, ga_means)
    impr_mean = impr_pct(np.mean(btc_means), np.mean(ga_means))
    impr_dev  = np.mean(btc_devs) - np.mean(ga_devs)

    print(f"\n   ── 10次重复统计 ──")
    print(f"   GA触发: {np.mean(ga_trigs):.1f}±{np.std(ga_trigs):.1f} 次")
    print(f"   BTC偏差: {np.mean(btc_devs):+.2f}%±{np.std(btc_devs):.2f}%")
    print(f"   GA偏差:  {np.mean(ga_devs):+.2f}%±{np.std(ga_devs):.2f}%")
    print(f"   改善:    {impr_dev:+.2f}pp | 配对t检验 p={p_val:.4f}")

    # ── 绘图
    fig = plt.figure(figsize=(16,12), facecolor="white")
    gs  = gridspec.GridSpec(3,2,figure=fig,height_ratios=[1.5,1,1],hspace=0.52,wspace=0.40)
    ax_bt  = fig.add_subplot(gs[0,0]); ax_d = fig.add_subplot(gs[1,0])
    ax_box = fig.add_subplot(gs[2,0])
    ax_t   = fig.add_subplot(gs[0:2,1])
    ax_bar = fig.add_subplot(gs[2,1])

    timeseries_axes(ax_bt, ax_d, btc_r, ga_r, "实验3：阶梯式哈希率增长（seed=42）",
                    trig_blocks=ga_r["ga_trigger_blocks"])

    # 箱线图
    bp = ax_box.boxplot(
        [btc_r["block_times"], ga_r["block_times"]],
        labels=["Bitcoin DAA","GA-DAA"],
        patch_artist=True, notch=False,
        medianprops=dict(color="black",lw=2),
        flierprops=dict(marker=".", markersize=2, alpha=0.3)
    )
    bp["boxes"][0].set_facecolor(C_BTC); bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(C_GA);  bp["boxes"][1].set_alpha(0.7)
    ax_box.axhline(TARGET_BLOCK_TIME, color=C_TGT, ls="--", lw=1.2)
    ax_box.set_ylabel("出块时间 (s)"); ax_box.set_title("出块时间分布（seed=42）")
    ax_box.set_facecolor("#FAFAFA"); ax_box.grid(axis="y", color=C_GRID, lw=0.5)

    # 统计对比表
    rows = [
        [f"{np.mean(btc_means):.1f}±{np.std(btc_means):.1f}",
         f"{np.mean(ga_means):.1f}±{np.std(ga_means):.1f}",
         f"{impr_mean:+.1f}%"],
        [f"{np.mean(btc_stds):.1f}±{np.std(btc_stds):.1f}",
         f"{np.mean(ga_stds):.1f}±{np.std(ga_stds):.1f}", "—"],
        [f"{np.mean(btc_devs):+.2f}%±{np.std(btc_devs):.2f}%",
         f"{np.mean(ga_devs):+.2f}%±{np.std(ga_devs):.2f}%",
         f"{impr_dev:+.2f}pp"],
        ["—",
         f"{np.mean(ga_trigs):.1f}±{np.std(ga_trigs):.1f}次", "—"],
        [f"5000块≈{SIM_DAYS_5000:.0f}天", f"p={p_val:.4f}",
         "✓" if p_val<0.05 else "✗"],
    ]
    styled_table(ax_t, rows,
        col_labels=["Bitcoin DAA（均值±std）","GA-DAA（均值±std）","改善幅度"],
        row_labels=["平均出块时间(s)","出块时间标准差(s)",
                    "相对目标偏差","GA触发次数","实验规模/显著性"],
        title=f"表4-6  实验3阶梯增长统计对比（10次重复，每次5000块≈{SIM_DAYS_5000:.0f}天）")

    # GA触发次数分布柱状
    ax_bar.bar(range(10), ga_trigs, color=C_GA, alpha=0.85)
    ax_bar.axhline(np.mean(ga_trigs), color="red", ls="--", lw=1.5,
                   label=f"均值={np.mean(ga_trigs):.1f}±{np.std(ga_trigs):.1f}")
    ax_bar.set_xticks(range(10))
    ax_bar.set_xticklabels([f"S{i+1}" for i in range(10)], fontsize=8)
    ax_bar.set_ylabel("GA触发次数"); ax_bar.set_xlabel("实验编号（不同随机种子）")
    ax_bar.set_title(f"10次独立实验GA触发次数分布\n（总仿真={5000*10}块≈{SIM_DAYS_5000*10:.0f}天）")
    ax_bar.legend(fontsize=9); ax_bar.set_facecolor("#FAFAFA"); ax_bar.grid(axis="y",color=C_GRID,lw=0.5)

    fig.suptitle(
        f"实验3：阶梯式哈希率增长\n"
        f"（含10次重复统计；GA触发均值={np.mean(ga_trigs):.1f}次；"
        f"出块偏差改善{impr_dev:+.2f}pp；p={p_val:.4f}）",
        fontsize=11, fontweight="bold")
    fig.savefig(f"{OUT}/exp3_staircase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return np.mean(ga_trigs), np.std(ga_trigs), impr_dev, p_val


# ══════════════════════════════════════════════
# 实验4：Coin-Hopping
# ══════════════════════════════════════════════
def exp4():
    print("▶ 实验4：Coin-Hopping攻击 n=4032 …")
    hr = HashRateScenario.coin_hopping(4032)
    btc_r, ga_r, sim = run_pair(hr, 42)

    fig = plt.figure(figsize=(14,8), facecolor="white")
    gs  = gridspec.GridSpec(2,2,figure=fig,height_ratios=[1.6,1],hspace=0.42,wspace=0.38)
    ax_bt = fig.add_subplot(gs[0,0]); ax_d = fig.add_subplot(gs[1,0])
    ax_t  = fig.add_subplot(gs[:,1])
    timeseries_axes(ax_bt, ax_d, btc_r, ga_r, "实验4：Coin-Hopping攻击场景",
                    trig_blocks=ga_r["ga_trigger_blocks"])

    nd_btc = np.std(btc_r["difficulties"]) / btc_r["difficulties"][0]
    nd_ga  = np.std(ga_r["difficulties"])  / ga_r["difficulties"][0]
    rows = [
        [f"{np.mean(btc_r['block_times']):.1f}", f"{np.mean(ga_r['block_times']):.1f}",
         f"{impr_pct(np.mean(btc_r['block_times']),np.mean(ga_r['block_times'])):+.1f}%"],
        [f"{np.std(btc_r['block_times']):.1f}",  f"{np.std(ga_r['block_times']):.1f}", "—"],
        [f"{pct_dev(btc_r['block_times']):+.2f}%", f"{pct_dev(ga_r['block_times']):+.2f}%", "—"],
        [f"{nd_btc:.4f}", f"{nd_ga:.4f}", "—"],
        ["—", f"{ga_r['ga_trigger_count']}次", "—"],
    ]
    styled_table(ax_t, rows,
        col_labels=["Bitcoin DAA","GA-DAA","改善"],
        row_labels=["平均出块时间(s)","出块时间标准差(s)","相对目标偏差",
                    "归一化难度标准差","GA触发次数"],
        title="表4-7  实验4 Coin-Hopping对比（n=4032块）")
    fig.suptitle("实验4：Coin-Hopping攻击测试", fontsize=12, fontweight="bold")
    fig.savefig(f"{OUT}/exp4_coinhopping.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   完成 BTC偏差={pct_dev(btc_r['block_times']):+.2f}% GA偏差={pct_dev(ga_r['block_times']):+.2f}% 触发={ga_r['ga_trigger_count']}")


# ══════════════════════════════════════════════
# 实验5：五场景综合
# ══════════════════════════════════════════════
def exp5():
    print("▶ 实验5：五场景综合对比 n=3000 …")
    n = 3000
    scenarios = [
        ("恒定",        HashRateScenario.constant),
        ("突发激增",    lambda nb: HashRateScenario.surge(nb,1000)),
        ("阶梯增长",    HashRateScenario.staircase),
        ("Coin-Hopping",HashRateScenario.coin_hopping),
        ("正弦波动",    HashRateScenario.sinusoidal),
    ]

    all_res = []
    for name, fn in scenarios:
        hr = fn(n)
        btc_r, ga_r, _ = run_pair(hr, 42)
        all_res.append((name, btc_r, ga_r))

    fig, axes = plt.subplots(1,3,figsize=(18,6),facecolor="white")
    names    = [r[0] for r in all_res]
    btc_m    = [np.mean(r[1]["block_times"]) for r in all_res]
    ga_m     = [np.mean(r[2]["block_times"]) for r in all_res]
    btc_s    = [np.std(r[1]["block_times"])  for r in all_res]
    ga_s     = [np.std(r[2]["block_times"])  for r in all_res]
    trigs    = [r[2]["ga_trigger_count"]     for r in all_res]
    x = np.arange(len(names)); w = 0.35

    ax = axes[0]
    ax.bar(x-w/2, btc_m, w, color=C_BTC, alpha=0.85, label="Bitcoin DAA")
    ax.bar(x+w/2, ga_m,  w, color=C_GA,  alpha=0.85, label="GA-DAA")
    ax.axhline(TARGET_BLOCK_TIME, color=C_TGT, ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("平均出块时间 (s)"); ax.set_title("五场景平均出块时间")
    ax.legend(fontsize=9); ax.set_facecolor("#FAFAFA"); ax.grid(axis="y",color=C_GRID,lw=0.5)

    ax = axes[1]
    ax.bar(x-w/2, btc_s, w, color=C_BTC, alpha=0.85, label="Bitcoin DAA")
    ax.bar(x+w/2, ga_s,  w, color=C_GA,  alpha=0.85, label="GA-DAA")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("出块时间标准差 (s)"); ax.set_title("五场景出块时间标准差")
    ax.legend(fontsize=9); ax.set_facecolor("#FAFAFA"); ax.grid(axis="y",color=C_GRID,lw=0.5)

    rows = []
    for nm, btc_r, ga_r in all_res:
        ip = impr_pct(np.mean(btc_r["block_times"]), np.mean(ga_r["block_times"]))
        rows.append([f"{np.mean(btc_r['block_times']):.0f}",
                     f"{np.mean(ga_r['block_times']):.0f}",
                     f"{ip:+.1f}%",
                     f"{ga_r['ga_trigger_count']}"])
    styled_table(axes[2], rows,
        col_labels=["BTC均值(s)","GA均值(s)","改善%","GA触发"],
        row_labels=names,
        title="表4-8  五场景汇总（n=3000块/场景）")

    fig.suptitle("实验5：五场景综合性能对比（n=3000块/场景）",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUT}/exp5_comprehensive.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   完成")


# ══════════════════════════════════════════════
# 补充统计实验（导师建议核心）
# ══════════════════════════════════════════════
def supplementary():
    print("\n▶ 补充实验：全场景10种子统计分析（导师建议）…")
    N = 5000
    scenarios = {
        "恒定哈希率":    HashRateScenario.constant,
        "突发激增":      lambda nb: HashRateScenario.surge(nb,1500),
        "阶梯式增长":    HashRateScenario.staircase,
        "Coin-Hopping": HashRateScenario.coin_hopping,
        "正弦波动":      HashRateScenario.sinusoidal,
    }

    records = {}
    for sname, fn in scenarios.items():
        print(f"   → {sname} …")
        btc_devs=[]; ga_devs=[]; ga_trigs=[]; btc_mbt=[]; ga_mbt=[]
        for s in SEEDS:
            hr = fn(N)
            b, g, _ = run_pair(hr, s)
            btc_devs.append(pct_dev(b["block_times"]))
            ga_devs.append(pct_dev(g["block_times"]))
            ga_trigs.append(g["ga_trigger_count"])
            btc_mbt.append(np.mean(b["block_times"]))
            ga_mbt.append(np.mean(g["block_times"]))

        t_stat, p_val = stats.ttest_rel(btc_mbt, ga_mbt)
        records[sname] = dict(
            btc_dev_m=np.mean(btc_devs), btc_dev_s=np.std(btc_devs),
            ga_dev_m=np.mean(ga_devs),   ga_dev_s=np.std(ga_devs),
            impr_pp=np.mean(btc_devs)-np.mean(ga_devs),
            impr_pct=impr_pct(np.mean(btc_mbt),np.mean(ga_mbt)),
            trig_m=np.mean(ga_trigs), trig_s=np.std(ga_trigs),
            p_val=p_val,
        )
        print(f"     BTC偏差={np.mean(btc_devs):+.2f}% GA偏差={np.mean(ga_devs):+.2f}% "
              f"改善={np.mean(btc_devs)-np.mean(ga_devs):+.2f}pp p={p_val:.4f} "
              f"触发={np.mean(ga_trigs):.1f}±{np.std(ga_trigs):.1f}")

    # 绘图
    fig = plt.figure(figsize=(18,11), facecolor="white")
    gs  = gridspec.GridSpec(2,2,figure=fig,hspace=0.55,wspace=0.42)
    ax_dev  = fig.add_subplot(gs[0,0])
    ax_impr = fig.add_subplot(gs[0,1])
    ax_trig = fig.add_subplot(gs[1,0])
    ax_tbl  = fig.add_subplot(gs[1,1])

    snames = list(records.keys())
    x = np.arange(len(snames)); w = 0.35

    # ① 偏差对比（误差棒）
    btc_dm = [records[s]["btc_dev_m"] for s in snames]
    btc_ds = [records[s]["btc_dev_s"] for s in snames]
    ga_dm  = [records[s]["ga_dev_m"]  for s in snames]
    ga_ds  = [records[s]["ga_dev_s"]  for s in snames]
    ax_dev.bar(x-w/2, btc_dm, w, yerr=btc_ds, color=C_BTC, alpha=0.85,
               capsize=5, label="Bitcoin DAA")
    ax_dev.bar(x+w/2, ga_dm,  w, yerr=ga_ds,  color=C_GA,  alpha=0.85,
               capsize=5, label="GA-DAA")
    ax_dev.axhline(0, color="gray", ls="-", lw=0.8)
    ax_dev.set_xticks(x); ax_dev.set_xticklabels(snames, rotation=18, ha="right", fontsize=9)
    ax_dev.set_ylabel("相对目标偏差 (%)")
    ax_dev.set_title(f"出块时间相对目标偏差（10次重复，含95%误差棒）\n仿真规模：每次 {N}块 ≈ {N*TARGET_BLOCK_TIME/86400:.0f} 天")
    ax_dev.legend(fontsize=9); ax_dev.set_facecolor("#FAFAFA"); ax_dev.grid(axis="y",color=C_GRID,lw=0.5)

    # ② 改善率
    impr_vals = [records[s]["impr_pct"] for s in snames]
    colors = [C_GA if v >= 0 else "#EF5350" for v in impr_vals]
    bars = ax_impr.bar(x, impr_vals, color=colors, alpha=0.85)
    ax_impr.axhline(0, color="gray", lw=0.8)
    ax_impr.set_xticks(x); ax_impr.set_xticklabels(snames, rotation=18, ha="right", fontsize=9)
    ax_impr.set_ylabel("平均出块时间改善率 (%)")
    ax_impr.set_title("GA-DAA相对Bitcoin DAA平均出块时间改善率\n（正=GA更优，负=Bitcoin更优）")
    for bar, v in zip(bars, impr_vals):
        ax_impr.text(bar.get_x()+bar.get_width()/2, v + 0.4*np.sign(v) if v != 0 else 0.4,
                     f"{v:+.1f}%", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=8.5, fontweight="bold")
    ax_impr.set_facecolor("#FAFAFA"); ax_impr.grid(axis="y",color=C_GRID,lw=0.5)

    # ③ GA触发次数
    trig_ms = [records[s]["trig_m"] for s in snames]
    trig_ss = [records[s]["trig_s"] for s in snames]
    ax_trig.bar(x, trig_ms, yerr=trig_ss, color=C_GA, alpha=0.85, capsize=5)
    ax_trig.set_xticks(x); ax_trig.set_xticklabels(snames, rotation=18, ha="right", fontsize=9)
    ax_trig.set_ylabel("GA触发次数")
    ax_trig.set_title(f"各场景GA优化触发次数（10次重复均值±std）\n仿真总量：每场景 {N*10} 块 ≈ {N*10*TARGET_BLOCK_TIME/86400:.0f} 天")
    ax_trig.set_facecolor("#FAFAFA"); ax_trig.grid(axis="y",color=C_GRID,lw=0.5)

    # ④ 汇总统计表
    rows = []
    for s in snames:
        r = records[s]
        rows.append([
            f"{r['btc_dev_m']:+.2f}%\n±{r['btc_dev_s']:.2f}%",
            f"{r['ga_dev_m']:+.2f}%\n±{r['ga_dev_s']:.2f}%",
            f"{r['impr_pp']:+.2f}pp",
            f"{r['p_val']:.4f}",
            "✓" if r['p_val'] < 0.05 else "✗",
            f"{r['trig_m']:.1f}±{r['trig_s']:.1f}",
        ])
    styled_table(ax_tbl, rows,
        col_labels=["BTC偏差(均值\n±std)","GA偏差(均值\n±std)",
                    "改善\n(pp)","p值","显著\n(α=0.05)","GA触发\n(均值±std)"],
        row_labels=snames,
        title=f"导师建议补充统计表\n（10次重复×{N}块/次≈{SIM_DAYS_5000:.0f}天/次，百分比为统计意义下偏差）")

    fig.suptitle(
        f"导师建议补充实验：全场景多随机种子统计分析\n"
        f"（每场景10次独立重复，每次{N}块≈{SIM_DAYS_5000:.0f}天，"
        f"呈现均值±std与统计显著性p值）",
        fontsize=11, fontweight="bold"
    )
    fig.savefig(f"{OUT}/supplementary_statistical.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 打印控制台摘要
    print(f"\n   ╔═ 导师建议补充统计摘要 {'═'*46}╗")
    print(f"   {'场景':<14} {'BTC偏差':>14} {'GA偏差':>14} {'改善':>8} {'p值':>8} {'显著':>5} {'触发':>12}")
    print("   " + "─"*80)
    for s in snames:
        r = records[s]
        print(f"   {s:<14} {r['btc_dev_m']:+7.2f}%±{r['btc_dev_s']:.2f}% "
              f"{r['ga_dev_m']:+7.2f}%±{r['ga_dev_s']:.2f}% "
              f"{r['impr_pp']:+6.2f}pp "
              f"{r['p_val']:8.4f} "
              f"{'✓' if r['p_val']<0.05 else '✗':>5} "
              f"{r['trig_m']:5.1f}±{r['trig_s']:.1f}")
    print("   ╚" + "═"*79 + "╝")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("="*65)
    print("  GA-DAA 仿真实验（复现 + 导师建议补充统计实验）")
    print("  GA超参：pop=40, gen=12, sub=150")
    print("="*65)
    exp1()
    exp2()
    trig_m, trig_s, impr_dev, p_val = exp3()
    exp4()
    exp5()
    supplementary()
    print(f"\n{'='*65}")
    print("  ✓ 全部完成！输出文件：")
    print("    exp1_constant.png")
    print("    exp2_surge.png")
    print("    exp3_staircase.png")
    print("    exp4_coinhopping.png")
    print("    exp5_comprehensive.png")
    print("    supplementary_statistical.png")
    print(f"{'='*65}")
