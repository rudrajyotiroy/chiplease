#!/usr/bin/env python3
"""Plot AccelSim A100 experiment results — all workloads, expanded 10M run."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config labels / colors ─────────────────────────────────────────────────────
CFG_LABELS = ["cfg1\n(baseline)", "cfg2\n(dram_instr=20)", "cfg3\n(both=10)", "cfg4\n(both=20)"]
CFG_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── Full dataset (sim_results_a100_expanded10m_20260415) ───────────────────────
# All 10M-capped workloads hit exactly 10M cycles for cfg1/cfg2.
# WL08 naturally completed (~18.35M cycles).
# WL03 all configs crashed/timed-out partway through kernel 274 (~18.3M cycles).
#   IPC values for WL03 represent partial execution (273/300 kernels + partial k274).
data = {
    "WL01":  {"ipc": [1875.7242, 1875.7242, 1902.6086, 1904.2521],
              "cycles": [10000000, 10000000, 10000000, 10000000]},
    "WL02\n(FlashInfer)":
             {"ipc": [2868.7725, 2868.7725, 2868.7725, 2868.7725],
              "cycles": [10000000, 10000000, 10000000, 10000000]},
    "WL03*\n(OpenFold)":
             {"ipc": [2734.2031, 2734.2031, 2725.5959, 2724.7029],
              "cycles": [18297341, 18297341, 18355113, 18361130]},
    "WL04\n(GPUDrive)":
             {"ipc": [398.2688, 398.2688, 398.2688, 398.2688],
              "cycles": [10000000, 10000000, 10000000, 10000000]},
    "WL05\n(OpenPCDet)":
             {"ipc": [2342.9656, 2342.9656, 2331.0781, 2333.3125],
              "cycles": [10000000, 10000000, 10000000, 10000000]},
    "WL07\n(LAMMPS)":
             {"ipc": [3943.4084, 3943.4084, 3935.5671, 3930.9758],
              "cycles": [10000000, 10000000, 10000000, 10000000]},
    "WL08":  {"ipc": [3754.9126, 3754.9126, 3752.2173, 3753.5720],
              "cycles": [18351190, 18351190, 18364371, 18357744]},
    "WL10\n(GASAL2)":
             {"ipc": [1283.8494, 1283.8494, 1283.7229, 1283.4075],
              "cycles": [10000000, 10000000, 10000000, 10000000]},
    "WL11\n(Clara)":
             {"ipc": [13.7262,   13.7262,   13.7262,   13.7262],
              "cycles": [10000000, 10000000, 10000000, 10000000]},
    "WL13\n(GPT-2)":
             {"ipc": [1539.3004, 1539.3004, 1527.8817, 1527.6094],
              "cycles": [10000000, 10000000, 10000000, 10000000]},
    "WL14\n(LLaMA-3)":
             {"ipc": [992.0215,  992.0215,  992.3803,  994.9440],
              "cycles": [10000000, 10000000, 10000000, 10000000]},
    "WL15\n(Qwen)":
             {"ipc": [1862.8718, 1862.8718, 1861.3297, 1862.4995],
              "cycles": [10000000, 10000000, 10000000, 10000000]},
}

workloads = list(data.keys())
n_wl  = len(workloads)
n_cfg = 4
x      = np.arange(n_wl)
bar_w  = 0.18
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w

# ── Normalized IPC (to cfg1 baseline) ─────────────────────────────────────────
norm_ipc = np.array([
    [data[wl]["ipc"][c] / data[wl]["ipc"][0] for c in range(n_cfg)]
    for wl in workloads
])

fig1, ax1 = plt.subplots(figsize=(16, 5))
fig1.suptitle(
    "AccelSim A100 — Normalized IPC (relative to cfg1 baseline)\n"
    "cfg2: dram_instr_verif_lat=20 | cfg3: both_lat=10 | cfg4: both_lat=20\n"
    "* WL03 partial: all configs crashed/timed-out mid kernel-274 (273/300 kernels)",
    fontsize=10, fontweight="bold")

for i, (color, label) in enumerate(zip(CFG_COLORS, CFG_LABELS)):
    bars = ax1.bar(x + offsets[i], norm_ipc[:, i], bar_w, color=color,
                   alpha=0.85, label=label.replace("\n", " "),
                   edgecolor="white", linewidth=0.5)
    for j, bar in enumerate(bars):
        v = norm_ipc[j, i]
        if i > 0 and abs(v - 1.0) > 0.003:
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.003,
                     f"{v:.3f}", ha="center", va="bottom",
                     fontsize=6, color="black")

ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax1.set_xticks(x)
ax1.set_xticklabels(workloads, fontsize=8)
ax1.set_ylabel("Normalized IPC (higher = better)", fontsize=9)
ax1.set_title("IPC normalized to cfg1 baseline", fontsize=10, fontweight="bold")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_ylim(0, max(norm_ipc.max() * 1.12, 1.15))
ax1.grid(axis="y", alpha=0.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
plt.tight_layout()
out1 = "/tmp/accelsim_norm_ipc.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")

# ── Absolute IPC (excluding WL11 which has anomalously low IPC=13.7) ──────────
wl_main = [wl for wl in workloads if "WL11" not in wl]
x2 = np.arange(len(wl_main))
abs_ipc = np.array([[data[wl]["ipc"][c] for c in range(n_cfg)] for wl in wl_main])

fig2, ax2 = plt.subplots(figsize=(16, 5))
fig2.suptitle("AccelSim A100 — Absolute IPC (WL11 excluded: IPC≈13.7, off-scale)",
              fontsize=10, fontweight="bold")
for i, (color, label) in enumerate(zip(CFG_COLORS, CFG_LABELS)):
    ax2.bar(x2 + offsets[i], abs_ipc[:, i], bar_w, color=color, alpha=0.85,
            label=label.replace("\n", " "), edgecolor="white", linewidth=0.5)
ax2.set_xticks(x2)
ax2.set_xticklabels(wl_main, fontsize=8)
ax2.set_ylabel("IPC", fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(axis="y", alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.tight_layout()
out2 = "/tmp/accelsim_abs_ipc.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")

# ── Cycle counts for variable-cycle workloads (WL03, WL08) ────────────────────
wl_var = ["WL03*\n(OpenFold)", "WL08"]
x3 = np.arange(len(wl_var))
var_cycles = np.array([[data[wl]["cycles"][c] / 1e6 for c in range(n_cfg)] for wl in wl_var])

fig3, ax3 = plt.subplots(figsize=(8, 4))
fig3.suptitle("Simulation Cycles — Variable-length workloads\n"
              "(WL03 partial/crashed; WL08 natural completion)",
              fontsize=10, fontweight="bold")
for i, (color, label) in enumerate(zip(CFG_COLORS, CFG_LABELS)):
    ax3.bar(x3 + offsets[i], var_cycles[:, i], bar_w, color=color, alpha=0.85,
            label=label.replace("\n", " "), edgecolor="white", linewidth=0.5)
ax3.set_xticks(x3)
ax3.set_xticklabels(wl_var, fontsize=9)
ax3.set_ylabel("Total Sim Cycles (millions)", fontsize=9)
ax3.legend(fontsize=8)
ax3.grid(axis="y", alpha=0.3)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
plt.tight_layout()
out3 = "/tmp/accelsim_var_cycles.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
