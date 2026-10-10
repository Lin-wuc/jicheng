# -*- coding: utf-8 -*-
"""
进阶挑战 · 任务 1-② 验证戴维南定理
====================================
任务书要求（第五节）：
    ② 戴维南验证：自己画的含源二端网络图（标注端口）；
       V_oc、I_sc 两次仿真的「手算 vs 仿真」表；
       等效电路替换后接负载的电压/电流验证表。

运行方式：
    set PYTHONPATH=E:\\jicheng-main\\.pylib
    python circuit2_thevenin.py

自定参数（取整数便于手算）：
    含源二端网络 = 12 V 电压源 + R1 = 2 kΩ 串联 + R2 = 3 kΩ 并联（端口 a-b）
    负载 RL = 4 kΩ
"""

import os
import math

# ----------------------------------------------------------------------
# 运行环境：定位 ngspice 并与 PySpice 对接
# ----------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_ngspice():
    """返回 (spinit 所在目录, ngspice.dll 所在目录)。"""
    env_dll = os.environ.get("NGSPICE_DLL_DIR")
    if env_dll and os.path.isdir(env_dll):
        return (os.environ.get("NGSPICE_SPICE_LIB_DIR")
                or os.path.join(os.path.dirname(env_dll), "share", "ngspice"), env_dll)
    try:
        import PySpice
        base = os.path.join(os.path.dirname(os.path.abspath(PySpice.__file__)),
                            "Spice", "NgSpice", "Spice64_dll")
        if os.path.isdir(os.path.join(base, "dll-vs")):
            return os.path.join(base, "share", "ngspice"), os.path.join(base, "dll-vs")
    except Exception:
        pass
    for depth in range(1, 6):
        for name in ("_tools/spice_root/Spice64_dll", "_tools/ngspice/Spice64_dll"):
            cand = os.path.abspath(os.path.join(_HERE, *([".."] * depth), name))
            if os.path.isdir(os.path.join(cand, "dll-vs")):
                return os.path.join(cand, "share", "ngspice"), os.path.join(cand, "dll-vs")
    raise RuntimeError("找不到 ngspice 运行库，请设置环境变量 NGSPICE_DLL_DIR")


_spice_lib, _spice_dll = _find_ngspice()
os.environ["SPICE_LIB_DIR"] = os.path.dirname(_spice_dll)
os.add_dll_directory(_spice_dll)

from PySpice.Spice.Netlist import Circuit           # noqa: E402
from PySpice.Unit import u_Ω, u_kΩ, u_V, u_mA, u_A  # noqa: E402

import numpy as np                                  # noqa: E402
import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 输出工具
# ----------------------------------------------------------------------
_REPORT = []


def _to_ascii(text):
    return (text.replace("µ", "u").replace("Ω", "ohm").replace("τ", "tau")
                .replace("π", "pi").replace("→", "->").replace("×", "x")
                .replace("·", "*").replace("①", "(1)").replace("②", "(2)")
                .replace("—", "-").replace("《", "\"").replace("》", "\""))


def say(text=""):
    _REPORT.append(text)
    try:
        print(text)
    except UnicodeEncodeError:
        print(_to_ascii(text))


def save_report(filename):
    path = os.path.join(_HERE, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_REPORT) + "\n")
    print("报告已保存：%s" % path)
    return path


# ----------------------------------------------------------------------
# 电路参数与手算
# ----------------------------------------------------------------------
VS = 12.0        # 电源电压 12 V
R1 = 2e3         # 2 kΩ（与电源串联）
R2 = 3e3         # 3 kΩ（并联在端口上）
RL = 4e3         # 负载 4 kΩ

# 1) 开路电压 V_oc：端口开路时，R2 与 R1 串联分压
VOC_HAND = VS * R2 / (R1 + R2)
# 2) 短路电流 I_sc：端口短路时仅有 R1 限流
ISC_HAND = VS / R1
# 3) 等效电阻 R_th = V_oc / I_sc
RTH_HAND = VOC_HAND / ISC_HAND
# 4) 等效电路接负载后的电压与电流
VL_HAND = VOC_HAND * RL / (RTH_HAND + RL)
IL_HAND = VL_HAND / RL

# 原电路（含源二端网络）接同一负载的结果，作为独立验算
VL_ORIG_HAND = VS * (R2 * RL / (R2 + RL)) / (R1 + R2 * RL / (R2 + RL))
IL_ORIG_HAND = VL_ORIG_HAND / RL

say("=" * 84)
say("任务 1-② 验证戴维南定理")
say("=" * 84)
say()
say("含源二端网络：V1 = %g V，R1 = %g kΩ 串联，R2 = %g kΩ 并在端口 a-b 上；负载 RL = %g kΩ"
    % (VS, R1 / 1e3, R2 / 1e3, RL / 1e3))
say()
say("【手算过程】")
say("  ① 开路电压 V_oc（端口开路，R2 与 R1 分压）：")
say("       V_oc = V1 · R2/(R1+R2) = %g × %g/%g = %.6g V" % (VS, R2, R1 + R2, VOC_HAND))
say("  ② 短路电流 I_sc（端口短接，R2 被短路，仅 R1 限流）：")
say("       I_sc = V1/R1 = %g/%g = %.6g A = %.6g mA" % (VS, R1, ISC_HAND, ISC_HAND * 1e3))
say("  ③ 戴维南等效电阻：")
say("       R_th = V_oc/I_sc = %.6g/%.6g = %.6g Ω = %.4g kΩ" % (VOC_HAND, ISC_HAND, RTH_HAND, RTH_HAND / 1e3))
say("       （也可由 R_th = R1∥R2 = %g∥%g = %.6g Ω 独立验证）"
    % (R1, R2, R1 * R2 / (R1 + R2)))
say("  ④ 等效电路接负载 RL 后的端口电压与电流：")
say("       V_L = V_oc · RL/(R_th+RL) = %.6g × %g/%g = %.6f V" % (VOC_HAND, RL, RTH_HAND + RL, VL_HAND))
say("       I_L = V_L/RL = %.6f mA" % (IL_HAND * 1e3))
say("  ⑤ 原电路直接算（用于交叉验算）：")
say("       R2∥RL = %.6g Ω，V_L = V1·(R2∥RL)/(R1+R2∥RL) = %.6f V，I_L = %.6f mA"
    % (R2 * RL / (R2 + RL), VL_ORIG_HAND, IL_ORIG_HAND * 1e3))
say()

# ----------------------------------------------------------------------
# 仿真一：原电路 —— 开路电压 V_oc
# ----------------------------------------------------------------------
ckt_oc = Circuit("thevenin Voc")
ckt_oc.V("1", "n1", ckt_oc.gnd, VS @ u_V)
ckt_oc.R("1", "n1", "a", R1 @ u_Ω)
ckt_oc.R("2", "a", ckt_oc.gnd, R2 @ u_Ω)
op_oc = ckt_oc.simulator().operating_point()
voc_sim = float(op_oc["a"][0])

# ----------------------------------------------------------------------
# 仿真二：原电路 —— 短路电流 I_sc（端口用 0 V 电压源短接，读该源电流）
# ----------------------------------------------------------------------
ckt_sc = Circuit("thevenin Isc")
ckt_sc.V("1", "n1", ckt_sc.gnd, VS @ u_V)
ckt_sc.R("1", "n1", "a", R1 @ u_Ω)
ckt_sc.R("2", "a", ckt_sc.gnd, R2 @ u_Ω)
ckt_sc.V("short", "a", "b", 0 @ u_V)        # b 接地，a-b 之间被 0 V 源短接
ckt_sc.R("load", "b", ckt_sc.gnd, 1e-9 @ u_Ω)   # 给 b 一个直流通路
op_sc = ckt_sc.simulator().operating_point()
isc_sim = abs(float(op_sc["vshort"][0]))
rth_sim = voc_sim / isc_sim

# ----------------------------------------------------------------------
# 仿真三：等效电路（V_oc 串联 R_th）接负载 —— 端口电压与电流
# ----------------------------------------------------------------------
ckt_eq = Circuit("thevenin equivalent + load")
ckt_eq.V("oc", "a", ckt_eq.gnd, VOC_HAND @ u_V)
ckt_eq.R("th", "a", "b", RTH_HAND @ u_Ω)
ckt_eq.V("sense", "b", "bn", 0 @ u_V)       # 串联 0 V 源测负载电流
ckt_eq.R("L", "bn", ckt_eq.gnd, RL @ u_Ω)
op_eq = ckt_eq.simulator().operating_point()
vl_eq_sim = float(op_eq["b"][0])
il_eq_sim = abs(float(op_eq["vsense"][0]))

# ----------------------------------------------------------------------
# 仿真四：原电路接同一负载 —— 端口电压与电流（验证等效前后一致）
# ----------------------------------------------------------------------
ckt_or = Circuit("original network + load")
ckt_or.V("1", "n1", ckt_or.gnd, VS @ u_V)
ckt_or.R("1", "n1", "a", R1 @ u_Ω)
ckt_or.R("2", "a", ckt_or.gnd, R2 @ u_Ω)
ckt_or.V("sense", "a", "an", 0 @ u_V)
ckt_or.R("L", "an", ckt_or.gnd, RL @ u_Ω)
op_or = ckt_or.simulator().operating_point()
vl_or_sim = float(op_or["an"][0])
il_or_sim = abs(float(op_or["vsense"][0]))

say("【仿真结果】")
say("  V_oc（端口开路，原电路）  = %.6f V" % voc_sim)
say("  I_sc（端口短接，原电路）  = %.6f mA" % (isc_sim * 1e3))
say("  R_th = V_oc/I_sc         = %.6f Ω = %.4f kΩ" % (rth_sim, rth_sim / 1e3))
say("  等效电路接 RL：V_L = %.6f V，I_L = %.6f mA" % (vl_eq_sim, il_eq_sim * 1e3))
say("  原电路 接 RL：V_L = %.6f V，I_L = %.6f mA" % (vl_or_sim, il_or_sim * 1e3))
say()

# ----------------------------------------------------------------------
# 仿真五：端口伏安特性（戴维南等效电路 + 电流源负载，直流扫描）
#         把 RL 折算成负载电流 i = V_L/RL，扫描该电流即得 V-I 特性线
# ----------------------------------------------------------------------
I_SWEEP = np.linspace(0, ISC_HAND, 121)

ckt_sw = Circuit("thevenin equivalent, V-I sweep")
ckt_sw.V("oc", "a", ckt_sw.gnd, VOC_HAND @ u_V)
ckt_sw.R("th", "a", "b", RTH_HAND @ u_Ω)
ckt_sw.I("Load", ckt_sw.gnd, "b", 0 @ u_A)      # 由 b 向地灌电流，电流值由扫描给定
# PySpice 的 dc() 用法：源名 = slice(start, stop, step)
STEP_A = ISC_HAND / 120
dc_sw = ckt_sw.simulator().dc(iLoad=slice(0, ISC_HAND, STEP_A))
i_sw = np.array([abs(float(x)) for x in dc_sw.sweep])
v_sw = np.array([float(x) for x in dc_sw["b"]])
v_sw_theory = VOC_HAND - RTH_HAND * i_sw

# 由 V-I 特性换算成「端口电压随负载电阻变化」：R = V/i
r_from_vi = np.divide(v_sw, i_sw, out=np.full_like(v_sw, np.inf), where=i_sw > 1e-12)

# 图 1-②-b 用的理论曲线：V_L 随 RL 变化
RL_SWEEP = np.logspace(math.log10(10), math.log10(1e6), 200)
vl_theory_curve = VOC_HAND * RL_SWEEP / (RTH_HAND + RL_SWEEP)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
ax = axes[0]
ax.plot(i_sw * 1e3, v_sw_theory, "-", color="#1f77b4", linewidth=2,
        label="理论 $V=V_{oc}-R_{th}I$")
ax.plot(i_sw * 1e3, v_sw, "--", color="#d62728", linewidth=1.6, label="仿真（等效电路）")
ax.plot([isc_sim * 1e3], [0], "o", color="#2ca02c", markersize=7,
        label="短路点 $I_{sc}$ = %.3f mA" % (isc_sim * 1e3))
ax.plot([0], [VOC_HAND], "s", color="#ff7f0e", markersize=6,
        label="开路点 $V_{oc}$ = %.3f V" % VOC_HAND)
ax.plot([il_or_sim * 1e3], [vl_or_sim], "D", color="#9467bd", markersize=6,
        label="工作点 $R_L$=%.0f kΩ" % (RL / 1e3))
ax.set_xlabel("端口电流 $I$ / mA")
ax.set_ylabel("端口电压 $V$ / V")
ax.set_title("图 1-②-a  含源二端网络的伏安特性（一条直线）")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper right")

ax = axes[1]
mask = np.isfinite(r_from_vi) & (r_from_vi > 0)
ax.semilogx(r_from_vi[mask] / 1e3, v_sw[mask], "-", color="#d62728", linewidth=2,
            label="仿真（由 V-I 特性换算）")
ax.semilogx(RL_SWEEP / 1e3, vl_theory_curve, "--", color="#1f77b4", linewidth=1.6,
            label="理论 $V_L=V_{oc}R_L/(R_{th}+R_L)$")
ax.axhline(VOC_HAND, color="#2ca02c", linestyle=":", linewidth=1,
           label="$V_{oc}$ = %.3f V（RL→∞）" % VOC_HAND)
ax.plot([RL / 1e3], [vl_or_sim], "o", color="#ff7f0e", markersize=7,
        label="工作点 $R_L$=%.0f kΩ → %.4f V" % (RL / 1e3, vl_or_sim))
ax.set_xlabel("负载 $R_L$ / kΩ")
ax.set_ylabel("端口电压 $V_L$ / V")
ax.set_title("图 1-②-b  端口电压随负载电阻的变化")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8, loc="lower right")

fig.tight_layout()
png = os.path.join(_HERE, "fig2_thevenin.png")
fig.savefig(png, dpi=150)
say("图片已保存：%s" % png)
say()

# ----------------------------------------------------------------------
# 对比表
# ----------------------------------------------------------------------
def err_of(theo, sim):
    return "%.4f%%" % (abs(sim - theo) / abs(theo) * 100 if theo else 0)


say("=" * 92)
say("表 1-②-a  V_oc 与 I_sc 两次仿真的「手算 vs 仿真」")
say("=" * 92)
say("%-30s %-20s %-20s %-12s" % ("项目", "理论值（手算）", "仿真值", "相对误差"))
say("-" * 92)
say("%-30s %-20s %-20s %-12s" % ("开路电压 V_oc", "%.6f V" % VOC_HAND, "%.6f V" % voc_sim,
                                 err_of(VOC_HAND, voc_sim)))
say("%-30s %-20s %-20s %-12s" % ("短路电流 I_sc", "%.6f mA" % (ISC_HAND * 1e3),
                                 "%.6f mA" % (isc_sim * 1e3), err_of(ISC_HAND, isc_sim)))
say("%-30s %-20s %-20s %-12s" % ("等效电阻 R_th = V_oc/I_sc",
                                 "%.4f kΩ" % (RTH_HAND / 1e3), "%.4f kΩ" % (rth_sim / 1e3),
                                 err_of(RTH_HAND, rth_sim)))
say("%-30s %-20s %-20s %-12s" % ("等效电阻 R_th = R1∥R2",
                                 "%.4f kΩ" % (R1 * R2 / (R1 + R2) / 1e3),
                                 "%.4f kΩ" % (rth_sim / 1e3), err_of(R1 * R2 / (R1 + R2), rth_sim)))
say("=" * 92)
say()
say("=" * 92)
say("表 1-②-b  等效电路替换后接负载的电压/电流验证（RL = %.4g kΩ）" % (RL / 1e3))
say("=" * 92)
say("%-34s %-16s %-16s %-16s" % ("项目", "理论值（手算）", "原电路仿真", "等效电路仿真"))
say("-" * 92)
say("%-34s %-16s %-16s %-16s" % ("端口电压 V_L", "%.6f V" % VL_HAND,
                                 "%.6f V" % vl_or_sim, "%.6f V" % vl_eq_sim))
say("%-34s %-16s %-16s %-16s" % ("负载电流 I_L", "%.6f mA" % (IL_HAND * 1e3),
                                 "%.6f mA" % (il_or_sim * 1e3), "%.6f mA" % (il_eq_sim * 1e3)))
say("%-34s %-16s %-16s %-16s" % ("负载消耗功率 P_L", "%.6f mW" % (VL_HAND * IL_HAND * 1e3),
                                 "%.6f mW" % (vl_or_sim * il_or_sim * 1e3),
                                 "%.6f mW" % (vl_eq_sim * il_eq_sim * 1e3)))
say("-" * 92)
say("原电路 vs 等效电路  电压相对偏差：%s；电流相对偏差：%s"
    % (err_of(vl_or_sim, vl_eq_sim), err_of(il_or_sim, il_eq_sim)))
say("原电路仿真 vs 手算  电压相对偏差：%s；电流相对偏差：%s"
    % (err_of(VL_ORIG_HAND, vl_or_sim), err_of(IL_ORIG_HAND, il_or_sim)))
say("=" * 92)
say()
say("【结论】")
say("  · 由端口开路、短路两次仿真得到 V_oc = %.4f V、I_sc = %.4f mA，相除得 R_th = %.4f kΩ，"
    % (voc_sim, isc_sim * 1e3, rth_sim / 1e3))
say("    与 R_th = R1∥R2 = %.4f kΩ 一致；" % (R1 * R2 / (R1 + R2) / 1e3))
say("  · 用「V_oc 串联 R_th」的等效电路替换原含源二端网络后，接同一负载 RL 得到的端口")
say("    电压与电流和原电路完全一致（偏差 0），戴维南定理得到验证；")
say("  · 图 1-②-a 中仿真点全部落在 V = V_oc − R_th·I 这条直线上，说明该二端网络对外")
say("    只等效为一个电压源 V_oc 与一个电阻 R_th 串联；图 1-②-b 中 V_L 随 R_L 单调上升，")
say("    并在 R_L → ∞ 时趋于 V_oc、在 R_L = R_th 时降到 V_oc/2（最大功率匹配点）。")

save_report("report2_thevenin.txt")
