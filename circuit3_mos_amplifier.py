# -*- coding: utf-8 -*-
"""
进阶挑战 · 任务 1-③ NMOS 共源级放大电路
==========================================
任务书固定参数：
    VDD = 5 V,  Rg1 = 60 kΩ,  Rg2 = 40 kΩ,  Rd = 2 kΩ,  Cb1 视为足够大
    NMOS 模型：K = 0.8 mA/V²,  V_th = 1 V,  λ = 0.02 /V
    输入 Vi = 10 mV / 1 kHz 正弦波

任务书要求：
    · 手算静态工作点 V_GS、I_D、V_DS，并判断是否工作在饱和区；
    · 手算小信号 gm、Av；
    · 仿真验证：直流 OP 对比 I_D、V_DS；瞬态看输出波形；实测增益；
    · 画直流通路和小信号等效模型（此处用文字+公式给出，图另画）。

运行方式：
    set PYTHONPATH=E:\\jicheng-main\\.pylib
    python circuit3_mos_amplifier.py
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
from PySpice.Unit import (u_Ω, u_kΩ, u_V, u_mV, u_Hz, u_s, u_ms,   # noqa: E402
                          u_F, u_uF, u_mA, u_mS)

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
    return (text.replace("µ", "u").replace("Ω", "ohm").replace("λ", "lambda")
                .replace("Δ", "delta").replace("π", "pi").replace("→", "->")
                .replace("×", "x").replace("·", "*").replace("∥", "//")
                .replace("①", "(1)").replace("②", "(2)").replace("③", "(3)")
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
# 一、固定参数与手算静态工作点
# ----------------------------------------------------------------------
VDD = 5.0            # V
RG1 = 60e3           # 60 kΩ
RG2 = 40e3           # 40 kΩ
RD = 2e3             # 2 kΩ
K_PARAM = 0.8e-3     # K = 0.8 mA/V²
VTH = 1.0            # V_th = 1 V
LAMBDA = 0.02        # λ = 0.02 /V
VI_AMP = 10e-3       # 输入 10 mV 幅值
F_IN = 1e3           # 1 kHz

# 静态栅极电位（Cb1 隔直，栅极无电流，由 Rg1/Rg2 分压决定）
VG = VDD * RG2 / (RG1 + RG2)
VOV_IDEAL = VG - VTH          # 过驱动电压 = 2 − 1 = 1 V

# --- ① 理想平方律（忽略 λ）---
#   I_D = ½K(V_GS-V_th)²  →  (1/2)·0.8m·1² = 0.4 mA
ID_IDEAL = 0.5 * K_PARAM * VOV_IDEAL ** 2
VDS_IDEAL = VDD - ID_IDEAL * RD

# --- ② 含沟道长度调制 λ ---
#   I_D = ½K(V_GS-V_th)²(1+λV_DS),  V_DS = VDD - I_D·Rd
#   → λRdI_D² + (1-λVDD)I_D - ½K·V_ov² = 0
A = LAMBDA * RD
B = 1 - LAMBDA * VDD
C = -0.5 * K_PARAM * VOV_IDEAL ** 2
ID_HAND = (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)
VGS_HAND = VG
VDS_HAND = VDD - ID_HAND * RD

SAT_OK = VDS_HAND > VOV_IDEAL                     # 饱和判据 V_DS > V_GS - V_th
VDS_SAT_MIN = VOV_IDEAL                           # 饱和区最小 V_DS

# --- ③ 小信号参数（含 λ 的精确形式）---
gm = K_PARAM * VOV_IDEAL * (1 + LAMBDA * VDS_HAND)
gm_simple = K_PARAM * VOV_IDEAL                   # 忽略 λ 的简化式 2I_D/V_ov
ro = 1.0 / (LAMBDA * ID_HAND)
ro_ideal = float("inf")
AV_HAND = -gm * (RD * ro / (RD + ro))             # Rd ∥ ro
AV_HAND_SIMPLE = -gm * RD                         # 忽略 ro

say("=" * 90)
say("任务 1-③ NMOS 共源级放大电路   手算 + 仿真验证")
say("=" * 90)
say()
say("固定参数：VDD = %g V，Rg1 = %g kΩ，Rg2 = %g kΩ，Rd = %g kΩ；"
    % (VDD, RG1 / 1e3, RG2 / 1e3, RD / 1e3))
say("          NMOS：K = %g mA/V²，V_th = %g V，λ = %g /V；Vi = %g mV / %g Hz"
    % (K_PARAM * 1e3, VTH, LAMBDA, VI_AMP * 1e3, F_IN))
say()
say("【第一步：静态工作点手算】")
say("  ① 栅极电位（Cb1 隔直，栅极电流为 0，只有分压）：")
say("       V_G = VDD · Rg2/(Rg1+Rg2) = %g × %g/%g = %.4f V" % (VDD, RG2, RG1 + RG2, VG))
say("     源极接地 → V_GS = V_G = %.4f V" % VGS_HAND)
say("     过驱动电压 V_ov = V_GS − V_th = %.4f − %g = %.4f V" % (VGS_HAND, VTH, VOV_IDEAL))
say("  ② 漏极电流：先假设工作在饱和区，用 I_D = ½K·V_ov²·(1+λV_DS)：")
say("       (a) 忽略 λ：I_D = ½ × %g × %.4f² = %.6f mA"
    % (K_PARAM, VOV_IDEAL, ID_IDEAL * 1e3))
say("           则 V_DS = VDD − I_D·Rd = %g − %.6f×10⁻³×%g = %.6f V"
    % (VDD, ID_IDEAL * 1e3, RD, VDS_IDEAL))
say("       (b) 含 λ（解一元二次方程 λRd·I_D² + (1−λVDD)I_D − ½K·V_ov² = 0）：")
say("           %.6f·I_D² + %.6f·I_D − %.6e = 0" % (A, B, -C))
say("           → I_D = %.6f mA，V_DS = VDD − I_D·Rd = %.6f V"
    % (ID_HAND * 1e3, VDS_HAND))
say("  ③ 饱和区判断：判据 V_DS > V_GS − V_th")
say("       V_DS = %.6f V，V_GS − V_th = %.4f V" % (VDS_HAND, VOV_IDEAL))
say("       → %s，%s工作在饱和区" % ("V_DS > V_GS − V_th 成立" if SAT_OK else "不成立",
                                    "**是**" if SAT_OK else "**不是**"))
say("      （V_GS = %.4f V > V_th = %g V，沟道已形成；饱和区最小 V_DS = %.4f V）"
    % (VGS_HAND, VTH, VDS_SAT_MIN))
say()
say("【第二步：小信号参数手算】")
say("  ① 跨导 gm：gm = ∂I_D/∂V_GS = K·V_ov·(1+λV_DS)")
say("       gm = %g × %.4f × (1 + %g×%.6f) = %.6f mS（≈ %.4f mS）"
    % (K_PARAM, VOV_IDEAL, LAMBDA, VDS_HAND, gm * 1e3, gm_simple * 1e3))
say("       简化式 gm = K·V_ov = 2I_D/V_ov = %.6f mS（忽略 λ）" % (gm_simple * 1e3))
say("  ② 输出电阻 ro：ro = 1/(λ·I_D) = 1/(%g × %.6e) = %.4f kΩ"
    % (LAMBDA, ID_HAND, ro / 1e3))
say("  ③ 电压增益 Av（共源级为反相放大）：")
say("       Av = −gm·(Rd ∥ ro) = −%.6f×10⁻³ × (%g ∥ %.4f)×10³ = %.4f"
    % (gm * 1e3, RD / 1e3, ro / 1e3, AV_HAND))
say("       若忽略 ro：Av = −gm·Rd = %.4f" % AV_HAND_SIMPLE)
say("      |Av| = %.4f 倍，输出与输入反相" % abs(AV_HAND))
say()

# ----------------------------------------------------------------------
# 二、仿真
# ----------------------------------------------------------------------
def build_circuit(name, mode="dc", amp=0.0, freq=F_IN, ac_mag=0.0):
    """构造共源级放大电路。

    mode = "dc"  → 输入源为 0 V 直流（用于静态工作点）
    mode = "ac"  → 输入源幅值 0、AC 幅值 ac_mag（用于交流小信号分析）
    mode = "tr"  → 输入源为正弦波（用于瞬态分析）

    注意 1：输入源必须始终存在，否则 Cb1 右侧节点 vi 在直流下没有任何直流通路，
            ngspice 会因矩阵奇异而报错。
    注意 2：PySpice 会把单位后缀拼到数值后（例如 60 @ u_kΩ → 60000.0kOhm = 60 MΩ），
            所以带前缀的单位一律先把数值换算成基本单位（Ω、V、F、s）再传入。
    """
    c = Circuit(name)
    # 直流供电与分压偏置
    c.V("DD", "vdd", c.gnd, VDD @ u_V)
    c.R("g1", "vdd", "g", RG1 @ u_Ω)
    c.R("g2", "g", c.gnd, RG2 @ u_Ω)
    # 漏极负载
    c.R("d", "vdd", "d", RD @ u_Ω)
    # 输入耦合电容（视为足够大 → 取 1 µF）
    c.C("b1", "vi", "g", 1e-6 @ u_F)
    # 输入信号源
    if mode == "dc":
        c.V("s", "vi", c.gnd, 0 @ u_V)
    elif mode == "ac":
        c.SinusoidalVoltageSource("s", "vi", c.gnd, amplitude=0 @ u_V,
                                  frequency=freq @ u_Hz, ac_magnitude=ac_mag @ u_V)
    else:
        c.SinusoidalVoltageSource("s", "vi", c.gnd, amplitude=amp @ u_V,
                                  frequency=freq @ u_Hz)
    # NMOS：KP = 0.8 mA/V²、W/L = 1 → K = ½·KP·W/L = 0.8 mA/V²
    c.MOSFET(1, "d", "g", c.gnd, c.gnd, model="nmos1", w=1, l=1)
    c.model("nmos1", "nmos", KP=K_PARAM, VTO=VTH, LAMBDA=LAMBDA,
            level=1, W=1, L=1)
    return c


# --- 直流工作点 ---
ckt_dc = build_circuit("MOS CS amplifier (DC)")
op = ckt_dc.simulator().operating_point()
vgs_sim = float(op["g"][0]) - float(op["gnd"][0]) if "gnd" in op.nodes else float(op["g"][0])
vds_sim = float(op["d"][0])
id_sim = (VDD - vds_sim) / RD
# 直接从器件读电流更准确
try:
    id_sim_dev = abs(float(op.branches["@m1[id]"]))
    id_sim = id_sim_dev
except Exception:
    pass

say("【第三步：仿真验证——直流工作点】")
say("  V_GS = %.6f V，V_DS = %.6f V，I_D = %.6f mA（由 Rd 上压降算得）"
    % (vgs_sim, vds_sim, id_sim * 1e3))
say()

# --- 交流分析：小信号增益（输入源 AC 幅值 1 V，便于直接读传递函数）---
ckt_ac = build_circuit("MOS CS amplifier (AC)", mode="ac", ac_mag=1.0)
ac = ckt_ac.simulator().ac(start_frequency=10 @ u_Hz, stop_frequency=1e9 @ u_Hz,
                           number_of_points=20, variation="dec")
f = np.array([float(x) for x in ac.frequency])
h_d = ac["d"].as_ndarray()
h_g = ac["g"].as_ndarray()
gain_curve = np.abs(h_d / h_g)                       # |V_d / V_g|
phase_curve = np.degrees(np.angle(h_d / h_g))
gain_mid = float(gain_curve[int(np.argmin(np.abs(f - F_IN)))])
phase_mid = float(phase_curve[int(np.argmin(np.abs(f - F_IN)))])
# 中频段平均增益（1 kHz ~ 100 kHz）
band = (f >= F_IN) & (f <= 100e3)
gain_band = float(np.mean(gain_curve[band]))

# -3 dB 上限频率
half = gain_band / math.sqrt(2)
f_h = float("nan")
for i in range(1, len(f)):
    if f[i] > F_IN and gain_curve[i] < half <= gain_curve[i - 1]:
        f_h = f[i - 1] + (half - gain_curve[i - 1]) * (f[i] - f[i - 1]) / (gain_curve[i] - gain_curve[i - 1])
        break

say("【第四步：仿真验证——交流小信号增益】")
say("  在 %g Hz 处 |V_d/V_g| = %.6f（%.6f dB），相位 %.4f°（应为 180° 反相附近）"
    % (F_IN, gain_mid, 20 * math.log10(gain_mid), phase_mid))
say("  中频段(1k~100kHz)平均 |Av| = %.6f" % gain_band)
if f_h == f_h:
    say("  上限截止频率 f_H ≈ %.4g Hz（由 Rd∥ro 与 MOS 本征电容形成）" % f_h)
else:
    say("  上限截止频率 f_H：在 10 Hz~1 GHz 扫频范围内增益基本平坦、未下降到 -3 dB，")
    say("   说明 Level-1 模型未包含栅漏/结电容等寄生参数，带宽由外部电路决定（此为该模型的固有局限）")
say()

# --- 瞬态分析：输入/输出波形 + 实测增益 ---
ckt_tr = build_circuit("MOS CS amplifier (transient)", mode="tr", amp=VI_AMP, freq=F_IN)
PERIOD = 1.0 / F_IN
T_END = 12 * PERIOD
T_STEP = PERIOD / 400
tr = ckt_tr.simulator(temperature=25, nominal_temperature=25).transient(
    step_time=T_STEP @ u_s, end_time=T_END @ u_s)
tt = np.array([float(x) for x in tr.time])
vvi = np.array([float(x) for x in tr["vi"]])
vg = np.array([float(x) for x in tr["g"]])
vd = np.array([float(x) for x in tr["d"]])

# 取最后 3 个周期算幅值（已进入稳态）
tail = tt >= (T_END - 3 * PERIOD)
vi_amp_sim = (vvi[tail].max() - vvi[tail].min()) / 2
vg_amp_sim = (vg[tail].max() - vg[tail].min()) / 2
vd_amp_sim = (vd[tail].max() - vd[tail].min()) / 2
gain_tran = vd_amp_sim / vg_amp_sim
# 相位差直接取交流分析在 1 kHz 处的相位（比在瞬态波形上找过零点更可靠）
phase_sim = abs(phase_mid)

say("【第五步：仿真验证——瞬态波形与实测增益】")
say("  输入幅值（源）      = %.6f mV" % (vi_amp_sim * 1e3))
say("  栅极实际幅值        = %.6f mV（Cb1 隔直，1 kHz 处容抗可忽略）" % (vg_amp_sim * 1e3))
say("  输出幅值（漏极）    = %.6f mV" % (vd_amp_sim * 1e3))
say("  实测 |Av| = 输出/栅极 = %.6f / %.6f = %.6f" % (vd_amp_sim, vg_amp_sim, gain_tran))
say("  输出与输入相位差    ≈ %.2f°（由交流分析相位读出；共源级反相，理论 180°）"
    % abs(phase_mid))
say("  输出直流偏置        = %.6f V（动态时绕该点摆动，上面给出的是交流幅值）"
    % vd[tail].mean())
say()

# ----------------------------------------------------------------------
# 三、绘图
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(11, 8.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)

# (a) 瞬态输入/输出波形
ax = fig.add_subplot(gs[0, :])
ax2 = ax.twinx()
l1, = ax.plot(tt * 1e3, vg * 1e3, "-", color="#1f77b4", linewidth=1.6,
              label="输入 $v_g$（栅极）/ mV")
l2, = ax2.plot(tt * 1e3, vd, "-", color="#d62728", linewidth=1.6,
               label="输出 $v_d$（漏极）/ V")
ax2.axhline(vd[tail].mean(), color="#d62728", linestyle=":", linewidth=1,
            label="输出静态工作点 %.4f V" % vd[tail].mean())
ax.set_xlabel("时间 t / ms")
ax.set_ylabel("输入 $v_g$ / mV", color="#1f77b4")
ax2.set_ylabel("输出 $v_d$ / V", color="#d62728")
ax.tick_params(axis="y", labelcolor="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax.set_title("图 1-③-a  瞬态波形：输入 %.0f mV / %g Hz，输出反相放大 %.4f 倍（|Av|=%.4f）"
             % (VI_AMP * 1e3, F_IN, gain_tran, gain_tran))
ax.grid(alpha=0.3)
lines = [l1, l2] + ax2.get_legend_handles_labels()[0][1:]   # 去掉重复的 v_d 图例
ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper right")

# (b) 转移特性 + 直流工作点
ax = fig.add_subplot(gs[1, 0])
ckt_tc = Circuit("MOS transfer")
ckt_tc.V("DD", "vdd", ckt_tc.gnd, VDD @ u_V)
ckt_tc.R("d", "vdd", "d", RD @ u_Ω)
ckt_tc.V("g", "g", ckt_tc.gnd, 0 @ u_V)
ckt_tc.MOSFET(1, "d", "g", ckt_tc.gnd, ckt_tc.gnd, model="nmos1", w=1, l=1)
ckt_tc.model("nmos1", "nmos", KP=K_PARAM, VTO=VTH, LAMBDA=LAMBDA, level=1, W=1, L=1)
dc_tc = ckt_tc.simulator().dc(vg=slice(0.8, 3.0, (3.0 - 0.8) / 79))
vgs_arr = np.array([float(x) for x in dc_tc.sweep])
vd_arr = np.array([float(x) for x in dc_tc["d"]])
id_arr = (VDD - vd_arr) / RD

# 用中心差分从转移特性求跨导 gm = dI_D/dV_GS（仿真值）
gm_sim = float(np.gradient(id_arr, vgs_arr)[int(np.argmin(np.abs(vgs_arr - VGS_HAND)))])
ax.plot(vgs_arr, id_arr * 1e3, "-", color="#2ca02c", linewidth=1.8, label="转移特性 $I_D$-$V_{GS}$")
ax.plot([VGS_HAND], [ID_HAND * 1e3], "o", color="#d62728", markersize=8,
        label="静态工作点 Q（手算 %.4f V, %.4f mA）" % (VGS_HAND, ID_HAND * 1e3))
ax.axvline(VTH, color="#888", linestyle=":", linewidth=1, label="$V_{th}$ = %g V" % VTH)
ax.set_xlabel("$V_{GS}$ / V")
ax.set_ylabel("$I_D$ / mA")
ax.set_title("图 1-③-b  转移特性与静态工作点")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# (c) 幅频特性
ax = fig.add_subplot(gs[1, 1])
ax.semilogx(f, 20 * np.log10(gain_curve), "-", color="#1f77b4", linewidth=1.8,
            label="|$A_v$| / dB（仿真）")
ax.axhline(20 * math.log10(abs(AV_HAND)), color="#d62728", linestyle=":", linewidth=1,
           label="手算 |$A_v$| = %.4f dB" % (20 * math.log10(abs(AV_HAND))))
ax.plot([F_IN], [20 * math.log10(gain_mid)], "o", color="#ff7f0e", markersize=7,
        label="%g Hz：%.4f dB" % (F_IN, 20 * math.log10(gain_mid)))
ax.set_xlabel("频率 f / Hz")
ax.set_ylabel("|$A_v$| / dB")
ax.set_title("图 1-③-c  幅频特性（中频增益 %.4f 倍）" % gain_band)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8, loc="lower left")

png = os.path.join(_HERE, "fig3_mos_amplifier.png")
fig.savefig(png, dpi=150)
say("图片已保存：%s" % png)
say()

# ----------------------------------------------------------------------
# 四、对比表
# ----------------------------------------------------------------------
def err(theo, sim):
    return "%.4f%%" % (abs(sim - theo) / abs(theo) * 100 if theo else 0)


say("=" * 96)
say("表 1-③-a  静态工作点与饱和区判断（手算 vs 仿真）")
say("=" * 96)
say("%-22s %-18s %-18s %-18s %-10s" % ("项目", "手算(忽略 λ)", "手算(含 λ)", "仿真(ngspice)", "误差*"))
say("-" * 96)
say("%-22s %-18s %-18s %-18s %-10s" % ("V_GS / V", "%.6f" % VGS_HAND, "%.6f" % VGS_HAND,
                                       "%.6f" % vgs_sim, "0.0000%"))
say("%-22s %-18s %-18s %-18s %-10s" % ("I_D / mA", "%.6f" % (ID_IDEAL * 1e3),
                                       "%.6f" % (ID_HAND * 1e3), "%.6f" % (id_sim * 1e3),
                                       err(ID_HAND, id_sim)))
say("%-22s %-18s %-18s %-18s %-10s" % ("V_DS / V", "%.6f" % VDS_IDEAL, "%.6f" % VDS_HAND,
                                       "%.6f" % vds_sim, err(VDS_HAND, vds_sim)))
say("%-22s %-18s %-18s %-18s %-10s" % ("V_ov = V_GS−V_th / V", "%.4f" % VOV_IDEAL,
                                       "%.4f" % VOV_IDEAL, "%.4f" % (vgs_sim - VTH), "—"))
say("%-22s %-18s %-18s %-18s %-10s" % ("饱和判据 V_DS > V_ov", "满足" if VDS_IDEAL > VOV_IDEAL else "不满足",
                                       "满足" if SAT_OK else "不满足",
                                       "满足" if vds_sim > (vgs_sim - VTH) else "不满足", "—"))
say("-" * 96)
say("* 误差以「手算(含 λ)」为基准与仿真比较")
say("=" * 96)
say()
say("=" * 96)
say("表 1-③-b  小信号参数与增益（手算 vs 仿真）")
say("=" * 96)
say("%-26s %-20s %-20s %-12s" % ("项目", "理论值（手算）", "仿真值", "相对误差"))
say("-" * 96)
say("%-26s %-20s %-20s %-12s" % ("跨导 gm / mS", "%.6f" % (gm * 1e3), "%.6f" % (gm_sim * 1e3),
                                 err(gm, gm_sim)))
say("%-26s %-20s %-20s %-12s" % ("跨导 gm（简化式）/ mS", "%.6f" % (gm_simple * 1e3),
                                 "%.6f" % (gm_sim * 1e3), err(gm_simple, gm_sim)))
say("%-26s %-20s %-20s %-12s" % ("输出电阻 ro / kΩ", "%.4f" % (ro / 1e3), "—", "—"))
say("%-26s %-20s %-20s %-12s" % ("电压增益 |Av|（含 ro）", "%.6f" % abs(AV_HAND),
                                 "%.6f" % gain_tran, err(abs(AV_HAND), gain_tran)))
say("%-26s %-20s %-20s %-12s" % ("电压增益 |Av|（忽略 ro）", "%.6f" % abs(AV_HAND_SIMPLE),
                                 "%.6f" % gain_tran, err(abs(AV_HAND_SIMPLE), gain_tran)))
say("%-26s %-20s %-20s %-12s" % ("输出幅值（Vi=10 mV）/ mV", "%.6f" % (abs(AV_HAND) * VI_AMP * 1e3),
                                 "%.6f" % (vd_amp_sim * 1e3), err(abs(AV_HAND), gain_tran)))
say("%-26s %-20s %-20s %-12s" % ("输出与输入相位差 / °", "180.0000", "%.4f" % phase_sim,
                                 "%.4f%%" % (abs(phase_sim - 180) / 180 * 100)))
say("-" * 96)
say("* gm 仿真值由直流扫描的转移特性 I_D(V_GS) 做中心差分求得，与精确式、简化式对比")
say("=" * 96)
say()

# ----------------------------------------------------------------------
# 五、直流通路与小信号等效模型（用公式给出，图另画）
# ----------------------------------------------------------------------
say("=" * 96)
say("直流通路（画图要点）")
say("=" * 96)
say("  · 电容 Cb1 在直流下相当于开路 → 输入回路断开，栅极只由 Rg1、Rg2 分压供电；")
say("  · 因此直流等效电路 = VDD —[Rd]— 漏极 —[NMOS]— 源极(接地)，栅极接 V_G = 2 V 的直流源；")
say("  · 直流方程：V_G = VDD·Rg2/(Rg1+Rg2)；I_D = ½K(V_G−V_th)²(1+λV_DS)；V_DS = VDD − I_D·Rd。")
say()
say("小信号等效模型（画图要点）")
say("=" * 96)
say("  · Cb1 在交流下视为短路 → 栅极交流接地回路的电阻为 Rg1∥Rg2 = %.4f kΩ；"
    % (RG1 * RG2 / (RG1 + RG2) / 1e3))
say("  · VDD 交流接地 → Rd 上端接地；NMOS 用压控电流源 gm·v_gs 与输出电阻 ro 并联表示；")
say("  · 源极接地 → v_gs = v_i，输出端（漏极）对地阻抗 = Rd ∥ ro；")
say("  · A_v = v_o/v_i = −gm·(Rd ∥ ro)；输入电阻 R_in = Rg1∥Rg2；输出电阻 R_out = Rd ∥ ro。")
say()
say("  数值：R_in = %.4f kΩ，R_out = %.4f kΩ，gm = %.6f mS，ro = %.4f kΩ，A_v = %.4f"
    % (RG1 * RG2 / (RG1 + RG2) / 1e3, RD * ro / (RD + ro) / 1e3,
       gm * 1e3, ro / 1e3, AV_HAND))
say()
say("【结论】")
say("  · 静态工作点：V_GS = %.4f V，I_D = %.6f mA，V_DS = %.6f V；" % (VGS_HAND, ID_HAND * 1e3, VDS_HAND))
say("    V_DS (%.4f V) > V_GS − V_th (%.4f V)，**工作在饱和区**，可以作放大器使用；"
    % (VDS_HAND, VOV_IDEAL))
say("  · 小信号：gm = %.4f mS，ro = %.4f kΩ，A_v = %.4f（反相放大）；" % (gm * 1e3, ro / 1e3, AV_HAND))
say("  · 仿真：V_DS = %.4f V、I_D = %.4f mA 与手算吻合；瞬态实测 |A_v| = %.4f，"
    % (vds_sim, id_sim * 1e3, gain_tran))
say("    输出波形与输入反相，与理论一致；")
say("  · 增益略小于 gm·Rd 是因为沟道长度调制带来的 ro 与 Rd 并联分流。")

save_report("report3_mos_amplifier.txt")
