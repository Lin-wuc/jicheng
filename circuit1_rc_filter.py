# -*- coding: utf-8 -*-
"""
进阶挑战 · 任务 1-① RC 低通滤波电路
====================================
任务书要求（第五节）：
    ① RC 滤波电路：自己画的电路图（手绘/画图软件均可，本文件不画）；
       方波输入/输出瞬态波形图 + 波特图；τ、截止频率的「手算 vs 仿真」表。

运行方式（Windows，PySpice + 本机 ngspice 运行库已就绪）：
    set PYTHONPATH=E:\\jicheng-main\\.pylib
    python circuit1_rc_filter.py

自定参数（取整数便于手算）：R = 1 kΩ，C = 100 nF
"""

import os
import sys
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
    for depth in range(1, 6):                     # 兜底：向上找 _tools/spice_root
        for name in ("_tools/spice_root/Spice64_dll", "_tools/ngspice/Spice64_dll"):
            cand = os.path.abspath(os.path.join(_HERE, *([".."] * depth), name))
            if os.path.isdir(os.path.join(cand, "dll-vs")):
                return os.path.join(cand, "share", "ngspice"), os.path.join(cand, "dll-vs")
    raise RuntimeError("找不到 ngspice 运行库，请设置环境变量 NGSPICE_DLL_DIR")


_spice_lib, _spice_dll = _find_ngspice()
# ngspice 的 spinit 里用 "../lib/ngspice/*.cm" 引用代码模型，
# 所以 SPICE_LIB_DIR 要指向 dll-vs 的上一级（即 Spice64_dll 根目录）
os.environ["SPICE_LIB_DIR"] = os.path.dirname(_spice_dll)
os.add_dll_directory(_spice_dll)

from PySpice.Spice.Netlist import Circuit           # noqa: E402
from PySpice.Unit import u_Ω, u_nF, u_V, u_Hz, u_s  # noqa: E402

import numpy as np                                  # noqa: E402
import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

np.set_printoptions(suppress=True)

# ----------------------------------------------------------------------
# 输出工具：控制台可能是 GBK，无法显示 µ/Ω 等字符，自动降级为 ASCII
#           同时把所有内容写入 UTF-8 报告文件，方便直接粘进 README
# ----------------------------------------------------------------------
_REPORT = []


def _to_ascii(text):
    return (text.replace("µ", "u").replace("Ω", "ohm").replace("ω", "w")
                .replace("τ", "tau").replace("π", "pi").replace("φ", "phi")
                .replace("√", "sqrt").replace("→", "->").replace("×", "x")
                .replace("《", "\"").replace("》", "\"").replace("·", "*")
                .replace("①", "(1)").replace("—", "-").replace("√", "sqrt"))


def say(text=""):
    """打印 + 收集到报告（控制台编码不支持时自动降级）。"""
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
R_VAL = 1e3            # 1 kΩ
C_VAL = 100e-9         # 100 nF
V_HIGH = 1.0           # 方波高电平（0 → 1 V）

TAU_HAND = R_VAL * C_VAL                       # τ = RC
FC_HAND = 1.0 / (2 * math.pi * TAU_HAND)       # fc = 1/(2πRC)
SW_FREQ = 1.0 / (20 * TAU_HAND)                # 方波周期取 20τ，半周期 10τ，足够接近稳态
PERIOD = 1.0 / SW_FREQ

print("=" * 78)
say("=" * 78)
say("任务 1-① RC 低通滤波电路      R = %.3g Ω,  C = %.3g F" % (R_VAL, C_VAL))
say("=" * 78)
say()
say("【手算过程】")
say("  τ  = R·C = %.6g × %.6g = %.6g s = %.4g µs" % (R_VAL, C_VAL, TAU_HAND, TAU_HAND * 1e6))
say("  fc = 1/(2πRC) = 1/(2π × %.6g) = %.6g Hz" % (TAU_HAND, FC_HAND))
say("  传递函数 H(jω) = 1/(1 + jωRC)，相移 φ = -arctan(2πfRC)，fc 处 φ = -45°")
say("  阶跃响应 v(t) = V·(1 - e^(-t/τ))：")
say("     t=τ → 63.2121%   t=2τ → 86.4665%   t=3τ → 95.0213%   t=5τ → 99.3262%")
say()

# ----------------------------------------------------------------------
# 一、瞬态仿真：方波输入（验证 τ 与积分效应）
# ----------------------------------------------------------------------
T_STEP = TAU_HAND / 2000          # 步长 τ/2000，足以看清指数上升沿
T_END = 6 * PERIOD                # 跑 6 个方波周期

ckt_t = Circuit("RC low-pass transient")
ckt_t.PulseVoltageSource("in", "vi", ckt_t.gnd,
                         initial_value=0,
                         pulsed_value=V_HIGH,
                         delay_time=0,
                         rise_time=1e-9,
                         fall_time=1e-9,
                         pulse_width=0.5 * PERIOD,
                         period=PERIOD)
ckt_t.R(1, "vi", "out", R_VAL @ u_Ω)
ckt_t.C(1, "out", ckt_t.gnd, (C_VAL * 1e9) @ u_nF)

ana_t = ckt_t.simulator(temperature=25, nominal_temperature=25).transient(
    step_time=T_STEP @ u_s, end_time=T_END @ u_s)

t = np.array([float(x) for x in ana_t.time])
vin = np.array([float(x) for x in ana_t["vi"]])
vout = np.array([float(x) for x in ana_t["out"]])

# 显式找出稳态里最后一次上升沿：输入方波由低变高、且已进入稳态（跳过前 4 个周期）
cross = np.where((vin[:-1] < 0.5 * V_HIGH) & (vin[1:] >= 0.5 * V_HIGH))[0]
cross = cross[t[cross] >= 4 * PERIOD]
t_rise = float(t[cross[-1] + 1])
sel = (t >= t_rise) & (t <= t_rise + 0.5 * PERIOD)
tr = t[sel] - t_rise
vr = vout[sel]
vin_r = vin[sel]

if len(tr) < 10 or vr[0] > 0.5 * V_HIGH:
    raise RuntimeError("未能定位上升沿，请检查方波源参数（t_rise=%.6g）" % t_rise)


def v_at(sec):
    i = int(np.argmin(np.abs(tr - sec)))
    return float(vr[i])


v_tau, v_2tau = v_at(TAU_HAND), v_at(2 * TAU_HAND)
v_3tau, v_5tau = v_at(3 * TAU_HAND), v_at(5 * TAU_HAND)

# 由 v = 0.6321·V 反推实测 τ
idx = int(np.argmin(np.abs(vr - 0.63212 * V_HIGH)))
tau_sim = float(tr[idx])

say("【瞬态仿真】方波 f = %.4g Hz（周期 %.4g ms），取稳态上升沿测量：" % (SW_FREQ, PERIOD * 1e3))
say("  v(τ)  = %.6f V   （手算 %.6f V）" % (v_tau,  V_HIGH * (1 - math.exp(-1))))
say("  v(2τ) = %.6f V   （手算 %.6f V）" % (v_2tau, V_HIGH * (1 - math.exp(-2))))
say("  v(3τ) = %.6f V   （手算 %.6f V）" % (v_3tau, V_HIGH * (1 - math.exp(-3))))
say("  v(5τ) = %.6f V   （手算 %.6f V）" % (v_5tau, V_HIGH * (1 - math.exp(-5))))
say("  由 63.2121%% 点反推 τ_sim = %.6g µs   （手算 τ = %.6g µs）" % (tau_sim * 1e6, TAU_HAND * 1e6))
say()

# ----------------------------------------------------------------------
# 二、交流仿真：波特图（验证 fc 与相移）
# ----------------------------------------------------------------------
F_START, F_STOP, PTS_PER_DEC = 10.0, 100e3, 40

ckt_ac = Circuit("RC low-pass ac")
ckt_ac.SinusoidalVoltageSource("in", "vi", ckt_ac.gnd, amplitude=1 @ u_V)
ckt_ac.R(1, "vi", "out", R_VAL @ u_Ω)
ckt_ac.C(1, "out", ckt_ac.gnd, (C_VAL * 1e9) @ u_nF)

ana_ac = ckt_ac.simulator().ac(start_frequency=F_START @ u_Hz,
                               stop_frequency=F_STOP @ u_Hz,
                               number_of_points=PTS_PER_DEC,
                               variation="dec")

f = np.array([float(x) for x in ana_ac.frequency])
h = ana_ac["out"].as_ndarray()          # 复数数组（实部 + 虚部）
mag = np.abs(h)
mag_db = 20 * np.log10(mag)
phase = np.degrees(np.angle(h))


def curve_at(freq):
    i = int(np.argmin(np.abs(f - freq)))
    return float(mag[i]), float(mag_db[i]), float(phase[i])


def fc_from_curve():
    """在幅频曲线上找 |H| 首次跌破 1/√2 的频率并线性插值。"""
    target = 1 / math.sqrt(2)
    for i in range(1, len(mag)):
        if mag[i] <= target <= mag[i - 1]:
            return float(f[i - 1] + (target - mag[i - 1]) * (f[i] - f[i - 1])
                         / (mag[i] - mag[i - 1]))
    return float("nan")


fc_sim = fc_from_curve()
mag_lf, db_lf, ph_lf = curve_at(F_START)
mag_hf, db_hf, ph_hf = curve_at(F_STOP)
mag_fc, db_fc, ph_fc = curve_at(FC_HAND)
mag_hf_theory = 1 / math.sqrt(1 + (F_STOP / FC_HAND) ** 2)

say("【交流仿真】%g Hz ~ %g Hz，每十倍频程 %d 点" % (F_START, F_STOP, PTS_PER_DEC))
say("  低频 |H| (10 Hz)  = %.6f  （理论 1.000000，0 dB）" % mag_lf)
say("  高频 |H| (100 kHz) = %.6f  （理论 %.6f）" % (mag_hf, mag_hf_theory))
say("  曲线插值得 fc = %.4g Hz  （手算 %.4g Hz，误差 %.4f%%）" % (fc_sim, FC_HAND, abs(fc_sim - FC_HAND) / FC_HAND * 100))
say("  fc 处：|H| = %.6f（理论 %.6f）, %.4f dB（理论 -3.0103 dB）, φ = %.4f°（理论 -45°）" % (mag_fc, 1 / math.sqrt(2), db_fc, ph_fc))
say()

# ----------------------------------------------------------------------
# 三、绘制：瞬态波形 + 波特图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 8.4))

ax = axes[0]
ax.plot(t * 1e3, vin, "--", color="#888888", linewidth=1.1, label="输入方波 $v_{in}$")
ax.plot(t * 1e3, vout, "-", color="#1f77b4", linewidth=1.8, label="输出 $v_{out}$")
ax.axhline(0.63212 * V_HIGH, color="#d62728", linestyle=":", linewidth=1)
ax.annotate("t=τ 处 63.21%%（仿真 %.4f V）" % v_tau,
            xy=((t_rise + TAU_HAND) * 1e3, v_tau),
            xytext=((t_rise + 0.12 * PERIOD) * 1e3, 0.80),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1),
            color="#d62728", fontsize=9)
ax.set_xlim((t_rise - 0.8 * PERIOD) * 1e3, (t_rise + 1.4 * PERIOD) * 1e3)
ax.set_xlabel("时间 t / ms")
ax.set_ylabel("电压 v / V")
ax.set_title("图 1-①-a  方波输入/输出瞬态波形（R=1 kΩ, C=100 nF, τ=%.0f µs, f=%.0f Hz）"
             % (TAU_HAND * 1e6, SW_FREQ))
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=9)

ax = axes[1]
ax.semilogx(f, mag_db, "-", color="#1f77b4", linewidth=1.8, label="幅频 |H| / dB（仿真）")
ax.axvline(FC_HAND, color="#d62728", linestyle=":", linewidth=1,
           label="手算 $f_c$ = %.1f Hz" % FC_HAND)
ax.axhline(-3.0103, color="#d62728", linestyle=":", linewidth=1)
f_ref = np.array([FC_HAND, F_STOP])
ax.semilogx(f_ref, -3.0103 - 20 * np.log10(f_ref / FC_HAND), "--",
            color="#2ca02c", linewidth=1, label="-20 dB/dec 渐近线")
ax2 = ax.twinx()
ax2.semilogx(f, phase, "-", color="#ff7f0e", linewidth=1.4, label="相移 φ（仿真）")
ax2.axhline(-45, color="#ff7f0e", linestyle=":", linewidth=1)
ax2.set_ylabel("相移 φ / °", color="#ff7f0e")
ax2.tick_params(axis="y", labelcolor="#ff7f0e")
ax2.set_ylim(-95, 5)
ax.set_xlabel("频率 f / Hz")
ax.set_ylabel("幅频 |H| / dB")
ax.set_title("图 1-①-b  波特图（$f_c$ 处 %.4f dB、相移 %.2f°）" % (db_fc, ph_fc))
ax.grid(alpha=0.3, which="both")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)

fig.tight_layout()
png = os.path.join(_HERE, "fig1_rc_filter.png")
fig.savefig(png, dpi=150)
say("图片已保存：%s" % png)
say()

# ----------------------------------------------------------------------
# 四、对比表
# ----------------------------------------------------------------------
def row(name, theo, sim):
    return name, theo, sim, "%.3f%%" % (abs(sim - theo) / abs(theo) * 100 if theo else 0)


say("=" * 84)
say("表 1-①-a  RC 低通滤波器「手算 vs 仿真」——频率特性")
say("=" * 84)
say("%-24s %-18s %-18s %-12s" % ("项目", "理论值（手算）", "仿真值", "相对误差"))
say("-" * 84)
table_a = [
    row("时间常数 τ",  TAU_HAND, tau_sim),
    row("截止频率 fc", FC_HAND, fc_sim),
    row("|H| 低频 (10 Hz)", 1.0, mag_lf),
    row("|H| 截止处 (fc)", 1 / math.sqrt(2), mag_fc),
    row("衰减 截止处 (fc)", -3.0103, db_fc),
    row("相移 截止处 (fc)", -45.0, ph_fc),
    row("|H| 高频 (100 kHz)", mag_hf_theory, mag_hf),
]
for name, theo, sim, err in table_a:
    if name.startswith("时间常数"):
        say("%-24s %-18s %-18s %-12s" % (name, "%.4g µs" % (theo * 1e6), "%.4g µs" % (sim * 1e6), err))
    elif name.startswith("截止频率"):
        say("%-24s %-18s %-18s %-12s" % (name, "%.2f Hz" % theo, "%.2f Hz" % sim, err))
    elif "相移" in name:
        say("%-24s %-18s %-18s %-12s" % (name, "%.4f°" % theo, "%.4f°" % sim, err))
    elif "衰减" in name:
        say("%-24s %-18s %-18s %-12s" % (name, "%.4f dB" % theo, "%.4f dB" % sim, err))
    else:
        say("%-24s %-18s %-18s %-12s" % (name, "%.6f" % theo, "%.6f" % sim, err))
say("=" * 84)
say()
say("=" * 84)
say("表 1-①-b  方波瞬态响应：阶跃后各时刻的输出电压（输入 0→1 V）")
say("=" * 84)
say("%-14s %-18s %-18s %-12s" % ("时刻", "理论 v / V", "仿真 v / V", "相对误差"))
say("-" * 84)
for label, mult in [("t = τ", 1), ("t = 2τ", 2), ("t = 3τ", 3), ("t = 5τ", 5)]:
    theo = V_HIGH * (1 - math.exp(-mult))
    sim = v_at(mult * TAU_HAND)
    say("%-14s %-18.6f %-18.6f %-12s" % (label, theo, sim, "%.4f%%" % (abs(sim - theo) / theo * 100)))
say("=" * 84)
say()
say("【结论】")
say("  · τ 与 fc 的仿真值和手算值吻合（误差来自仿真步长与曲线插值，<1%）；")
say("  · fc 处幅值 -3.01 dB、相移 -45°，符合一阶低通滤波器特征；")
say("  · fc 以上幅频按 -20 dB/十倍频程 下降，与渐近线重合；")
say("  · 方波经该电路后边沿按 e 指数规律变化，体现电容的充放电（积分）作用。")

save_report("report1_rc_filter.txt")
