import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

st.set_page_config(
    page_title="Central Limit Theorem Demo",
    page_icon="📊",
    layout="wide"
)

# ===== Japanese font auto-detection =====
def _set_japanese_font():
    candidates = [
        "IPAexGothic", "IPAGothic", "Noto Sans CJK JP", "Noto Sans JP",
        "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo",
        "MS Gothic", "TakaoGothic", "VL PGothic"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None

_set_japanese_font()

st.set_page_config(page_title="LLN/CLT Playground", layout="wide")

def _dist_label(name, p):
    try:
        if name == "コーシー分布":
            return f"コーシー(x0={p['x0']:.2f}, γ={p['gamma']:.2f})"
        if name == "正規分布":
            return f"正規(μ={p['mu']:.2f}, σ={p['sigma']:.2f})"
        if name == "一様分布":
            return f"一様(a={p['a']:.2f}, b={p['b']:.2f})"
        if name == "指数分布":
            return f"指数(λ={p['lam']:.2f})"
        if name == "パレート分布":
            return f"パレート(α={p['alpha']:.2f}, x_m={p['xm']:.2f})"
        if name == "2群混合分布（正規×正規）":
            return (f"mix[ r={p['r']:.2f} : 正規A(μ={p['muA']:.2f}, σ={p['sigmaA']:.2f}), "
                    f"正規B(μ={p['muB']:.2f}, σ={p['sigmaB']:.2f}) ]")
    except Exception:
        pass
    return name

st.title("大数の法則と中心極限定理のデモ")
st.caption("左の設定パネルで分布とパラメータを選び，標本サイズ n と試行回数を指定してください．\n"
           "各試行の標本平均をヒストグラム表示します．")

# ========== Sidebar (all controls) ==========
with st.sidebar:
    st.header("設定")

    dist_name = st.selectbox("分布を選ぶ", [
        "正規分布", "一様分布", "指数分布", "コーシー分布", "パレート分布", "2群混合分布（正規×正規）"
    ])

    # Sample size & trials: number inputs
    n = st.number_input("標本サイズ n", min_value=1, max_value=50000, value=30, step=1)
    trials = st.number_input("トライアル回数", min_value=1, max_value=100000, value=1000, step=1)
    seed = st.number_input("乱数シード（累積用）", min_value=0, max_value=2_147_483_647, value=0, step=1)

    # Distribution parameter sliders
    params = {}
    if dist_name == "正規分布":
        params["mu"] = st.slider("平均 μ", -10.0, 10.0, 0.0, step=0.1)
        params["sigma"] = st.slider("標準偏差 σ", 0.1, 10.0, 1.0, step=0.1)

    elif dist_name == "一様分布":
        a = st.slider("下限 a", -10.0, 9.0, -0.5, step=0.1)
        b = st.slider("上限 b", a+0.1, 10.0, 0.5, step=0.1)
        params["a"], params["b"] = a, b

    elif dist_name == "指数分布":
        params["lam"] = st.slider("レート λ (平均=1/λ)", 0.1, 10.0, 1.0, step=0.1)

    elif dist_name == "コーシー分布":
        params["x0"] = st.slider("位置 x0", -10.0, 10.0, 0.0, step=0.1)
        params["gamma"] = st.slider("スケール γ", 0.1, 5.0, 1.0, step=0.1)

    elif dist_name == "パレート分布":
        params["alpha"] = st.slider("形状 α", 0.1, 8.0, 2.0, step=0.1)
        params["xm"] = st.slider("スケール x_m (最小値)", 0.1, 10.0, 1.0, step=0.1)

    elif dist_name == "2群混合分布（正規×正規）":
        st.subheader("グループA")
        params["muA"] = st.slider("Aの平均 μ_A", -10.0, 10.0, 0.0, step=0.1)
        params["sigmaA"] = st.slider("Aの標準偏差 σ_A", 0.1, 10.0, 1.0, step=0.1)
        st.subheader("グループB")
        params["muB"] = st.slider("Bの平均 μ_B", -10.0, 10.0, 3.0, step=0.1)
        params["sigmaB"] = st.slider("Bの標準偏差 σ_B", 0.1, 10.0, 1.0, step=0.1)
        params["r"] = st.slider("Aの比率 r（n×r 個がA）", 0.0, 1.0, 0.5, step=0.01)

    st.markdown("---")
    st.subheader("表示範囲の指定（任意）")
    clip_mean = st.checkbox("標本平均ヒストグラムの x 範囲を指定する", value=False)
    mean_xmin = st.number_input("標本平均 x 最小", value=-10.0, step=0.5)
    mean_xmax = st.number_input("標本平均 x 最大", value=10.0, step=0.5)

    clip_view = st.checkbox("元の分布ヒストの x 範囲を指定する", value=False)
    view_xmin = st.number_input("元の分布 x 最小", value=-20.0, step=1.0)
    view_xmax = st.number_input("元の分布 x 最大", value=20.0, step=1.0)

# ========== RNG (seeded for cumulative behavior) ==========
rng = np.random.default_rng(int(seed))

# ========== Sampling helpers ==========
def sample_once(dist, n, p):
    if dist == "正規分布":
        return rng.normal(p["mu"], p["sigma"], n)
    elif dist == "一様分布":
        return rng.uniform(p["a"], p["b"], n)
    elif dist == "指数分布":
        return rng.exponential(1.0/p["lam"], n)
    elif dist == "コーシー分布":
        return p["x0"] + p["gamma"] * rng.standard_cauchy(n)
    elif dist == "パレート分布":
        return (rng.pareto(p["alpha"], n) + 1.0) * p["xm"]
    elif dist == "2群混合分布（正規×正規）":
        mask = rng.random(n) < p["r"]
        x = np.empty(n, dtype=float)
        m = int(mask.sum())
        if m>0:
            x[mask] = rng.normal(p["muA"], p["sigmaA"], m)
        if n-m>0:
            x[~mask] = rng.normal(p["muB"], p["sigmaB"], n-m)
        return x
    else:
        return rng.normal(0,1,n)

def theoretical_mean(dist, p):
    if dist == "正規分布":
        return p["mu"]
    elif dist == "一様分布":
        return 0.5*(p["a"]+p["b"])
    elif dist == "指数分布":
        return 1.0/p["lam"]
    elif dist == "コーシー分布":
        return None  # undefined
    elif dist == "パレート分布":
        return p["xm"] * p["alpha"]/(p["alpha"]-1.0) if p["alpha"]>1.0 else None
    elif dist == "2群混合分布（正規×正規）":
        return p["r"]*p["muA"] + (1.0-p["r"])*p["muB"]
    return None

def theoretical_var_single_X(dist, p):
    if dist == "正規分布":
        return p["sigma"]**2
    elif dist == "一様分布":
        return (p["b"]-p["a"])**2 / 12.0
    elif dist == "指数分布":
        return 1.0/(p["lam"]**2)
    elif dist == "コーシー分布":
        return None
    elif dist == "パレート分布":
        if p["alpha"]>2.0:
            a = p["alpha"]; xm = p["xm"]
            return (a * xm**2) / ((a-1.0)**2 * (a-2.0))
        else:
            return None
    elif dist == "2群混合分布（正規×正規）":
        r = p["r"]
        muA, muB = p["muA"], p["muB"]
        sA2, sB2 = p["sigmaA"]**2, p["sigmaB"]**2
        mu_mix = r*muA + (1.0-r)*muB
        return r*(sA2 + (muA-mu_mix)**2) + (1.0-r)*(sB2 + (muB-mu_mix)**2)
    return None

def normal_pdf(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2) / (np.sqrt(2*np.pi)*sigma)

# ========== Compute sample means across trials ==========
means = np.empty(int(trials), dtype=float)
for t in range(int(trials)):
    x = sample_once(dist_name, int(n), params)
    means[t] = float(np.mean(x))

emp_mean = float(np.mean(means))
emp_std = float(np.std(means, ddof=1)) if int(trials)>1 else np.nan

theo_mean = theoretical_mean(dist_name, params)
varX = theoretical_var_single_X(dist_name, params)
theo_std = None
if (theo_mean is not None) and (varX is not None) and (int(n)>0):
    theo_std = np.sqrt(varX / int(n))

# ========== Plotting ==========
col1, col2 = st.columns([2,1])

with col1:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    finite = means[np.isfinite(means)]
    if finite.size>0:
        # Optionally clip the histogram range for sample means
        hist_kwargs = dict(bins=60, density=True, alpha=0.6)
        if clip_mean and (mean_xmax > mean_xmin):
            hist_kwargs["range"] = (float(mean_xmin), float(mean_xmax))
            ax.set_xlim(float(mean_xmin), float(mean_xmax))
        ax.hist(finite, **hist_kwargs)

        ax.axvline(emp_mean, linestyle="--", linewidth=2, label="Empirical mean")
        if theo_mean is not None:
            ax.axvline(theo_mean, linestyle="-", linewidth=2, label="Theoretical mean")

        # X-range for curves
        if clip_mean and (mean_xmax > mean_xmin):
            xs = np.linspace(float(mean_xmin), float(mean_xmax), 400)
        else:
            x_min, x_max = float(np.min(finite)), float(np.max(finite))
            span = x_max - x_min if x_max>x_min else (np.abs(emp_std) if np.isfinite(emp_std) else 1.0)
            pad = 0.15*span if span>0 else 1.0
            xs = np.linspace(x_min - pad, x_max + pad, 400)

        # Theoretical normal curve (if mean & variance are finite)
        if (theo_mean is not None) and (theo_std is not None) and (theo_std>0):
            ax.plot(xs, normal_pdf(xs, theo_mean, theo_std), linewidth=2, label="Theoretical normal")

        # Empirical normal fit (fit to means)
        if np.isfinite(emp_std) and emp_std>0:
            ax.plot(xs, normal_pdf(xs, emp_mean, emp_std), linewidth=2, linestyle="--", label="Empirical normal fit")

        title = f"Distribution of sample means (n={int(n)}, trials={int(trials)})\n" 
        ax.set_title(title)
        ax.set_xlabel("Sample means")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("")

with col2:
    st.subheader("数値サマリー")
    st.write(f"- 経験平均（全トライアルの平均）: **{emp_mean:.6g}**")
    if np.isfinite(emp_std):
        st.write(f"- 標本平均の経験標準偏差: **{emp_std:.6g}**")
    if theo_mean is not None:
        st.write(f"- 理論平均 μ: **{theo_mean:.6g}**")
    else:
        st.write("- 理論平均 μ: **定義されない**（例：コーシー、パレート α≤1）")
    if (theo_std is not None):
        st.write(f"- 理論的な標本平均の標準偏差: **{theo_std:.6g}**  （Var(X̄)=Var(X)/n）")
        st.caption("i.i.d.（または2群混合でも分散有限）なら、n を増やすと理論曲線は急速に尖ります。")
    else:
        st.caption("この分布/パラメータでは分散が有限でないため、理論正規カーブは表示しません。")

with st.expander("元の確率変数 X の分布を表示"):
    fig0, ax0 = plt.subplots(figsize=(8, 4.0))
    # For viewing, draw enough samples without affecting cumulative behavior (use a separate RNG)
    view_rng = np.random.default_rng(int(seed)+12345)
    def sample_once_view(dist, n, p, rngv):
        if dist == "正規分布":
            return rngv.normal(p["mu"], p["sigma"], n)
        elif dist == "一様分布":
            return rngv.uniform(p["a"], p["b"], n)
        elif dist == "指数分布":
            return rngv.exponential(1.0/p["lam"], n)
        elif dist == "コーシー分布":
            return p["x0"] + p["gamma"] * rngv.standard_cauchy(n)
        elif dist == "パレート分布":
            return (rngv.pareto(p["alpha"], n) + 1.0) * p["xm"]
        elif dist == "2群混合分布（正規×正規）":
            mask = rngv.random(n) < p["r"]
            x = np.empty(n, dtype=float)
            m = int(mask.sum())
            if m>0:
                x[mask] = rngv.normal(p["muA"], p["sigmaA"], m)
            if n-m>0:
                x[~mask] = rngv.normal(p["muB"], p["sigmaB"], n-m)
            return x
        else:
            return rngv.normal(0,1,n)

    Nview = max(int(n), 10000)
    x_view = sample_once_view(dist_name, Nview, params, view_rng)

    view_hist_kwargs = dict(bins=100, density=True, alpha=0.6)
    if clip_view and (view_xmax > view_xmin):
        view_hist_kwargs["range"] = (float(view_xmin), float(view_xmax))
        ax0.set_xlim(float(view_xmin), float(view_xmax))
    ax0.hist(x_view, **view_hist_kwargs)

    ax0.set_title(f"Underlying distribution (10,000 samples for display)")
    ax0.set_xlabel("x")
    ax0.set_ylabel("Density")
    st.pyplot(fig0, clear_figure=True)
