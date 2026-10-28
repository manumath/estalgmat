import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    return mo, np, pd, plt, stats


@app.cell
def _(mo):
    mo.md(r"# S06 · Variables aleatorias y momentos")
    return


@app.cell
def _(mo):
    familia = mo.ui.dropdown(
        ["normal", "t(5)", "t(3)", "exponencial", "lognormal(0.5)"], value="t(5)", label="distribución"
    )
    n = mo.ui.slider(1_000, 500_000, value=100_000, step=1_000, label="n")
    mo.hstack([familia, n], justify="start")
    return familia, n


@app.cell
def _(familia, n, np, pd, stats):
    dd = {
        "normal": stats.norm(),
        "t(5)": stats.t(5),
        "t(3)": stats.t(3),
        "exponencial": stats.expon(),
        "lognormal(0.5)": stats.lognorm(0.5),
    }[familia.value]
    x = dd.rvs(n.value, random_state=np.random.default_rng(0))
    m, v, sk, ku = dd.stats(moments="mvsk")
    tabla = pd.DataFrame(
        {
            "teórico": [float(m), float(v), float(sk), float(ku)],
            "muestral": [x.mean(), x.var(ddof=1), stats.skew(x), stats.kurtosis(x)],
        },
        index=["media", "varianza", "asimetría", "curtosis exc."],
    ).round(3)
    tabla
    return dd, ku, m, sk, tabla, v, x


@app.cell
def _(mo):
    mo.md(
        "Con `t(3)` y `t(5)` la curtosis teórica es `inf` o poco estable: "
        "cambia la semilla y observa cuánto varían la asimetría y la curtosis muestrales."
    )
    return


@app.cell
def _(mo):
    mo.md(r"## Jensen y Monte Carlo: $E[e^X]$ con $X\sim N(0,\sigma^2)$")
    return


@app.cell
def _(mo):
    sig = mo.ui.slider(0.1, 1.5, value=0.5, step=0.1, label="σ")
    sig
    return (sig,)


@app.cell
def _(mo, np, sig):
    xx = np.random.default_rng(7).normal(0, sig.value, 1_000_000)
    est = np.exp(xx)
    ee = est.std(ddof=1) / np.sqrt(len(est))
    mo.md(
        f"$E[e^X]\\approx$ **{est.mean():.4f}** ± {1.96 * ee:.4f} · teórico $e^{{\\sigma^2/2}}=$ "
        f"**{np.exp(sig.value**2 / 2):.4f}** · $e^{{E[X]}}=$ {np.exp(xx.mean()):.4f}"
    )
    return ee, est, xx


@app.cell
def _(mo):
    mo.md(r"## Portafolio de dos activos")
    return


@app.cell
def _(mo):
    rho = mo.ui.slider(-1, 1, value=0.0, step=0.1, label="ρ")
    s1 = mo.ui.slider(0.05, 0.5, value=0.2, step=0.05, label="σ₁")
    s2 = mo.ui.slider(0.05, 0.5, value=0.3, step=0.05, label="σ₂")
    mo.hstack([rho, s1, s2], justify="start")
    return rho, s1, s2


@app.cell
def _(np, plt, rho, s1, s2):
    w = np.linspace(0, 1, 201)
    sp = np.sqrt(w**2 * s1.value**2 + (1 - w) ** 2 * s2.value**2 + 2 * w * (1 - w) * rho.value * s1.value * s2.value)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(w, sp)
    ax.scatter([w[sp.argmin()]], [sp.min()], c="r", zorder=3)
    ax.set_xlabel("peso activo 1")
    ax.set_ylabel("σ del portafolio")
    ax.set_title(f"mínimo σ = {sp.min():.3f} con w₁ = {w[sp.argmin()]:.2f}")
    fig
    return ax, fig, sp, w


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Deriva el peso de mínima varianza $w^\star=\dfrac{\sigma_2^2-\rho\sigma_1\sigma_2}{\sigma_1^2+\sigma_2^2-2\rho\sigma_1\sigma_2}$ y verifícalo contra el mínimo del gráfico.
        """
    )
    return


if __name__ == "__main__":
    app.run()
