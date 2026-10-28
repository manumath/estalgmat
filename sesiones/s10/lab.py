import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import optimize, stats

    from estalgmat import retornos_garch_t

    return mo, np, optimize, pd, plt, retornos_garch_t, stats


@app.cell
def _(mo):
    mo.md(r"# S10 · Estimación puntual: MM y MV")
    return


@app.cell
def _(mo):
    mo.md(r"## 1. Sesgo y ECM de estimadores de $\sigma^2$")
    return


@app.cell
def _(mo):
    n = mo.ui.slider(3, 100, value=10, step=1, label="n")
    n
    return (n,)


@app.cell
def _(n, np, pd):
    rng = np.random.default_rng(10)
    X = rng.normal(0, 2, (100_000, n.value))
    est = {"MV (÷n)": X.var(1, ddof=0), "S² (÷n−1)": X.var(1, ddof=1)}
    pd.DataFrame(
        {k: {"sesgo": v.mean() - 4, "varianza": v.var(), "ECM": ((v - 4) ** 2).mean()} for k, v in est.items()}
    ).round(4)
    return X, est, rng


@app.cell
def _(mo):
    mo.md(r"## 2. EMV numérico de la $t$ de Student")
    return


@app.cell
def _(mo):
    nu_v = mo.ui.slider(2.5, 15, value=4, step=0.5, label="ν verdadero")
    n_obs = mo.ui.slider(300, 5000, value=1500, step=100, label="n")
    mo.hstack([nu_v, n_obs], justify="start")
    return n_obs, nu_v


@app.cell
def _(n_obs, np, nu_v, optimize, stats):
    rng2 = np.random.default_rng(5)
    x = 0.1 + 1.5 * rng2.standard_t(nu_v.value, n_obs.value)

    def nll(th):
        mu, ls, lu = th
        return -stats.t.logpdf(x, 2 + np.exp(lu), mu, np.exp(ls)).sum()

    op = optimize.minimize(nll, [x.mean(), np.log(x.std()), np.log(3.0)], method="BFGS")
    mu_h, sg_h, nu_h = op.x[0], np.exp(op.x[1]), 2 + np.exp(op.x[2])
    k_ex = stats.kurtosis(x)
    nu_mm = 4 + 6 / k_ex if k_ex > 0 else float("nan")
    return k_ex, mu_h, nll, nu_h, nu_mm, op, rng2, sg_h, x


@app.cell
def _(mo, mu_h, nu_h, nu_mm, nu_v, sg_h):
    mo.md(
        f"""
        | | valor |
        |---|---|
        | ν verdadero | {nu_v.value} |
        | **ν̂ (MV)** | {nu_h:.2f} |
        | ν̂ (MM, curtosis) | {nu_mm:.2f} |
        | μ̂ | {mu_h:.3f} (verdadero 0.100) |
        | escala σ̂ | {sg_h:.3f} (verdadera 1.500) |
        """
    )
    return


@app.cell
def _(np, nu_h, plt, stats, x):
    grid = np.linspace(2.5, 20, 50)
    ll = np.array([stats.t.logpdf(x, v, *stats.t.fit(x, f0=v)[1:]).sum() for v in grid])
    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.plot(grid, ll)
    ax.axhline(ll.max() - stats.chi2.ppf(0.95, 1) / 2, c="r", ls="--")
    ax.axvline(nu_h, c="k", lw=0.8)
    ax.set_xlabel("ν")
    ax.set_ylabel("log-verosimilitud perfilada")
    fig
    return ax, fig, grid, ll


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Repite con $\nu=3$ y $n=300$. ¿Qué estimador de $\nu$ (MM o MV) es más estable al cambiar la semilla? Escribe un bucle de 200 réplicas y compara sus desviaciones estándar.
        """
    )
    return


if __name__ == "__main__":
    app.run()
