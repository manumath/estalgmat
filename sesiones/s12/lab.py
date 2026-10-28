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

    from estalgmat import retornos_garch_t

    return mo, np, pd, plt, retornos_garch_t, stats


@app.cell
def _(mo):
    mo.md(r"# S12 · Pruebas de hipótesis paramétricas")
    return


@app.cell
def _(mo):
    mu = mo.ui.slider(0.0, 0.10, value=0.04, step=0.01, label="μ diaria verdadera (%)")
    n = mo.ui.slider(250, 5000, value=1000, step=250, label="n")
    mo.hstack([mu, n], justify="start")
    return mu, n


@app.cell
def _(mo, mu, n, np, retornos_garch_t, stats):
    x = retornos_garch_t(n.value, mu=mu.value, seed=7)["ret"].to_numpy()
    t, p = stats.ttest_1samp(x, 0)
    lo, hi = stats.t.interval(0.95, len(x) - 1, x.mean(), stats.sem(x))
    mo.md(
        f"Media = **{x.mean():.4f}** · t = **{t:.2f}** · p = **{p:.4f}** · IC95 = [{lo:.4f}, {hi:.4f}] → "
        f"{'**rechaza** H₀: μ=0' if p < 0.05 else '**no rechaza** H₀: μ=0'}"
    )
    return hi, lo, p, t, x


@app.cell
def _(mo):
    mo.md(r"## Potencia")
    return


@app.cell
def _(mo):
    alfa = mo.ui.slider(0.01, 0.10, value=0.05, step=0.01, label="α")
    delta = mo.ui.slider(0.0, 0.5, value=0.1, step=0.05, label="δ = μ/σ")
    mo.hstack([alfa, delta], justify="start")
    return alfa, delta


@app.cell
def _(alfa, delta, mo, np, plt, stats):
    rng = np.random.default_rng(12)

    def potencia(d, n, R=3000):
        z = rng.normal(d, 1, (R, n))
        tt = z.mean(1) / (z.std(1, ddof=1) / np.sqrt(n))
        return np.mean(np.abs(tt) > stats.t.ppf(1 - alfa.value / 2, n - 1))

    ns = np.array([20, 50, 100, 200, 400, 800, 1600])
    pot = [potencia(delta.value, int(k)) for k in ns]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.semilogx(ns, pot, "o-")
    ax.axhline(0.8, ls="--", c="gray")
    ax.set_xlabel("n")
    ax.set_ylabel("potencia")
    ax.set_ylim(0, 1)
    mo.vstack([fig, mo.md(f"Con δ={delta.value} se requiere n ≈ **{int(np.ceil(((1.96 + 0.84) / max(delta.value, 1e-9)) ** 2))}** para 80 % de potencia (aprox. normal).")])
    return ax, fig, ns, pot, potencia, rng


@app.cell
def _(mo):
    mo.md(r"## Tamaño real de la prueba t con datos GARCH-t")
    return


@app.cell
def _(mo):
    reps = mo.ui.slider(300, 2000, value=600, step=100, label="réplicas")
    ne = mo.ui.slider(50, 500, value=250, step=50, label="n")
    mo.hstack([reps, ne], justify="start")
    return ne, reps


@app.cell
def _(mo, ne, np, pd, reps, retornos_garch_t, stats):
    rg = np.random.default_rng(1)
    gens = {
        "normal iid": lambda n: rg.normal(0, 1, n),
        "t(3) iid": lambda n: rg.standard_t(3, n),
        "GARCH-t": lambda n: retornos_garch_t(n, mu=0.0, seed=int(rg.integers(1e9)))["ret"].to_numpy(),
    }
    tam = {
        k: np.mean([stats.ttest_1samp(g(ne.value), 0).pvalue < 0.05 for _ in range(reps.value)])
        for k, g in gens.items()
    }
    mo.ui.table(pd.Series(tam, name="tasa de rechazo (nominal 0.05)").round(3).reset_index(), selection=None)
    return gens, rg, tam


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Contrasta $H_0:\sigma_1^2=\sigma_2^2$ antes y después de una fecha con la prueba F y con Levene (`stats.levene`). ¿Coinciden con datos GARCH-$t$? ¿Por qué?
        """
    )
    return


if __name__ == "__main__":
    app.run()
