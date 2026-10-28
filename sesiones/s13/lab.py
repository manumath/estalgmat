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
    from statsmodels.stats.multitest import multipletests

    from estalgmat import retornos_garch_t

    return mo, multipletests, np, pd, plt, retornos_garch_t, stats


@app.cell
def _(mo):
    mo.md(r"# S13 · Pruebas no paramétricas y múltiples")
    return


@app.cell
def _(mo):
    n = mo.ui.slider(100, 5000, value=1000, step=100, label="n por serie")
    n
    return (n,)


@app.cell
def _(mo, n, np, pd, retornos_garch_t, stats):
    rng = np.random.default_rng(13)
    series = {
        "normal": rng.normal(0, 1, n.value),
        "t(10)": rng.standard_t(10, n.value),
        "t(4)": rng.standard_t(4, n.value),
        "GARCH-t": retornos_garch_t(n.value, seed=1)["ret"].to_numpy(),
        "lognormal": rng.lognormal(0, 0.5, n.value),
    }
    tab = pd.DataFrame(
        {
            k: {
                "Shapiro–Wilk": stats.shapiro(v).pvalue,
                "Jarque–Bera": stats.jarque_bera(v).pvalue,
                "D'Agostino": stats.normaltest(v).pvalue,
            }
            for k, v in series.items()
        }
    ).T.round(4)
    mo.vstack([mo.md("**p-valores de normalidad** (rechazo si < 0.05)"), mo.ui.table(tab.reset_index(), selection=None)])
    return rng, series, tab


@app.cell
def _(mo):
    mo.md(r"## Efectos de calendario sobre ruido puro")
    return


@app.cell
def _(mo):
    m = mo.ui.slider(1, 100, value=20, step=1, label="número de pruebas m")
    metodo = mo.ui.dropdown(["ninguna", "bonferroni", "holm", "fdr_bh"], value="ninguna", label="corrección")
    mo.hstack([m, metodo], justify="start")
    return m, metodo


@app.cell
def _(m, metodo, mo, multipletests, np, pd, stats):
    rg = np.random.default_rng(99)
    p = np.array([stats.ttest_1samp(rg.standard_normal(250), 0).pvalue for _ in range(m.value)])
    if metodo.value == "ninguna":
        rech = p < 0.05
        padj = p
    else:
        rech, padj, _, _ = multipletests(p, method=metodo.value)
    mo.vstack(
        [
            mo.md(
                f"Rechazos con la corrección **{metodo.value}**: **{int(rech.sum())}** de {m.value} "
                f"(no hay ningún efecto real). P(≥1 falso positivo sin corrección) = {1 - 0.95**m.value:.0%}"
            ),
            mo.ui.table(pd.DataFrame({"p": p, "p ajustado": padj, "rechaza": rech}).sort_values("p").head(8).round(4), selection=None),
        ]
    )
    return p, padj, rech, rg


@app.cell
def _(mo):
    mo.md(r"## Permutación: ¿cambió la volatilidad?")
    return


@app.cell
def _(mo):
    corte = mo.ui.slider(200, 800, value=500, step=50, label="día del evento")
    corte
    return (corte,)


@app.cell
def _(corte, np, plt, retornos_garch_t, stats):
    y = retornos_garch_t(1000, seed=3)["ret"].to_numpy()
    a, b = y[: corte.value], y[corte.value :]
    est = lambda u, v: np.log(u.std(ddof=1) / v.std(ddof=1))
    res = stats.permutation_test((a, b), est, permutation_type="independent", n_resamples=3000, random_state=2)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(res.null_distribution, bins=50, alpha=0.6)
    ax.axvline(res.statistic, c="r")
    ax.set_title(f"p (permutación) = {res.pvalue:.3f}")
    fig
    return a, ax, b, est, fig, res, y


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Implementa una permutación **por bloques** (bloques de 20 días) y compara el p-valor con el de la permutación simple. ¿Cuál es más confiable con agrupamiento de volatilidad?
        """
    )
    return


if __name__ == "__main__":
    app.run()
