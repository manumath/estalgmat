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
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf

    from estalgmat import retornos_garch_t, serie_macro

    return STL, acf, mo, np, pd, plt, retornos_garch_t, serie_macro, stats


@app.cell
def _(mo):
    mo.md("# S04 · EDA de series\n\n## 1. Serie macro: STL y atípicos")
    return


@app.cell
def _(mo):
    umbral = mo.ui.slider(2.5, 6.0, value=3.5, step=0.5, label="umbral z robusto")
    umbral
    return (umbral,)


@app.cell
def _(STL, np, pd, plt, serie_macro, umbral):
    y = serie_macro()
    res = STL(y, period=12, robust=True).fit()
    mad = np.median(np.abs(res.resid - np.median(res.resid)))
    z = pd.Series(0.6745 * (res.resid - np.median(res.resid)) / mad, index=y.index)
    marcados = z.abs() > umbral.value

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(y, lw=0.9)
    ax.scatter(y.index[marcados], y[marcados], c="r", zorder=3, label=f"atípicos ({marcados.sum()})")
    ax.legend()
    fig
    return ax, fig, mad, marcados, res, y, z


@app.cell
def _(mo):
    mo.md(r"## 2. Retornos: ¿qué muestran la ACF de $r$ y de $r^2$?")
    return


@app.cell
def _(acf, np, plt, retornos_garch_t):
    r = retornos_garch_t(3000, seed=2026)["ret"]
    nl = 25
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for a, x, t in zip(ax2, (r, r**2), ("r", "r²")):
        a.bar(range(1, nl + 1), acf(x, nlags=nl)[1:], width=0.6)
        for s in (1, -1):
            a.axhline(s * 1.96 / np.sqrt(len(x)), ls="--", c="r", lw=0.8)
        a.set_title(f"ACF de {t}")
    fig2.tight_layout()
    fig2
    return ax2, fig2, nl, r


@app.cell
def _(mo, r, stats):
    mo.md(
        f"**Asimetría** {stats.skew(r):.2f} · **curtosis exceso** {stats.kurtosis(r):.2f}. "
        "Escribe en una celda tu interpretación (3 líneas) de la ACF de $r$ vs. $r^2$."
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Crea `eda.qmd` con parámetros (`ticker`, `inicio`) y al menos 6 figuras con `#| label: fig-...` y referencias cruzadas `@fig-...`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
