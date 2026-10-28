import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from estalgmat import retornos_garch_t, serie_macro

    return mo, np, pd, plt, retornos_garch_t, serie_macro


@app.cell
def _(mo):
    mo.md(
        r"""
        # S03 · pandas y series de tiempo

        ## 1. Construir el panel
        `construir_panel()` devolverá retornos diarios de `k` activos y una variable macro mensual unida **sin look-ahead**.
        """
    )
    return


@app.cell
def _(pd, retornos_garch_t, serie_macro):
    def construir_panel(k=4, n=1500, semilla=100, rezago_macro=1):
        ret = pd.DataFrame(
            {f"A{i}": retornos_garch_t(n, seed=semilla + i)["ret"] for i in range(1, k + 1)}
        )
        macro = serie_macro(n // 15, inicio=str(ret.index[0].date()))
        macro = macro.shift(rezago_macro).rename("macro")
        panel = pd.merge_asof(
            ret.reset_index(), macro.reset_index(), on="fecha", direction="backward"
        ).set_index("fecha")
        return panel

    return (construir_panel,)


@app.cell
def _(mo):
    k = mo.ui.slider(2, 8, value=4, label="activos")
    ventana = mo.ui.slider(5, 126, value=21, label="ventana móvil (días)")
    mo.hstack([k, ventana], justify="start")
    return k, ventana


@app.cell
def _(construir_panel, k, np, ventana):
    panel = construir_panel(k=k.value)
    activos = [c for c in panel.columns if c.startswith("A")]
    vol = panel[activos].rolling(ventana.value).std() * np.sqrt(252)
    return activos, panel, vol


@app.cell
def _(mo, panel):
    mo.vstack([mo.md("### Primeras filas con datos macro"), mo.ui.table(panel.dropna().head(8).round(3).reset_index(), selection=None)])
    return


@app.cell
def _(plt, vol):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    vol.plot(ax=ax, lw=0.8)
    ax.set_ylabel("vol. anualizada (%)")
    fig
    return ax, fig


@app.cell
def _(mo):
    mo.md(r"## 2. Comprobaciones")
    return


@app.cell
def _(activos, mo, panel):
    sin_nan_macro = panel["macro"].dropna().index[0] > panel.index[0]
    mensual = panel[activos].resample("MS").sum()
    checks = [
        ("el índice es DatetimeIndex ordenado", panel.index.is_monotonic_increasing),
        ("macro no disponible al inicio (rezago)", sin_nan_macro),
        ("resample mensual reduce filas", len(mensual) < len(panel)),
    ]
    mo.md("\n".join(f"- {n}: {'✔' if ok else '✘'}" for n, ok in checks))
    return checks, mensual, sin_nan_macro


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Guarda el panel con `panel.to_parquet("data/panel.parquet")` (instala `pyarrow` con `uv add pyarrow`) y vuelve a leerlo comprobando que el índice conserva su tipo.
        """
    )
    return


if __name__ == "__main__":
    app.run()
