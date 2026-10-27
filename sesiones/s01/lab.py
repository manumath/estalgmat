import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        # S01 · Laboratorio: verificación del entorno

        Este cuaderno es **reactivo**: al cambiar un control, todas las celdas que
        dependen de él se recalculan solas.

        ## 1. Verificación del entorno
        """
    )
    return


@app.cell
def _(mo):
    import sys

    import matplotlib
    import scipy
    import statsmodels

    import numpy

    versiones = {
        "Python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
    }
    mo.md("\n".join(f"- **{k}**: `{v}`" for k, v in versiones.items()))
    return matplotlib, numpy, scipy, statsmodels, sys, versiones


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Una caminata aleatoria con deriva

        $$P_t = P_{t-1}\,e^{\mu+\sigma\varepsilon_t},\qquad \varepsilon_t\sim N(0,1)$$

        Mueve los controles y observa cómo cambian la trayectoria y el resumen.
        """
    )
    return


@app.cell
def _(mo):
    mu = mo.ui.slider(-0.05, 0.10, value=0.03, step=0.01, label="μ diaria (%)")
    sigma = mo.ui.slider(0.2, 3.0, value=1.0, step=0.1, label="σ diaria (%)")
    semilla = mo.ui.number(0, 9999, value=2026, label="semilla")
    mo.hstack([mu, sigma, semilla], justify="start")
    return mu, semilla, sigma


@app.cell
def _(mu, np, plt, semilla, sigma):
    rng = np.random.default_rng(semilla.value)
    n = 1000
    eps = rng.standard_normal(n)
    ret = mu.value + sigma.value * eps                    # retornos log en %
    precio = 100 * np.exp(np.cumsum(ret) / 100)

    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(precio, lw=1)
    ax[0].set_ylabel("Precio")
    ax[1].plot(ret, lw=0.5)
    ax[1].set_ylabel("Retorno (%)")
    fig.tight_layout()
    fig
    return ax, eps, fig, n, precio, ret, rng


@app.cell
def _(mo, precio, ret):
    mo.md(
        f"""
        - Retorno medio diario: **{ret.mean():.3f} %** · Desv. estándar: **{ret.std():.3f} %**
        - Precio final: **{precio[-1]:.1f}** (inicio = 100)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. Ejercicio final

        Completa la función `max_drawdown(precio)` que devuelve la **máxima caída**
        desde un máximo previo (como proporción). Las pruebas de abajo te dirán si
        está bien.

        $$\mathrm{MDD}=\max_t\left(1-\frac{P_t}{\max_{s\le t}P_s}\right)$$
        """
    )
    return


@app.cell
def _(np):
    def max_drawdown(precio):
        # TODO: reemplaza la línea siguiente
        pico = np.maximum.accumulate(precio)
        return float(np.max(1 - precio / pico))

    return (max_drawdown,)


@app.cell
def _(max_drawdown, mo, np):
    casos = [
        (np.array([100, 120, 90, 110.0]), 0.25),
        (np.array([100, 101, 102, 103.0]), 0.0),
        (np.array([100, 50, 100, 25.0]), 0.75),
    ]
    resultados = [abs(max_drawdown(x) - esperado) < 1e-9 for x, esperado in casos]
    mo.md(
        "### Pruebas\n"
        + "\n".join(f"- caso {i + 1}: {'✔' if ok else '✘'}" for i, ok in enumerate(resultados))
    )
    return casos, resultados


if __name__ == "__main__":
    app.run()
