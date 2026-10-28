import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import acf, adfuller

    return acf, adfuller, mo, np, plt, sm


@app.cell
def _(mo):
    mo.md(r"# S20 · Puente al Módulo 2: procesos estocásticos")
    return


@app.cell
def _(mo):
    phi = mo.ui.slider(-0.99, 1.02, value=0.8, step=0.01, label="φ del AR(1)")
    n = mo.ui.slider(100, 3000, value=500, step=100, label="n")
    mo.hstack([phi, n], justify="start")
    return n, phi


@app.cell
def _(acf, adfuller, mo, n, np, phi, plt):
    e = np.random.default_rng(20).standard_normal(n.value)
    x = np.zeros(n.value)
    for t in range(1, n.value):
        x[t] = phi.value * x[t - 1] + e[t]
    a = acf(x, nlags=30)[1:]
    teo = phi.value ** np.arange(1, 31)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(x, lw=0.8)
    ax[0].set_title(f"AR(1), φ = {phi.value:.2f}")
    ax[1].bar(range(1, 31), a, width=0.6, label="ACF muestral")
    ax[1].plot(range(1, 31), teo, "r.-", label="φᵏ teórica")
    ax[1].legend()
    fig.tight_layout()
    stat, pv = adfuller(x, autolag="AIC", result_object=False)[:2]
    mo.vstack(
        [
            fig,
            mo.md(
                f"ADF: estadístico = **{stat:.2f}**, p = **{pv:.3f}** → "
                f"{'rechaza' if pv < 0.05 else 'no rechaza'} raíz unitaria. "
                f"{'Estacionario (|φ|<1)' if abs(phi.value) < 1 else 'NO estacionario (|φ|≥1)'}"
            ),
        ]
    )
    return a, ax, e, fig, pv, stat, teo, x


@app.cell
def _(mo):
    mo.md(r"## Regresión espuria")
    return


@app.cell
def _(mo):
    reps = mo.ui.slider(100, 1000, value=300, step=100, label="pares")
    nn = mo.ui.slider(50, 1000, value=250, step=50, label="n")
    mo.hstack([reps, nn], justify="start")
    return nn, reps


@app.cell
def _(mo, nn, np, reps, sm):
    rng = np.random.default_rng(2)
    res = []
    for _ in range(reps.value):
        u, v = np.cumsum(rng.standard_normal(nn.value)), np.cumsum(rng.standard_normal(nn.value))
        f = sm.OLS(u, sm.add_constant(v)).fit()
        res.append((f.rsquared, abs(f.tvalues[1])))
    res = np.array(res)
    # misma regresión con variables estacionarias (ruido blanco)
    res2 = []
    for _ in range(reps.value):
        u, v = rng.standard_normal(nn.value), rng.standard_normal(nn.value)
        f = sm.OLS(u, sm.add_constant(v)).fit()
        res2.append(abs(f.tvalues[1]))
    mo.md(
        f"**Caminatas independientes:** R² medio = {res[:, 0].mean():.2f} · rechazos con |t|>2 = **{100 * np.mean(res[:, 1] > 2):.0f} %**  \n"
        f"**Ruido blanco independiente:** rechazos con |t|>2 = {100 * np.mean(np.array(res2) > 2):.0f} % (≈ 5 % nominal)"
    )
    return f, res, res2, rng


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Regresa las **primeras diferencias** de las dos caminatas independientes. ¿Qué ocurre con el R² y con la tasa de rechazo? ¿Por qué diferenciar resuelve el problema aquí?
        """
    )
    return


if __name__ == "__main__":
    app.run()
