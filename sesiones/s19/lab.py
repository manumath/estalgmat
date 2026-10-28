import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.covariance import LedoitWolf

    from estalgmat import retornos_factor

    return LedoitWolf, mo, np, pd, plt, retornos_factor


@app.cell
def _(mo):
    mo.md(
        r"""
        # S19 · Taller: cartera de mínima varianza

        Explora cómo cambian los pesos y su incertidumbre con el número de activos *p*, el tamaño muestral *n* y el método de estimación de Σ.
        """
    )
    return


@app.cell
def _(mo):
    p = mo.ui.slider(3, 30, value=5, step=1, label="activos p")
    n = mo.ui.slider(40, 4000, value=1000, step=20, label="observaciones n")
    metodo = mo.ui.dropdown(["muestral", "Ledoit–Wolf"], value="muestral", label="Σ̂")
    mo.hstack([p, n, metodo], justify="start")
    return metodo, n, p


@app.cell
def _(LedoitWolf, metodo, n, np, p, retornos_factor):
    df = retornos_factor(4000, p=p.value, seed=3)
    X = df.drop(columns="mercado").to_numpy()
    h = min(n.value, len(X) // 2)

    def cov(Z):
        return np.cov(Z, rowvar=False) if metodo.value == "muestral" else LedoitWolf().fit(Z).covariance_

    def minvar(S):
        z = np.linalg.solve(S, np.ones(len(S)))
        return z / z.sum()

    Xe, Xo = X[:h], X[len(X) // 2 :]
    S = cov(Xe)
    w = minvar(S)
    return S, Xe, Xo, cov, df, h, minvar, w, X


@app.cell
def _(Xo, h, mo, np, p, S, w):
    eq = np.ones(p.value) / p.value
    mo.md(
        f"""
        | | valor |
        |---|---|
        | n usado para estimar | {h} |
        | κ(Σ̂) | {np.linalg.cond(S):.1f} |
        | varianza fuera de muestra: 1/p | {np.var(Xo @ eq, ddof=1):.4f} |
        | varianza fuera de muestra: mín. varianza | **{np.var(Xo @ w, ddof=1):.4f}** |
        | peso mínimo / máximo | {w.min():.2f} / {w.max():.2f} |
        """
    )
    return (eq,)


@app.cell
def _(Xe, cov, minvar, np, plt, p):
    rng = np.random.default_rng(1)
    W = np.array([minvar(cov(Xe[rng.integers(0, len(Xe), len(Xe))])) for _ in range(200)])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.boxplot(W)
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel("activo")
    ax.set_ylabel("peso (bootstrap)")
    fig
    return W, ax, fig, rng


@app.cell
def _(mo):
    mo.md(
        r"""
        **Experimentos sugeridos**

        1. Con *p* = 20, baja *n* de 1000 a 60: ¿qué le pasa a la dispersión de los pesos y a la varianza fuera de muestra?
        2. Repite con **Ledoit–Wolf**: ¿cuándo ayuda más el encogimiento?
        3. ¿Con qué cociente *n/p* deja de importar la diferencia entre métodos?
        """
    )
    return


if __name__ == "__main__":
    app.run()
