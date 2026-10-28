import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    from estalgmat import curva_tasas, retornos_factor

    return PCA, curva_tasas, mo, np, pd, plt, retornos_factor


@app.cell
def _(mo):
    mo.md(r"# S18 · SVD y PCA de la curva de rendimientos")
    return


@app.cell
def _(curva_tasas, np):
    curva = curva_tasas(1500, seed=11)
    dY = curva.diff().dropna()
    Xc = dY.to_numpy() - dY.to_numpy().mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    ve = s**2 / np.sum(s**2)
    return U, Vt, Xc, curva, dY, s, ve


@app.cell
def _(mo):
    k = mo.ui.slider(1, 8, value=3, step=1, label="rango k de la reconstrucción")
    k
    return (k,)


@app.cell
def _(U, Vt, Xc, curva, k, mo, np, plt, s, ve):
    Xk = (U[:, : k.value] * s[: k.value]) @ Vt[: k.value]
    err = np.linalg.norm(Xc - Xk, "fro")
    teo = np.sqrt(np.sum(s[k.value :] ** 2))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].bar(range(1, 9), 100 * ve)
    ax[0].plot(range(1, 9), 100 * np.cumsum(ve), "r-o", ms=3)
    ax[0].set_xlabel("componente")
    ax[0].set_ylabel("% varianza")
    for i, nom in enumerate(("PC1", "PC2", "PC3")):
        ax[1].plot(curva.columns, Vt[i], "o-", label=nom)
    ax[1].axhline(0, c="k", lw=0.5)
    ax[1].legend()
    ax[1].set_title("cargas")
    fig.tight_layout()
    mo.vstack(
        [
            fig,
            mo.md(
                f"Con k = **{k.value}**: varianza capturada = **{100 * ve[: k.value].sum():.2f} %** · "
                f"error de Frobenius = {err:.4f} · teórico √Σσᵢ² = {teo:.4f}"
            ),
        ]
    )
    return Xk, ax, err, fig, teo


@app.cell
def _(PCA, Vt, dY, mo, np, ve):
    pca = PCA().fit(dY.to_numpy())
    ok1 = np.allclose(pca.explained_variance_ratio_, ve)
    ok2 = all(np.allclose(np.abs(pca.components_[i]), np.abs(Vt[i])) for i in range(3))
    mo.md(f"**Comprobación vs. scikit-learn** — varianza explicada: {'✔' if ok1 else '✘'} · cargas (salvo signo): {'✔' if ok2 else '✘'}")
    return ok1, ok2, pca


@app.cell
def _(mo):
    mo.md(r"## PCA de retornos de activos")
    return


@app.cell
def _(mo):
    p = mo.ui.slider(3, 15, value=8, step=1, label="activos")
    p
    return (p,)


@app.cell
def _(mo, np, p, pd, retornos_factor):
    df = retornos_factor(2000, p=p.value, seed=5)
    Z = df.drop(columns="mercado")
    Zs = (Z - Z.mean()) / Z.std()
    U2, s2, Vt2 = np.linalg.svd(Zs.to_numpy(), full_matrices=False)
    pc1 = Zs.to_numpy() @ Vt2[0]
    mo.md(
        f"|corr(PC1, mercado)| = **{abs(np.corrcoef(pc1, df['mercado'])[0, 1]):.3f}** · "
        f"% varianza PC1 = **{100 * s2[0] ** 2 / np.sum(s2**2):.1f}**"
    )
    return U2, Vt2, Z, Zs, df, pc1, s2


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Repite el PCA de la curva en dos submuestras (primera y segunda mitad). ¿Son estables las cargas de PC1–PC3? ¿Y la varianza explicada?
        """
    )
    return


if __name__ == "__main__":
    app.run()
