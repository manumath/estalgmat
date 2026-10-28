import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from estalgmat import retornos_factor

    return mo, np, pd, plt, retornos_factor


@app.cell
def _(mo):
    mo.md(r"# S17 · Autovalores, Cholesky y estabilidad")
    return


@app.cell
def _(mo):
    p1 = mo.ui.slider(-1.5, 1.5, value=0.5, step=0.05, label="φ₁")
    p2 = mo.ui.slider(-1.0, 1.0, value=0.3, step=0.05, label="φ₂")
    mo.hstack([p1, p2], justify="start")
    return p1, p2


@app.cell
def _(mo, np, p1, p2, plt):
    F = np.array([[p1.value, p2.value], [1.0, 0.0]])
    ev = np.linalg.eigvals(F)
    mx = np.max(np.abs(ev))
    t = np.linspace(0, 2 * np.pi, 300)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot(np.cos(t), np.sin(t), "k", lw=0.8)
    ax.scatter(ev.real, ev.imag, c="C0" if mx < 1 else "C3", s=60)
    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(alpha=0.3)
    mo.vstack(
        [
            fig,
            mo.md(f"máx|λ| = **{mx:.3f}** → AR(2) **{'estacionario' if mx < 1 else 'NO estacionario'}**"),
        ]
    )
    return F, ax, ev, fig, mx, t


@app.cell
def _(mo):
    mo.md(r"## Cholesky: de $z\sim N(0,I)$ a $x\sim N(0,\Sigma)$")
    return


@app.cell
def _(mo):
    rho = mo.ui.slider(-0.95, 0.95, value=0.7, step=0.05, label="ρ")
    s2 = mo.ui.slider(0.5, 3.0, value=1.5, step=0.1, label="σ₂")
    mo.hstack([rho, s2], justify="start")
    return rho, s2


@app.cell
def _(mo, np, plt, rho, s2):
    Sig = np.array([[1.0, rho.value * s2.value], [rho.value * s2.value, s2.value**2]])
    L = np.linalg.cholesky(Sig)
    Z = np.random.default_rng(1).standard_normal((3000, 2))
    Xs = Z @ L.T
    fig2, ax2 = plt.subplots(1, 2, figsize=(8, 3.6))
    ax2[0].scatter(*Z.T, s=2, alpha=0.4)
    ax2[0].set_title("z ~ N(0, I)")
    ax2[1].scatter(*Xs.T, s=2, alpha=0.4)
    ax2[1].set_title("x = Lz ~ N(0, Σ)")
    for a in ax2:
        a.set_aspect("equal")
        a.set_xlim(-6, 6)
        a.set_ylim(-6, 6)
    fig2.tight_layout()
    mo.vstack(
        [
            fig2,
            mo.md(f"L = {np.round(L, 3).tolist()} · Cov(x) empírica = {np.round(np.cov(Xs.T), 2).tolist()}"),
        ]
    )
    return L, Sig, Xs, Z, ax2, fig2


@app.cell
def _(mo):
    mo.md(r"## Covarianza de activos: definitud y autovalores")
    return


@app.cell
def _(mo, np, pd, retornos_factor):
    X = retornos_factor(2000, p=4, seed=3).drop(columns="mercado").to_numpy()
    S = np.cov(X, rowvar=False)
    w, Q = np.linalg.eigh(S)
    mo.vstack(
        [
            mo.ui.table(
                pd.DataFrame({"λ": w[::-1], "% varianza": 100 * w[::-1] / w.sum()}, index=[f"λ{i}" for i in range(1, 5)])
                .round(3)
                .reset_index(),
                selection=None,
            ),
            mo.md(
                f"Σ = QΛQᵀ: {'✔' if np.allclose(Q @ np.diag(w) @ Q.T, S) else '✘'} · "
                f"todos λ>0 ⇒ definida positiva: {'✔' if (w > 0).all() else '✘'}"
            ),
        ]
    )
    return Q, S, X, w


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Construye la matriz compañera de un VAR(1) de 3 variables con coeficientes de tu elección. Comprueba su estacionariedad y simula 300 observaciones con `Σ` y Cholesky para ver la trayectoria.
        """
    )
    return


if __name__ == "__main__":
    app.run()
