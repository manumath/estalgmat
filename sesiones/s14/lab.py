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
    mo.md(r"# S14 · Matrices y covarianzas")
    return


@app.cell
def _(mo):
    p = mo.ui.slider(2, 10, value=4, step=1, label="activos p")
    n = mo.ui.slider(30, 2000, value=1000, step=10, label="observaciones n")
    mo.hstack([p, n], justify="start")
    return n, p


@app.cell
def _(n, np, p, retornos_factor):
    df = retornos_factor(n.value, p=p.value, seed=3)
    X = df.drop(columns="mercado").to_numpy()
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / (len(X) - 1)
    d = 1 / np.sqrt(np.diag(S))
    R = S * np.outer(d, d)
    return R, S, X, df


@app.cell
def _(R, df, np, p, plt):
    fig, ax = plt.subplots(figsize=(4.8, 4))
    im = ax.imshow(R, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(p.value), df.columns[1:])
    ax.set_yticks(range(p.value), df.columns[1:])
    fig.colorbar(im)
    fig
    return ax, fig, im


@app.cell
def _(S, X, mo, n, np, p):
    w = np.ones(p.value) / p.value
    v_forma = w @ S @ w
    v_dir = np.var(X @ w, ddof=1)
    v_ind = np.sum(w**2 * np.diag(S))
    mo.md(
        f"Cartera igualmente ponderada: **w'Σw = {v_forma:.4f}** · Var(Xw) = {v_dir:.4f} · "
        f"sin covarianzas = {v_ind:.4f}  \nRango de Σ̂ = **{np.linalg.matrix_rank(S)}** de {p.value} "
        f"(n = {n.value}) · κ(Σ̂) = {np.linalg.cond(S):.1e}"
    )
    return v_dir, v_forma, v_ind, w


@app.cell
def _(mo):
    mo.md("**Prueba:** baja *n* hasta 10 con *p* = 10: ¿qué le pasa al rango de Σ̂ y a su condicionamiento?")
    return


@app.cell
def _(mo):
    mo.md(r"## Propiedades de matrices (comprobaciones)")
    return


@app.cell
def _(mo, np):
    rng = np.random.default_rng(1)
    A, B, C = (rng.standard_normal((3, 3)) for _ in range(3))
    checks = [
        ("(AB)ᵀ = BᵀAᵀ", np.allclose((A @ B).T, B.T @ A.T)),
        ("(AB)⁻¹ = B⁻¹A⁻¹", np.allclose(np.linalg.inv(A @ B), np.linalg.inv(B) @ np.linalg.inv(A))),
        ("tr(AB) = tr(BA)", np.isclose(np.trace(A @ B), np.trace(B @ A))),
        ("det(AB) = det A · det B", np.isclose(np.linalg.det(A @ B), np.linalg.det(A) * np.linalg.det(B))),
        ("A(B+C) = AB + AC", np.allclose(A @ (B + C), A @ B + A @ C)),
        ("AB = BA (falso en general)", np.allclose(A @ B, B @ A)),
    ]
    mo.md("\n".join(f"- {n}: {'✔' if ok else '✘'}" for n, ok in checks))
    return A, B, C, checks, rng


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Escribe `varianza_portafolio(w, Sigma)` que valide que `w` suma 1 y que `Sigma` es simétrica, y devuelva $w'\Sigma w$. Pruébala con `w` que no suma 1 para ver la excepción.
        """
    )
    return


if __name__ == "__main__":
    app.run()
