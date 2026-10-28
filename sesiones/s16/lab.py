import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy import stats

    from estalgmat import retornos_factor

    return mo, np, pd, retornos_factor, sm, stats


@app.cell
def _(mo):
    mo.md(r"# S16 · MCO en forma matricial: CAPM")
    return


@app.cell
def _(mo):
    activo = mo.ui.dropdown([f"A{i}" for i in range(1, 6)], value="A3", label="activo")
    n = mo.ui.slider(100, 4000, value=1500, step=100, label="n")
    mo.hstack([activo, n], justify="start")
    return activo, n


@app.cell
def _(activo, n, np, retornos_factor):
    df = retornos_factor(n.value, p=5, seed=3)
    y = df[activo.value].to_numpy()
    X = np.column_stack([np.ones(len(df)), df["mercado"].to_numpy()])
    nn, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    s2 = e @ e / (nn - k)
    V = s2 * XtXi
    se = np.sqrt(np.diag(V))
    return V, X, XtXi, b, df, e, k, nn, s2, se, y


@app.cell
def _(b, df, mo, pd, se, sm, X, y, activo):
    fit = sm.OLS(y, X).fit()
    beta_v = df.attrs["betas"][int(activo.value[1:]) - 1]
    tabla = pd.DataFrame(
        {"β̂ (a mano)": b, "EE": se, "t": b / se, "β̂ statsmodels": fit.params, "EE statsmodels": fit.bse},
        index=["α", "β"],
    ).round(4)
    mo.vstack([mo.md(f"β verdadero = **{beta_v:.2f}**"), mo.ui.table(tabla.reset_index(), selection=None)])
    return beta_v, fit, tabla


@app.cell
def _(mo):
    mo.md(r"## Contraste lineal $H_0: R\beta=r$")
    return


@app.cell
def _(mo):
    a0 = mo.ui.number(-1, 2, value=0.0, step=0.1, label="α₀")
    b0 = mo.ui.number(-1, 3, value=1.0, step=0.1, label="β₀")
    mo.hstack([a0, b0], justify="start")
    return a0, b0


@app.cell
def _(V, a0, b, b0, mo, nn, k, np, stats):
    R, r = np.eye(2), np.array([a0.value, b0.value])
    d = R @ b - r
    F = d @ np.linalg.inv(R @ V @ R.T) @ d / 2
    p = stats.f.sf(F, 2, nn - k)
    mo.md(f"F = **{F:.3f}** · p-valor = **{p:.4f}** → {'rechaza' if p < 0.05 else 'no rechaza'} H₀: (α, β) = ({a0.value}, {b0.value})")
    return F, R, d, p, r


@app.cell
def _(X, XtXi, e, k, mo, nn, np):
    P = X @ XtXi @ X.T
    M = np.eye(nn) - P
    mo.md(
        "### Propiedades de las matrices proyección\n"
        f"- P simétrica: {'✔' if np.allclose(P, P.T) else '✘'} · idempotente: {'✔' if np.allclose(P @ P, P) else '✘'}\n"
        f"- tr(P) = {np.trace(P):.3f} (k = {k}) · tr(M) = {np.trace(M):.1f} (n−k = {nn - k})\n"
        f"- Xᵀe = 0: {'✔' if np.allclose(X.T @ e, 0, atol=1e-7) else '✘'}"
    )
    return M, P


@app.cell
def _(X, XtXi, e, k, mo, nn, np, se, sm, y):
    meat = (X * e[:, None] ** 2).T @ X
    se_hc = np.sqrt(np.diag(XtXi @ meat @ XtXi * nn / (nn - k)))
    se_sm = sm.OLS(y, X).fit(cov_type="HC1").bse
    mo.md(
        f"**EE de β** — clásico: {se[1]:.4f} · HC1 (a mano): {se_hc[1]:.4f} · HC1 (statsmodels): {se_sm[1]:.4f}"
    )
    return meat, se_hc, se_sm


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Añade un segundo factor (por ejemplo, el cuadrado del mercado) a $X$, recalcula $\hat\beta$ y contrasta $H_0:\beta_2=0$ con la fórmula matricial de F. Calcula el VIF de ambos regresores.
        """
    )
    return


if __name__ == "__main__":
    app.run()
