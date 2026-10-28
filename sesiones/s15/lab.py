import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from scipy import linalg

    return linalg, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"# S15 · Sistemas lineales, rango y condicionamiento")
    return


@app.cell
def _(mo):
    mo.md(r"## 1. Rouché–Frobenius")
    return


@app.cell
def _(mo):
    b3 = mo.ui.slider(6, 10, value=7, step=0.5, label="b₃ en (4, 11, b₃)")
    b3
    return (b3,)


@app.cell
def _(b3, mo, np):
    A = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 3.0], [1.0, 2.0, 2.0]])
    b = np.array([4.0, 11.0, b3.value])
    rA = np.linalg.matrix_rank(A)
    rAb = np.linalg.matrix_rank(np.column_stack([A, b]))
    veredicto = "**inconsistente** (sin solución)" if rA != rAb else ("única" if rA == 3 else "**infinitas** soluciones")
    mo.md(f"rango(A) = {rA} · rango([A|b]) = {rAb} → solución: {veredicto}")
    return A, b, rA, rAb, veredicto


@app.cell
def _(mo):
    mo.md(r"## 2. Matriz de Hilbert: pérdida de cifras")
    return


@app.cell
def _(mo):
    n = mo.ui.slider(2, 14, value=8, step=1, label="orden n")
    n
    return (n,)


@app.cell
def _(linalg, mo, n, np):
    H = linalg.hilbert(n.value)
    xv = np.ones(n.value)
    bb = H @ xv
    e_solve = np.linalg.norm(np.linalg.solve(H, bb) - xv)
    e_inv = np.linalg.norm(np.linalg.inv(H) @ bb - xv)
    mo.md(
        f"κ(H) = **{np.linalg.cond(H):.2e}** · error con `solve` = {e_solve:.1e} · error con `inv` = {e_inv:.1e}  \n"
        f"cifras decimales perdidas ≈ log₁₀κ = **{np.log10(np.linalg.cond(H)):.1f}**"
    )
    return H, bb, e_inv, e_solve, xv


@app.cell
def _(mo):
    mo.md(r"## 3. Colinealidad en una regresión")
    return


@app.cell
def _(mo):
    rho = mo.ui.slider(0.0, 0.99999, value=0.99, step=0.001, label="correlación entre regresores ρ")
    rho
    return (rho,)


@app.cell
def _(mo, np, pd, rho):
    rng = np.random.default_rng(15)
    x1 = rng.standard_normal(200)
    x2 = rho.value * x1 + np.sqrt(1 - rho.value**2) * rng.standard_normal(200)
    X = np.column_stack([np.ones(200), x1, x2])
    y = 1 + x1 + x2 + rng.standard_normal(200)
    b_inv = np.linalg.inv(X.T @ X) @ X.T @ y
    b_ls = np.linalg.lstsq(X, y, rcond=None)[0]
    tab = pd.DataFrame({"inv(X'X)X'y": b_inv, "lstsq": b_ls}, index=["β₀", "β₁", "β₂"]).round(3)
    mo.vstack(
        [
            mo.md(f"κ(X'X) = **{np.linalg.cond(X.T @ X):.2e}** · Var(β̂₁) ∝ {np.linalg.inv(X.T @ X)[1, 1]:.3f} (verdaderos β = 1, 1, 1)"),
            mo.ui.table(tab.reset_index(), selection=None),
        ]
    )
    return X, b_inv, b_ls, rng, tab, x1, x2, y


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Implementa `rref(M)` (Gauss–Jordan con pivoteo parcial) y verifica que el rango que obtienes coincide con `np.linalg.matrix_rank` en 5 matrices aleatorias de rango incompleto (por ejemplo, `A @ B` con `A: 5×2` y `B: 2×5`).
        """
    )
    return


if __name__ == "__main__":
    app.run()
