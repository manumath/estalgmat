import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    from statsmodels.tsa.stattools import acf

    return acf, mo, np, pd, plt, stats


@app.cell
def _(mo):
    mo.md(
        r"""
        # S09 · Colas pesadas y asimetría

        Genera retornos $r_t=\sigma\,z_t$ con $z_t\sim t_\nu$ (estandarizada a varianza 1)
        y diagnostica su no normalidad. Mueve **ν** y **n**.
        """
    )
    return


@app.cell
def _(mo):
    nu = mo.ui.slider(2.5, 50, value=5, step=0.5, label="ν (grados de libertad)")
    n = mo.ui.slider(250, 20000, value=5000, step=250, label="n (observaciones)")
    semilla = mo.ui.number(0, 9999, value=2026, label="semilla")
    mo.hstack([nu, n, semilla], justify="start")
    return n, nu, semilla


@app.cell
def _(n, np, nu, semilla):
    rng = np.random.default_rng(semilla.value)
    # t estandarizada: varianza 1 cuando ν > 2
    r = rng.standard_t(nu.value, size=n.value) / np.sqrt(nu.value / (nu.value - 2))
    return r, rng


@app.cell
def _(mo, r, stats):
    jb, p_jb = stats.jarque_bera(r)
    mo.md(
        f"""
        | Estadístico | Valor |
        |---|---|
        | Asimetría | {stats.skew(r):.3f} |
        | Curtosis en exceso (muestral) | {stats.kurtosis(r):.3f} |
        | Jarque–Bera | {jb:,.1f} |
        | p-valor | {p_jb:.4g} |
        """
    )
    return jb, p_jb


@app.cell
def _(mo, nu):
    teorica = 6 / (nu.value - 4) if nu.value > 4 else float("inf")
    mo.md(
        f"**Curtosis en exceso teórica** de $t_{{{nu.value:g}}}$: "
        f"`{teorica:.3f}` {'(no existe el momento de orden 4)' if nu.value <= 4 else ''}"
    )
    return (teorica,)


@app.cell
def _(np, nu, plt, r, stats):
    x = np.linspace(-8, 8, 800)
    esc = np.sqrt((nu.value - 2) / nu.value)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(r, bins=120, density=True, alpha=0.4, range=(-8, 8))
    ax[0].plot(x, stats.norm.pdf(x), label="N(0,1)")
    ax[0].plot(x, stats.t.pdf(x / esc, nu.value) / esc, label=f"t({nu.value:g}) estandarizada")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-5, 1)
    ax[0].legend()
    ax[0].set_title("Densidad (escala log)")
    stats.probplot(r, dist="norm", plot=ax[1])
    ax[1].set_title("QQ-plot vs normal")
    fig.tight_layout()
    fig
    return ax, esc, fig, x


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Probabilidades de cola y VaR

        Comparamos $P(r<-k)$ bajo la normal y bajo la $t$ estandarizada verdadera.
        """
    )
    return


@app.cell
def _(mo, nu, pd, r, stats):
    esc2 = (nu.value - 2) / nu.value
    filas = []
    for k in (2, 3, 4, 5):
        pn = stats.norm.cdf(-k)
        pt = stats.t.cdf(-k / esc2**0.5, nu.value)
        filas.append({"k": k, "normal": pn, "t verdadera": pt,
                      "empírica": (r < -k).mean(), "razón t/normal": pt / pn})
    tabla = pd.DataFrame(filas).set_index("k")
    var_n = -stats.norm.ppf(0.01)
    var_t = -stats.t.ppf(0.01, nu.value) * esc2**0.5
    mo.vstack([
        mo.ui.table(tabla.reset_index().round(6), selection=None),
        mo.md(f"**VaR 99 %** — normal: `{var_n:.3f}` · t verdadera: `{var_t:.3f}` · empírico: `{-pd.Series(r).quantile(0.01):.3f}`"),
    ])
    return esc2, filas, pn, pt, tabla, var_n, var_t


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicios

        1. ¿Para qué $\nu$ la razón $P_t/P_N$ en $k=4$ deja de ser mayor que 2?
        2. Fija $\nu=3$ y varía $n$: ¿es estable la curtosis muestral? ¿por qué?
        3. Con $\nu=30$, ¿rechaza JB la normalidad con $n=20\,000$? Interpreta.
        4. (Reto) Repite con retornos GARCH-$t$ de `estalgmat.retornos_garch_t` y compara la ACF de $r$ y de $r^2$.
        """
    )
    return


if __name__ == "__main__":
    app.run()
