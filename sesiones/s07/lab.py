import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    return mo, np, plt, stats


@app.cell
def _(mo):
    mo.md(r"# S07 · TCL, normal y log-normal")
    return


@app.cell
def _(mo):
    fuente = mo.ui.dropdown(["exponencial", "bernoulli(0.1)", "t(5)", "t(3)", "t(2)"], value="exponencial", label="fuente")
    n = mo.ui.slider(1, 500, value=30, step=1, label="n por media")
    mo.hstack([fuente, n], justify="start")
    return fuente, n


@app.cell
def _(fuente, n, np, plt, stats):
    rng = np.random.default_rng(1)
    gens = {
        "exponencial": (lambda s: rng.exponential(1, s), 1.0, 1.0),
        "bernoulli(0.1)": (lambda s: (rng.random(s) < 0.1).astype(float), 0.1, 0.3),
        "t(5)": (lambda s: rng.standard_t(5, s), 0.0, (5 / 3) ** 0.5),
        "t(3)": (lambda s: rng.standard_t(3, s), 0.0, 3**0.5),
        "t(2)": (lambda s: rng.standard_t(2, s), 0.0, float("nan")),  # varianza infinita
    }
    g, mu, sd = gens[fuente.value]
    if sd != sd:  # nan → estandariza con sd muestral
        sd = 1.0
    zz = (g((20_000, n.value)).mean(1) - mu) / (sd / n.value**0.5)
    zgrid = np.linspace(-4, 4, 300)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].hist(zz, bins=80, range=(-5, 5), density=True, alpha=0.5)
    ax[0].plot(zgrid, stats.norm.pdf(zgrid), "k")
    ax[0].set_title(f"media estandarizada ({fuente.value}, n={n.value})")
    stats.probplot(zz, dist="norm", plot=ax[1])
    ax[1].set_title("QQ vs normal")
    fig.tight_layout()
    fig
    return ax, fig, g, gens, mu, rng, sd, zgrid, zz


@app.cell
def _(mo):
    mo.md(
        "**Observa:** con `exponencial` la asimetría desaparece al subir *n*; con `t(3)` la cola persiste; "
        "con `t(2)` (varianza infinita) el TCL clásico **no** aplica."
    )
    return


@app.cell
def _(mo):
    mo.md(r"## GBM y log-normalidad")
    return


@app.cell
def _(mo):
    mu_a = mo.ui.slider(-0.05, 0.20, value=0.08, step=0.01, label="μ anual")
    sg_a = mo.ui.slider(0.05, 0.80, value=0.20, step=0.05, label="σ anual")
    T = mo.ui.slider(0.25, 5, value=1, step=0.25, label="T (años)")
    mo.hstack([mu_a, sg_a, T], justify="start")
    return T, mu_a, sg_a


@app.cell
def _(T, mo, mu_a, np, sg_a, stats):
    Z = np.random.default_rng(3).standard_normal(500_000)
    PT = 100 * np.exp((mu_a.value - 0.5 * sg_a.value**2) * T.value + sg_a.value * np.sqrt(T.value) * Z)
    teo = 100 * np.exp(mu_a.value * T.value)
    sw = stats.skew(np.log(PT))
    mo.md(
        f"$E[P_T]$ simulado **{PT.mean():.2f}** vs. teórico **{teo:.2f}** · mediana **{np.median(PT):.2f}** · "
        f"asimetría de $\\ln P_T$ ≈ {sw:.3f} (≈ 0)"
    )
    return PT, Z, sw, teo


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Estima por simulación $P(P_T<P_0)$ y compárala con $\Phi\!\left(-\dfrac{(\mu-\sigma^2/2)T}{\sigma\sqrt T}\right)$.
        """
    )
    return


if __name__ == "__main__":
    app.run()
