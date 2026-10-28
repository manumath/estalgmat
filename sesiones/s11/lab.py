import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    from estalgmat import retornos_garch_t

    return mo, np, pd, plt, retornos_garch_t, stats


@app.cell
def _(mo):
    mo.md(r"# S11 · Intervalos de confianza y bootstrap")
    return


@app.cell
def _(mo):
    fuente = mo.ui.dropdown(["normal", "t(3)", "lognormal(1)"], value="t(3)", label="distribución")
    n = mo.ui.slider(10, 1000, value=30, step=10, label="n")
    nivel = mo.ui.slider(0.80, 0.99, value=0.95, step=0.01, label="nivel")
    mo.hstack([fuente, n, nivel], justify="start")
    return fuente, n, nivel


@app.cell
def _(fuente, mo, n, nivel, np, stats):
    rng = np.random.default_rng(11)
    gens = {
        "normal": (lambda s: rng.normal(0, 1, s), 0.0),
        "t(3)": (lambda s: rng.standard_t(3, s), 0.0),
        "lognormal(1)": (lambda s: rng.lognormal(0, 1, s), np.exp(0.5)),
    }
    g, mu = gens[fuente.value]
    x = g((4000, n.value))
    m = x.mean(1)
    se = x.std(1, ddof=1) / np.sqrt(n.value)
    q = stats.t.ppf(1 - (1 - nivel.value) / 2, n.value - 1)
    cubre = (m - q * se <= mu) & (mu <= m + q * se)
    mo.md(f"**Cobertura empírica** del IC t al {nivel.value:.0%}: **{cubre.mean():.3f}** (nominal {nivel.value:.2f})")
    return cubre, g, gens, m, mu, q, rng, se, x


@app.cell
def _(mo):
    mo.md(r"## Bootstrap del ratio de Sharpe")
    return


@app.cell
def _(mo):
    B = mo.ui.slider(500, 5000, value=2000, step=500, label="B")
    mu_ret = mo.ui.slider(0.0, 0.1, value=0.05, step=0.01, label="μ diaria (%)")
    mo.hstack([B, mu_ret], justify="start")
    return B, mu_ret


@app.cell
def _(B, mo, mu_ret, np, pd, retornos_garch_t, stats):
    r = retornos_garch_t(1500, seed=2026, mu=mu_ret.value)["ret"].to_numpy()
    sharpe = lambda v: v.mean() / v.std(ddof=1)
    sr = sharpe(r)
    ee = np.sqrt((1 + 0.5 * sr**2) / len(r))
    bs = stats.bootstrap((r,), sharpe, n_resamples=B.value, method="percentile", random_state=1)
    mo.ui.table(
        pd.DataFrame(
            {
                "Sharpe": [sr, sr],
                "IC inf.": [sr - 1.96 * ee, bs.confidence_interval.low],
                "IC sup.": [sr + 1.96 * ee, bs.confidence_interval.high],
            },
            index=["analítico (iid normal)", "bootstrap iid"],
        )
        .round(4)
        .reset_index(),
        selection=None,
    )
    return bs, ee, r, sharpe, sr


@app.cell
def _(mo):
    mo.md(r"## VaR 1 %: bootstrap por bloques")
    return


@app.cell
def _(mo):
    largo = mo.ui.slider(1, 40, value=11, step=1, label="longitud de bloque ℓ")
    largo
    return (largo,)


@app.cell
def _(largo, mo, np, plt, r):
    rg = np.random.default_rng(3)
    var99 = lambda v: -np.quantile(v, 0.01)

    def boot_bloques(x, stat, B=1500, l=1):
        n_ = len(x)
        k = -(-n_ // l)
        return np.array(
            [stat(np.concatenate([x[i : i + l] for i in rg.integers(0, n_ - l + 1, k)])[:n_]) for _ in range(B)]
        )

    b = boot_bloques(r, var99, l=largo.value)
    lo, hi = np.percentile(b, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.hist(b, bins=50, alpha=0.6)
    ax.axvline(var99(r), c="k")
    ax.set_title(f"VaR 1 % = {var99(r):.3f} · IC95 bootstrap [{lo:.3f}, {hi:.3f}]")
    mo.vstack([fig, mo.md("Con ℓ = 1 se obtiene el bootstrap iid. ¿Cambia el IC al aumentar ℓ?")])
    return ax, b, boot_bloques, fig, hi, lo, rg, var99


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Calcula por bootstrap el IC del 95 % de la curtosis en exceso y explica por qué es tan ancho con colas pesadas.
        """
    )
    return


if __name__ == "__main__":
    app.run()
