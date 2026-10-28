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
    mo.md(r"# S08 · χ², t y F")
    return


@app.cell
def _(mo):
    nu = mo.ui.slider(1, 60, value=5, step=1, label="ν (g.l.)")
    nu
    return (nu,)


@app.cell
def _(mo, np, nu, plt, stats):
    x = np.linspace(-6, 6, 600)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(x, stats.norm.pdf(x), label="N(0,1)")
    ax[0].plot(x, stats.t.pdf(x, nu.value), label=f"t({nu.value})")
    ax[0].legend()
    ax[0].set_title("densidad")
    ax[1].semilogy(x, stats.norm.sf(x), label="N(0,1)")
    ax[1].semilogy(x, stats.t.sf(x, nu.value), label=f"t({nu.value})")
    ax[1].set_ylim(1e-8, 1)
    ax[1].legend()
    ax[1].set_title("cola derecha (log)")
    fig.tight_layout()
    mo.vstack(
        [
            fig,
            mo.md(
                f"$t_{{0{{,}}975}}$ = **{stats.t.ppf(0.975, nu.value):.3f}** (normal: 1.960) · "
                f"$P(|T|>3)$ = **{2 * stats.t.sf(3, nu.value):.4f}** (normal: 0.0027)"
            ),
        ]
    )
    return ax, fig, x


@app.cell
def _(mo):
    mo.md(r"## Construir χ² y t desde normales")
    return


@app.cell
def _(mo):
    k = mo.ui.slider(1, 30, value=5, step=1, label="k")
    k
    return (k,)


@app.cell
def _(k, np, plt, stats):
    rng = np.random.default_rng(2)
    R = 100_000
    chi = (rng.standard_normal((R, k.value)) ** 2).sum(1)
    tt = rng.standard_normal(R) / np.sqrt((rng.standard_normal((R, k.value)) ** 2).sum(1) / k.value)
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3.2))
    for a, v, d, rg, ti in (
        (ax2[0], chi, stats.chi2(k.value), (0, stats.chi2.ppf(0.999, k.value)), f"χ²({k.value})"),
        (ax2[1], tt, stats.t(k.value), (-6, 6), f"t({k.value})"),
    ):
        a.hist(v, bins=80, range=rg, density=True, alpha=0.5)
        g = np.linspace(*rg, 300)
        a.plot(g, d.pdf(g), "k")
        a.set_title(ti)
    fig2.tight_layout()
    fig2
    return R, ax2, chi, fig2, rng, tt


@app.cell
def _(mo):
    mo.md(r"## Ajustar t a retornos: ν̂ y su incertidumbre")
    return


@app.cell
def _(mo):
    n_obs = mo.ui.slider(500, 10_000, value=3_000, step=500, label="n")
    n_obs
    return (n_obs,)


@app.cell
def _(mo, n_obs, np, pd, retornos_garch_t, stats):
    r = retornos_garch_t(n_obs.value, seed=2026)["ret"]
    nu_h, loc, sc = stats.t.fit(r)
    mu_n, sd_n = stats.norm.fit(r)
    ll_t = stats.t.logpdf(r, nu_h, loc, sc).sum()
    ll_n = stats.norm.logpdf(r, mu_n, sd_n).sum()
    tab = pd.DataFrame(
        {"parámetros": [f"μ={mu_n:.3f}, σ={sd_n:.3f}", f"ν={nu_h:.2f}, loc={loc:.3f}, escala={sc:.3f}"],
         "log-verosimilitud": [ll_n, ll_t],
         "AIC": [2 * 2 - 2 * ll_n, 2 * 3 - 2 * ll_t]},
        index=["normal", "t"],
    ).round(2)
    mo.ui.table(tab.reset_index(), selection=None)
    return ll_n, ll_t, loc, mu_n, nu_h, r, sc, sd_n, tab


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Con $\hat\nu$ estimado, ¿cuál es la desviación estándar implícita $\hat\sigma_t\sqrt{\hat\nu/(\hat\nu-2)}$? Compárala con `r.std()`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
