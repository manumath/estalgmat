import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt, time


@app.cell
def _(mo):
    mo.md(
        r"""
        # S02 · Python científico I

        ## 1. Estadísticos sin librerías

        Implementa `media`, `varianza` (muestral, `ddof=1`) y compara con NumPy.
        """
    )
    return


@app.cell
def _():
    def media(x):
        return sum(x) / len(x)

    def varianza(x):
        m = media(x)
        return sum((v - m) ** 2 for v in x) / (len(x) - 1)

    return media, varianza


@app.cell
def _(media, mo, np, varianza):
    datos = [2.1, 3.4, 1.9, 5.0, 4.2, 3.3]
    ok = [
        np.isclose(media(datos), np.mean(datos)),
        np.isclose(varianza(datos), np.var(datos, ddof=1)),
    ]
    mo.md("### Pruebas\n" + "\n".join(f"- {n}: {'✔' if o else '✘'}" for n, o in zip(("media", "varianza"), ok)))
    return datos, ok


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Retornos y caminata aleatoria

        $$r^s_t=\frac{P_t}{P_{t-1}}-1,\qquad r_t=\ln P_t-\ln P_{t-1}$$
        """
    )
    return


@app.cell
def _(np):
    def retornos(precios, tipo="log"):
        p = np.asarray(precios, dtype=float)
        if tipo == "log":
            return np.diff(np.log(p))
        if tipo == "simple":
            return p[1:] / p[:-1] - 1
        raise ValueError("tipo debe ser 'log' o 'simple'")

    def simular_gbm(n_paths, n_steps, mu, sigma, seed=0):
        rng = np.random.default_rng(seed)
        eps = rng.standard_normal((n_paths, n_steps))
        return 100 * np.exp(np.cumsum(mu + sigma * eps, axis=1))

    return retornos, simular_gbm


@app.cell
def _(mo, np, retornos, simular_gbm):
    p = np.array([100, 110, 99.0])
    P = simular_gbm(50_000, 100, 0.0, 0.01, seed=1)
    checks = [
        np.allclose(retornos(p, "simple"), [0.10, -0.10]),
        np.isclose(retornos(p, "log").sum(), np.log(99 / 100)),
        P.shape == (50_000, 100),
        abs(np.log(P[:, -1] / 100).std() - 0.1) < 0.005,
    ]
    mo.md("### Pruebas\n" + "\n".join(f"- {i + 1}: {'✔' if c else '✘'}" for i, c in enumerate(checks)))
    return P, checks, p


@app.cell
def _(mo):
    n_show = mo.ui.slider(5, 100, value=20, label="trayectorias a mostrar")
    mu = mo.ui.slider(-0.002, 0.004, value=0.0005, step=0.0005, label="μ")
    sg = mo.ui.slider(0.002, 0.03, value=0.01, step=0.002, label="σ")
    mo.hstack([n_show, mu, sg], justify="start")
    return mu, n_show, sg


@app.cell
def _(mu, n_show, plt, sg, simular_gbm):
    PP = simular_gbm(n_show.value, 250, mu.value, sg.value, seed=3)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(PP.T, lw=0.7)
    ax.set_xlabel("día")
    ax.set_ylabel("precio")
    fig
    return PP, ax, fig


@app.cell
def _(mo):
    mo.md(r"## 3. Bucle vs. vectorización")
    return


@app.cell
def _(np, time):
    def cronometrar(f, *a):
        t0 = time.perf_counter()
        f(*a)
        return time.perf_counter() - t0

    def bucle(n):
        rng = np.random.default_rng(1)
        s = 0.0
        for _ in range(n):
            s += rng.standard_normal()
        return s

    def vec(n):
        return np.random.default_rng(1).standard_normal(n).sum()

    n_ = 300_000
    t_b, t_v = cronometrar(bucle, n_), cronometrar(vec, n_)
    return bucle, cronometrar, n_, t_b, t_v, vec


@app.cell
def _(mo, t_b, t_v):
    mo.md(f"Bucle: **{t_b:.3f} s** · Vectorizado: **{t_v:.4f} s** · factor ≈ **{t_b / t_v:,.0f}×**")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Escribe `max_drawdown(precio)` en una celda nueva (sin bucles) y verifica con `[100, 120, 90, 110]` → `0.25`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
