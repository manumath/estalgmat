import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from estalgmat import retornos_garch_t

    return mo, np, pd, retornos_garch_t


@app.cell
def _(mo):
    mo.md(r"# S05 · Probabilidad: conjuntos, condicional y Bayes")
    return


@app.cell
def _(mo):
    mo.md(r"## 1. Conjuntos")
    return


@app.cell
def _(mo):
    om = set(range(1, 11))
    A = {n for n in om if n % 2 == 0}
    B = {n for n in om if n > 6}
    comprobaciones = [
        ("De Morgan 1", om - (A | B) == (om - A) & (om - B)),
        ("De Morgan 2", om - (A & B) == (om - A) | (om - B)),
        ("Inclusión–exclusión", len(A | B) == len(A) + len(B) - len(A & B)),
    ]
    mo.md("\n".join(f"- {n}: {'✔' if ok else '✘'}" for n, ok in comprobaciones))
    return A, B, comprobaciones, om


@app.cell
def _(mo):
    mo.md(r"## 2. Bayes: la alerta de default")
    return


@app.cell
def _(mo):
    tasa = mo.ui.slider(0.005, 0.30, value=0.02, step=0.005, label="tasa base de default")
    sens = mo.ui.slider(0.5, 0.99, value=0.90, step=0.01, label="sensibilidad")
    fp = mo.ui.slider(0.01, 0.30, value=0.10, step=0.01, label="falsos positivos")
    mo.hstack([tasa, sens, fp], justify="start")
    return fp, sens, tasa


@app.cell
def _(fp, mo, np, sens, tasa):
    rng = np.random.default_rng(1)
    N = 500_000
    d = rng.random(N) < tasa.value
    alerta = np.where(d, rng.random(N) < sens.value, rng.random(N) < fp.value)
    sim = d[alerta].mean()
    teo = sens.value * tasa.value / (sens.value * tasa.value + fp.value * (1 - tasa.value))
    mo.md(
        f"**P(default | alerta)** simulado = `{sim:.4f}` · Bayes = `{teo:.4f}`  \n"
        f"Falsas alarmas entre las alertas: **{1 - teo:.1%}**"
    )
    return N, alerta, d, rng, sim, teo


@app.cell
def _(mo):
    mo.md(r"## 3. Condicionar por régimen de volatilidad")
    return


@app.cell
def _(mo):
    umbral = mo.ui.slider(-4, -1, value=-2, step=0.5, label="umbral de caída (%)")
    umbral
    return (umbral,)


@app.cell
def _(mo, pd, retornos_garch_t, umbral):
    df = retornos_garch_t(20000, seed=5)
    A_ = df["ret"] < umbral.value
    B_ = df["sigma"] > df["sigma"].median()
    tab = pd.DataFrame(
        {
            "P(A)": [A_.mean()],
            "P(A | vol alta)": [(A_ & B_).sum() / B_.sum()],
            "P(A | vol baja)": [(A_ & ~B_).sum() / (~B_).sum()],
        }
    ).round(4)
    mo.ui.table(tab, selection=None)
    return A_, B_, df, tab


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Ejercicio final
        Simula una cartera de crédito con dos segmentos (tasas base 1 % y 8 %, mezcla 70/30) y calcula $P(\text{segmento}\mid\text{default})$ con Bayes y por simulación.
        """
    )
    return


if __name__ == "__main__":
    app.run()
