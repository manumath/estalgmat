"""Generadores de datos reproducibles (offline) y cargador opcional de datos reales.

Todo el curso puede ejecutarse sin conexión: los datos simulados reproducen los
*hechos estilizados* que se estudian en el módulo (colas pesadas, agrupamiento de
volatilidad, estructura factorial de la curva de tasas, estacionalidad, etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def retornos_garch_t(
    n: int = 2500,
    *,
    omega: float = 0.02,
    alpha: float = 0.08,
    beta: float = 0.90,
    gamma: float = 0.0,
    nu: float = 5.0,
    mu: float = 0.03,
    inicio: str = "2015-01-01",
    seed: int = 2026,
) -> pd.DataFrame:
    """Retornos logarítmicos diarios (en %) de un GARCH(1,1) con innovaciones t de Student.

    ``gamma > 0`` añade el término de apalancamiento de GJR-GARCH (los choques negativos
    elevan más la volatilidad), lo que genera asimetría negativa.

    Devuelve un DataFrame indexado por días hábiles con columnas ``ret`` y ``sigma``.
    Las innovaciones se estandarizan a varianza unitaria.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, size=n) / np.sqrt(nu / (nu - 2))
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)
    r[0] = mu + np.sqrt(sigma2[0]) * z[0]
    for t in range(1, n):
        e = r[t - 1] - mu
        sigma2[t] = omega + (alpha + gamma * (e < 0)) * e**2 + beta * sigma2[t - 1]
        r[t] = mu + np.sqrt(sigma2[t]) * z[t]
    idx = pd.bdate_range(inicio, periods=n, name="fecha")
    return pd.DataFrame({"ret": r, "sigma": np.sqrt(sigma2)}, index=idx)


def precios_desde_retornos(ret: pd.Series, p0: float = 100.0) -> pd.Series:
    """Precio a partir de retornos logarítmicos en % (p0 = nivel inicial)."""
    return p0 * np.exp(ret.cumsum() / 100.0)


def serie_macro(n: int = 240, *, inicio: str = "2005-01-01", seed: int = 7) -> pd.Series:
    """Serie mensual tipo IMAE/ventas: tendencia + estacionalidad + AR(1) + atípicos."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    tendencia = 100 + 0.35 * t
    estacional = 4.0 * np.sin(2 * np.pi * t / 12) + 1.5 * np.cos(4 * np.pi * t / 12)
    ruido = np.zeros(n)
    for i in range(1, n):
        ruido[i] = 0.6 * ruido[i - 1] + rng.normal(0, 1.2)
    y = tendencia + estacional + ruido
    for i in rng.choice(np.arange(12, n - 12), size=4, replace=False):
        y[i] += rng.choice([-1, 1]) * rng.uniform(8, 12)  # atípicos aditivos
    idx = pd.date_range(inicio, periods=n, freq="MS", name="fecha")
    return pd.Series(y, index=idx, name="indice")


def curva_tasas(n: int = 1500, *, inicio: str = "2018-01-01", seed: int = 11) -> pd.DataFrame:
    """Curva de rendimientos (%) con 8 vencimientos generada por 3 factores latentes
    (nivel, pendiente, curvatura) más ruido. Útil para PCA/SVD (sesión 18)."""
    rng = np.random.default_rng(seed)
    venc = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
    lam = 0.6
    carga_nivel = np.ones_like(venc)
    carga_pend = (1 - np.exp(-lam * venc)) / (lam * venc)
    carga_curv = carga_pend - np.exp(-lam * venc)
    L = np.column_stack([carga_nivel, carga_pend, carga_curv])
    f = np.zeros((n, 3))
    phi = np.array([0.999, 0.99, 0.97])
    sd = np.array([0.04, 0.05, 0.08])
    for t in range(1, n):
        f[t] = phi * f[t - 1] + rng.normal(0, sd)
    medias = np.array([3.0, -1.0, 0.5])
    y = (f + medias) @ L.T + rng.normal(0, 0.03, size=(n, len(venc)))
    idx = pd.bdate_range(inicio, periods=n, name="fecha")
    return pd.DataFrame(y, index=idx, columns=[f"{v:g}a" for v in venc])


def cargar_yahoo(
    ticker: str,
    inicio: str = "2010-01-01",
    fin: str | None = None,
    cache: str | Path = "data/cache",
) -> pd.Series:
    """Precio de cierre ajustado desde Yahoo Finance, con caché local en parquet/CSV.

    Requiere ``uv sync --extra datos``. Si no hay conexión usa la caché si existe.
    """
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{ticker.replace('^', '_')}.csv"
    try:
        import yfinance as yf

        df = yf.download(ticker, start=inicio, end=fin, auto_adjust=True, progress=False)
        s = df["Close"].squeeze().rename(ticker)
        s.to_csv(f)
        return s
    except Exception:
        if f.exists():
            return pd.read_csv(f, index_col=0, parse_dates=True).squeeze("columns")
        raise
