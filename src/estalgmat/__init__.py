"""Utilidades del Módulo 1: datos simulados reproducibles y helpers de EDA."""

from .datos import (
    cargar_yahoo,
    curva_tasas,
    precios_desde_retornos,
    retornos_garch_t,
    serie_macro,
)

__all__ = [
    "cargar_yahoo",
    "curva_tasas",
    "precios_desde_retornos",
    "retornos_garch_t",
    "serie_macro",
]
