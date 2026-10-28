# Módulo 1 · Revisión de Estadística y Álgebra Matricial

Especialización Superior en **Econometría de Series de Tiempo y Ciencia de Datos** — 20 sesiones de 1 h 30 min (30 h).

Sitio del curso hecho con **Quarto** (contenidos + presentaciones Reveal.js), laboratorios en **Marimo**, entorno gestionado con **uv** y desarrollo recomendado en **Positron**.

## Puesta en marcha

```bash
# 1. Instalar uv: https://docs.astral.sh/uv/
uv sync                    # crea .venv con Python 3.12 y todas las dependencias
uv run quarto preview      # sitio con recarga en vivo  (o: quarto preview si ya tienes Quarto)
uv run quarto render       # genera _site/
uv run marimo edit sesiones/s01/lab.py   # abrir un laboratorio
```

En Positron: *File ▸ Open Folder…* y seleccionar este directorio; el intérprete `.venv` se detecta automáticamente.

## Estructura

```
_quarto.yml          configuración del sitio
index.qmd            portada
programa.qmd         plan del módulo y cronograma (20 sesiones)
entorno.qmd          guía de instalación (Positron, uv, Quarto, Marimo)
evaluacion.qmd       criterios de evaluación
bibliografia.qmd     referencias
sesiones/sXX/        index.qmd (contenido) · slides.qmd (Reveal.js) · lab.py (Marimo)
src/estalgmat/       datos simulados reproducibles y utilidades
scripts/             pre-render: exporta los laboratorios Marimo a HTML
```

## Estado de desarrollo

| Sesiones | Contenido | Diapositivas | Laboratorio |
|---|---|---|---|
| S01–S20 | desarrollado (contenidos ejecutables) | ✔ | ✔ |
