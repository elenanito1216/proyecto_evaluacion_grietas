"""Carga y acceso a ``config.yaml``.

Un unico punto de verdad para la configuracion. Los modulos reciben el
diccionario ya cargado como argumento en lugar de leerlo por su cuenta: eso los
mantiene puros y testeables (se les puede inyectar una configuracion falsa en
las pruebas sin tocar el disco).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.rutas import RAIZ_PROYECTO, resolver

CONFIG_POR_DEFECTO: str = "config.yaml"


def cargar_config(ruta: str | Path | None = None) -> dict[str, Any]:
    """Lee el archivo de configuracion YAML del proyecto.

    Args:
        ruta: Ruta al YAML. Si es ``None`` se usa ``config.yaml`` en la raiz del
            repositorio.

    Returns:
        Diccionario anidado con toda la configuracion.

    Raises:
        FileNotFoundError: Si el archivo indicado no existe.
        ValueError: Si el YAML esta vacio o no representa un diccionario.
    """
    destino = resolver(ruta) if ruta is not None else RAIZ_PROYECTO / CONFIG_POR_DEFECTO
    if not destino.is_file():
        raise FileNotFoundError(
            f"No se encontro el archivo de configuracion: {destino}\n"
            "Ejecuta los scripts desde la raiz del repositorio o pasa --config."
        )

    with destino.open("r", encoding="utf-8") as manejador:
        contenido = yaml.safe_load(manejador)

    if not isinstance(contenido, dict):
        raise ValueError(f"El archivo {destino} no contiene un mapeo YAML valido.")

    return contenido


def obtener(config: dict[str, Any], clave: str, defecto: Any = None) -> Any:
    """Lee una clave anidada usando notacion de punto.

    Evita cadenas frageles del tipo ``cfg["a"]["b"]["c"]`` que revientan con
    ``KeyError`` cuando falta un nivel intermedio.

    Args:
        config: Diccionario de configuracion.
        clave: Ruta de la clave separada por puntos, p. ej. ``"riesgo.umbral_grieta"``.
        defecto: Valor devuelto si la clave no existe.

    Returns:
        El valor encontrado o ``defecto``.

    Example:
        >>> obtener({"a": {"b": 1}}, "a.b")
        1
        >>> obtener({"a": {"b": 1}}, "a.z", defecto=0)
        0
    """
    actual: Any = config
    for parte in clave.split("."):
        if not isinstance(actual, dict) or parte not in actual:
            return defecto
        actual = actual[parte]
    return actual


def forma_entrada(config: dict[str, Any]) -> tuple[int, int, int]:
    """Devuelve la forma ``(alto, ancho, canales)`` de la entrada del modelo.

    Args:
        config: Diccionario de configuracion.

    Returns:
        Tupla ``(H, W, C)``. El tensor de un lote es entonces ``(N, H, W, C)``.
    """
    pre = config["preproceso"]
    return int(pre["alto"]), int(pre["ancho"]), int(pre["canales"])
