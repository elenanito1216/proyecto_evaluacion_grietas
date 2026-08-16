"""Configuracion compartida de las pruebas.

Inserta la raiz del repositorio en ``sys.path`` para que ``import src...``
funcione al ejecutar ``pytest`` desde cualquier directorio, y define las
fixtures comunes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Configuracion real del proyecto, leida de ``config.yaml``.

    Se usa la configuracion real y no una falsa: si alguien cambia un umbral en
    el YAML y eso rompe una regla, la prueba debe enterarse. Es la forma de que
    el archivo de configuracion no se desincronice del comportamiento esperado.

    Returns:
        Diccionario de configuracion.
    """
    from src.utils.config import cargar_config

    return cargar_config()


@pytest.fixture()
def config_riesgo_minima() -> dict[str, Any]:
    """Configuracion sintetica y minima para probar el motor de reglas aislado.

    Returns:
        Diccionario con solo la seccion ``riesgo``, con umbrales conocidos.
    """
    return {
        "riesgo": {
            "umbral_grieta": 0.5,
            "umbral_grieta_alta": 0.85,
            "desaplome_atencion_grados": 1.0,
            "desaplome_severo_grados": 2.0,
            "elementos_criticos": ["columna", "viga", "muro_portante"],
            "orientaciones_graves": {
                "muro_portante": ["diagonal"],
                "columna": ["horizontal", "diagonal"],
                "viga": ["diagonal", "vertical"],
            },
            "min_confianza_inclinacion": 3,
        }
    }
