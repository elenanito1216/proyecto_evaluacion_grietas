"""Evaluacion completa: desempeno, falsos negativos y costo computacional.

Produce **todos** los artefactos que despues consume la aplicacion Streamlit.
Ese es el contrato del proyecto: la app no calcula metricas ni las trae escritas
a mano, las lee de ``reports/metricas/``.

Que evalua:
    - Linea base, MobileNetV2 (.keras) y MobileNetV2 (.tflite int8), sobre el
      mismo conjunto de prueba y con el mismo umbral.
    - Los mismos modelos sobre ``data/propias/``, por separado. La brecha entre
      ambos conjuntos es la medida honesta de generalizacion.
    - Analisis de falsos negativos y busqueda del umbral que garantiza el recall
      objetivo.
    - Complejidad: parametros, tamano en disco, latencia y pico de memoria.

Uso tipico:
    python scripts/evaluate.py --config config.yaml
    python scripts/evaluate.py --config config.yaml --subset 0.2
    python scripts/evaluate.py --config config.yaml --sin-complejidad
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.loader import cargar_particiones, cargar_propias  # noqa: E402
from src.eval import metricas as met  # noqa: E402
from src.eval.complejidad import (  # noqa: E402
    construir_tabla_comparativa,
    perfilar_modelo_keras,
    perfilar_modelo_tflite,
)
from src.models.inferencia import cargar_predictor, predecir_dataset  # noqa: E402
from src.utils.config import cargar_config, forma_entrada, obtener  # noqa: E402
from src.utils.dispositivo import detectar_dispositivo  # noqa: E402
from src.utils.rutas import asegurar_directorio, resolver  # noqa: E402
from src.utils.semillas import fijar_semillas  # noqa: E402


def construir_parser() -> argparse.ArgumentParser:
    """Define la interfaz de linea de comandos del script.

    Returns:
        Parser configurado.
    """
    parser = argparse.ArgumentParser(
        description="Evalua desempeno y costo de todos los modelos entrenados.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al config YAML.")
    parser.add_argument(
        "--subset", type=float, default=None, help="Fraccion del dataset (para pruebas rapidas)."
    )
    parser.add_argument(
        "--sin-complejidad",
        action="store_true",
        help="Omite las mediciones de latencia y memoria (mas rapido).",
    )
    parser.add_argument(
        "--sin-propias",
        action="store_true",
        help="Omite la evaluacion sobre data/propias/.",
    )
    return parser


def _descubrir_modelos(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Localiza los artefactos disponibles y los ordena para la comparativa.

    No falla si falta alguno: es legitimo evaluar solo la linea base antes de
    haber entrenado el transfer, o evaluar sin haber exportado a TFLite.

    Args:
        config: Configuracion del proyecto.

    Returns:
        Lista de diccionarios con ``etiqueta``, ``formato`` y ``ruta``, solo de
        los artefactos que existen en disco.
    """
    candidatos = [
        {
            "etiqueta": "Linea base (CNN propia)",
            "formato": "keras",
            "ruta": obtener(config, "app.archivo_baseline"),
        },
        {
            "etiqueta": "MobileNetV2 (.keras)",
            "formato": "keras",
            "ruta": obtener(config, "app.archivo_keras"),
        },
        {
            "etiqueta": "MobileNetV2 TFLite int8",
            "formato": "tflite",
            "ruta": obtener(config, "app.archivo_tflite"),
        },
    ]
    disponibles = []
    for candidato in candidatos:
        if candidato["ruta"] and resolver(candidato["ruta"]).is_file():
            disponibles.append(candidato)
        else:
            print(f"  (omitido) {candidato['etiqueta']}: no se encontro {candidato['ruta']}")
    return disponibles


def _evaluar_conjunto(
    predictor: Any,
    dataset: Any,
    inventario: Any,
    umbral: float,
    etiqueta_modelo: str,
    nombre_conjunto: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evalua un modelo sobre un conjunto y genera sus artefactos.

    Args:
        predictor: Predictor cargado.
        dataset: ``tf.data.Dataset`` del conjunto.
        inventario: Inventario correspondiente (para nombres de archivo en el
            analisis de falsos negativos).
        umbral: Umbral de decision.
        etiqueta_modelo: Nombre del modelo para los titulos y nombres de archivo.
        nombre_conjunto: ``"test"`` o ``"propias"``.
        config: Configuracion del proyecto.

    Returns:
        Diccionario con ``metricas``, ``curva_pr``, ``umbral_recall`` y
        ``falsos_negativos``.
    """
    y_true, y_prob = predecir_dataset(predictor, dataset)
    if y_true.size == 0:
        return {"error": "El conjunto no produjo ninguna prediccion."}

    metricas = met.calcular_metricas(y_true, y_prob, umbral)
    resultado: dict[str, Any] = {"metricas": metricas}

    clases = obtener(config, "datos.clases", ["Negative", "Positive"])
    slug = _slug(etiqueta_modelo)
    met.figura_matriz_confusion(
        metricas,
        clases,
        f"Matriz de confusion - {etiqueta_modelo} ({nombre_conjunto})",
        f"reports/figuras/matriz_{slug}_{nombre_conjunto}.png",
    )

    if len(np.unique(y_true.astype(int))) == 2:
        curva = met.curva_precision_recall(y_true, y_prob)
        resultado["curva_pr"] = curva
        met.figura_precision_recall(
            curva,
            f"Curva precision-recall - {etiqueta_modelo} ({nombre_conjunto})",
            f"reports/figuras/pr_{slug}_{nombre_conjunto}.png",
        )
        objetivo = float(obtener(config, "evaluacion.recall_objetivo", 0.95))
        resultado["umbral_recall"] = met.buscar_umbral_para_recall(y_true, y_prob, objetivo)
    else:
        resultado["curva_pr"] = None
        resultado["umbral_recall"] = None

    resultado["falsos_negativos"] = met.analizar_falsos_negativos(
        y_true, y_prob, rutas=inventario.rutas if inventario is not None else None, umbral=umbral
    )
    resultado["probabilidades"] = {
        "y_true": y_true.tolist(),
        "y_prob": [round(float(v), 6) for v in y_prob],
    }
    return resultado


def _slug(texto: str) -> str:
    """Convierte una etiqueta legible en un identificador apto para archivos.

    Args:
        texto: Etiqueta original.

    Returns:
        Cadena en minusculas, sin espacios ni caracteres problematicos.
    """
    limpio = texto.lower()
    for viejo, nuevo in [(" ", "_"), ("(", ""), (")", ""), (".", ""), ("ñ", "n"), ("í", "i")]:
        limpio = limpio.replace(viejo, nuevo)
    return limpio


def main() -> int:
    """Punto de entrada de la evaluacion.

    Returns:
        Codigo de salida del proceso: 0 si termino correctamente.
    """
    args = construir_parser().parse_args()
    config = cargar_config(args.config)
    fijar_semillas(int(obtener(config, "proyecto.semilla", 42)))
    info_dispositivo = detectar_dispositivo()

    umbral = float(obtener(config, "evaluacion.umbral", 0.5))
    forma = forma_entrada(config)

    print("\n[1/5] Localizando modelos")
    modelos = _descubrir_modelos(config)
    if not modelos:
        print(
            "\nERROR: no hay ningun modelo entrenado.\n"
            "  python scripts/train_baseline.py --config config.yaml\n"
            "  python scripts/train_transfer.py --config config.yaml"
        )
        return 1
    for modelo in modelos:
        print(f"  encontrado: {modelo['etiqueta']}")

    print("\n[2/5] Cargando conjuntos de evaluacion")
    datasets, inventarios = cargar_particiones(config, subset=args.subset)
    ds_propias, inv_propias = (None, None)
    if not args.sin_propias:
        ds_propias, inv_propias = cargar_propias(config)
        if ds_propias is None:
            print(
                "  AVISO: no hay fotos propias en data/propias/. El docente exige\n"
                "         probar con fotografias tomadas por el equipo. Colocalas en\n"
                "         data/propias/Positive/ y data/propias/Negative/ y vuelve a\n"
                "         ejecutar este script."
            )
        else:
            conteo = inv_propias.conteo_por_clase()
            print(f"  Fotos propias: {len(inv_propias)} imagenes {conteo}")

    print("\n[3/5] Evaluando desempeno")
    resultados: dict[str, Any] = {
        "umbral": umbral,
        "dispositivo": info_dispositivo,
        "n_test": len(inventarios["test"]),
        "modelos": {},
    }
    metricas_por_modelo: dict[str, dict[str, Any]] = {}
    perfiles: list[dict[str, Any]] = []

    for entrada in modelos:
        etiqueta = entrada["etiqueta"]
        print(f"\n  --- {etiqueta} ---")
        predictor = cargar_predictor(config, entrada["formato"], entrada["ruta"])

        bloque: dict[str, Any] = {
            "formato": entrada["formato"],
            "archivo": str(Path(entrada["ruta"]).name),
            "descripcion": predictor.descripcion(),
        }

        bloque["test"] = _evaluar_conjunto(
            predictor, datasets["test"], inventarios["test"], umbral, etiqueta, "test", config
        )
        m = bloque["test"]["metricas"]
        metricas_por_modelo[etiqueta] = m
        print(
            f"  test    : accuracy={m['accuracy']:.4f}  precision={m['precision']:.4f}  "
            f"recall={m['recall']:.4f}  f1={m['f1']:.4f}  FN={m['matriz_confusion']['fn']}"
        )
        fn = bloque["test"]["falsos_negativos"]
        print(
            f"            falsos negativos: {fn['n_falsos_negativos']} de "
            f"{fn['n_positivos_reales']} grietas reales ({fn['tasa']:.2%})"
        )
        if bloque["test"].get("umbral_recall"):
            ur = bloque["test"]["umbral_recall"]
            print(
                f"            para recall>={ur['recall_objetivo']:.0%} usar umbral "
                f"{ur['umbral']:.3f} (precision resultante {ur['precision_resultante']:.3f})"
            )

        if ds_propias is not None:
            bloque["propias"] = _evaluar_conjunto(
                predictor, ds_propias, inv_propias, umbral, etiqueta, "propias", config
            )
            mp = bloque["propias"].get("metricas")
            if mp:
                print(
                    f"  propias : accuracy={mp['accuracy']:.4f}  recall={mp['recall']:.4f}  "
                    f"f1={mp['f1']:.4f}  (brecha F1 = {mp['f1'] - m['f1']:+.4f})"
                )

        if not args.sin_complejidad:
            print("  midiendo complejidad...", end=" ", flush=True)
            if entrada["formato"] == "keras":
                perfil = perfilar_modelo_keras(
                    predictor.modelo, forma, entrada["ruta"], config, etiqueta
                )
            else:
                perfil = perfilar_modelo_tflite(entrada["ruta"], config, etiqueta)
            perfiles.append(perfil)
            bloque["complejidad"] = perfil
            print(
                f"{perfil['latencia']['latencia_media_ms']:.2f} +- "
                f"{perfil['latencia']['latencia_std_ms']:.2f} ms/imagen, "
                f"{perfil['tamano_mb']:.2f} MB"
            )

        resultados["modelos"][etiqueta] = bloque

    print("\n[4/5] Tabla comparativa desempeno vs costo")
    if perfiles:
        filas = construir_tabla_comparativa(perfiles, metricas_por_modelo)
        tabla = pd.DataFrame(filas)
        destino_csv = asegurar_directorio("reports/metricas") / "comparativa.csv"
        tabla.to_csv(destino_csv, index=False, encoding="utf-8")
        resultados["comparativa"] = filas

        columnas = [
            "modelo",
            "parametros_total",
            "tamano_mb",
            "latencia_media_ms",
            "accuracy",
            "recall",
            "f1",
            "falsos_negativos",
        ]
        presentes = [c for c in columnas if c in tabla.columns]
        print(tabla[presentes].to_string(index=False))
        print(f"\n  Tabla -> {destino_csv}")
    else:
        print("  (omitida: se ejecuto con --sin-complejidad)")

    print("\n[5/5] Guardando artefactos para la aplicacion")
    ruta_json = met.guardar_json(resultados, "reports/metricas/evaluacion.json")
    print(f"  Evaluacion -> {ruta_json}")
    print("\nListo. Lanza la aplicacion con: streamlit run app/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
