"""Linea base: CNN pequena entrenada desde cero.

La asignatura exige un punto de comparacion propio antes de recurrir a transfer
learning. Este modelo cumple ese papel: es deliberadamente sencillo (3 bloques
convolucionales, ~100k parametros) para responder a la pregunta "cuanto aporta
realmente MobileNetV2 preentrenado frente a una red modesta entrenada desde
cero, y a que costo".

Entrada esperada: tensor ``(N, H, W, 3)`` en el rango ``[0, 1]``, tal como lo
entrega ``src.data.loader``. No lleva capa de reescalado adicional.
"""

from __future__ import annotations

from typing import Any

from src.utils.config import forma_entrada, obtener


def construir_baseline(config: dict[str, Any]) -> Any:
    """Construye la CNN de linea base.

    Arquitectura: ``[Conv-BN-ReLU -> MaxPool] x 3 -> GAP -> Dropout -> Densa ->
    Dropout -> Sigmoide``.

    Decisiones y su porque:
        - **BatchNormalization** tras cada convolucion: entrenar desde cero en
          CPU con lotes pequenos converge muy lento sin normalizacion interna.
        - **GlobalAveragePooling2D** en vez de ``Flatten``: un ``Flatten`` a esta
          resolucion generaria cientos de miles de pesos en la primera densa, y
          el modelo dejaria de ser una linea base ligera.
        - **Salida sigmoide de 1 unidad** (no softmax de 2): el problema es
          binario y una sola probabilidad permite mover el umbral de decision
          directamente, que es lo que se necesita para priorizar recall.

    Args:
        config: Configuracion del proyecto.

    Returns:
        Modelo de Keras sin compilar.
    """
    import tensorflow as tf

    alto, ancho, canales = forma_entrada(config)
    filtros: list[int] = obtener(config, "baseline.filtros", [16, 32, 64])
    kernel = int(obtener(config, "baseline.kernel", 3))
    dropout = float(obtener(config, "baseline.dropout", 0.3))
    unidades = int(obtener(config, "baseline.unidades_densa", 64))
    nombre = str(obtener(config, "baseline.nombre", "baseline_cnn"))

    entradas = tf.keras.Input(shape=(alto, ancho, canales), name="imagen")
    x = entradas
    for i, n_filtros in enumerate(filtros):
        x = tf.keras.layers.Conv2D(
            n_filtros, kernel, padding="same", use_bias=False, name=f"conv{i + 1}"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn{i + 1}")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu{i + 1}")(x)
        x = tf.keras.layers.MaxPooling2D(2, name=f"pool{i + 1}")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout1")(x)
    x = tf.keras.layers.Dense(unidades, activation="relu", name="densa")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout2")(x)
    salidas = tf.keras.layers.Dense(1, activation="sigmoid", name="p_grieta")(x)

    return tf.keras.Model(entradas, salidas, name=nombre)


def compilar(modelo: Any, tasa_aprendizaje: float) -> Any:
    """Compila un modelo binario con las metricas relevantes del dominio.

    Se incluyen ``Recall``, ``Precision`` y ``AUC`` ademas de ``accuracy``
    porque en deteccion de grietas la exactitud sola es enganosa: con un dataset
    desbalanceado, predecir siempre "sin grieta" puede dar una exactitud alta y
    un recall nulo en la clase peligrosa.

    Args:
        modelo: Modelo de Keras sin compilar.
        tasa_aprendizaje: Learning rate inicial para Adam.

    Returns:
        El mismo modelo, ya compilado.
    """
    import tensorflow as tf

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=tasa_aprendizaje),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return modelo
