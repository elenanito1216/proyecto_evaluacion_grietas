"""Deteccion del dispositivo de computo y medicion del costo de entrenamiento.

El enunciado exige registrar en que hardware se entrena y cuanto cuesta cada
epoca: forma parte del analisis de costo computacional de la asignatura, no es
solo informacion de consola.
"""

from __future__ import annotations

import platform
import time
from typing import Any


def detectar_dispositivo(verboso: bool = True) -> dict[str, Any]:
    """Detecta si hay GPU disponible para TensorFlow e informa por consola.

    Args:
        verboso: Si ``True``, imprime un resumen legible del hardware.

    Returns:
        Diccionario con ``dispositivo`` ("GPU" o "CPU"), ``n_gpus``,
        ``nombres_gpu``, ``version_tf``, ``python`` y ``so``.

    Note:
        En Windows nativo, TensorFlow > 2.10 **no** soporta GPU CUDA; solo via
        WSL2. Que aqui salga "CPU" en Windows es lo esperado, no un error de
        configuracion.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    info: dict[str, Any] = {
        "dispositivo": "GPU" if gpus else "CPU",
        "n_gpus": len(gpus),
        "nombres_gpu": [g.name for g in gpus],
        "version_tf": tf.__version__,
        "python": platform.python_version(),
        "so": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "desconocido",
    }

    if verboso:
        print("=" * 72)
        print("  ENTORNO DE COMPUTO")
        print("=" * 72)
        print(f"  TensorFlow      : {info['version_tf']}")
        print(f"  Python          : {info['python']}")
        print(f"  Sistema         : {info['so']}")
        print(f"  Procesador      : {info['cpu']}")
        if gpus:
            print(f"  Dispositivo     : GPU ({len(gpus)} disponible/s)")
            for nombre in info["nombres_gpu"]:
                print(f"                    - {nombre}")
        else:
            print("  Dispositivo     : CPU (no se detecto GPU)")
            if platform.system() == "Windows":
                print(
                    "  Aviso           : TensorFlow > 2.10 no soporta GPU en Windows\n"
                    "                    nativo. Es normal entrenar en CPU aqui."
                )
        print("=" * 72)

    return info


class MedidorDeEpocas:
    """Callback de Keras que cronometra cada epoca y estima el tiempo restante.

    Se implementa como callback en lugar de medir el ``fit`` completo porque la
    duracion por epoca es el dato util: permite decidir si abortar una corrida
    larga antes de perder una hora, y es la cifra que se reporta en el analisis
    de costo computacional.

    Attributes:
        tiempos: Duracion en segundos de cada epoca completada.
    """

    def __init__(self) -> None:
        """Inicializa el medidor sin ninguna epoca registrada."""
        self.tiempos: list[float] = []
        self._inicio_epoca: float = 0.0
        self._inicio_total: float = 0.0
        self._total_epocas: int = 0

    def crear_callback(self, total_epocas: int) -> Any:
        """Construye el ``keras.callbacks.Callback`` asociado a este medidor.

        Args:
            total_epocas: Numero de epocas planificadas, usado para estimar el
                tiempo restante.

        Returns:
            Instancia de callback lista para pasar a ``model.fit``.
        """
        import tensorflow as tf

        medidor = self
        medidor._total_epocas = total_epocas

        class _CallbackTiempo(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs: dict | None = None) -> None:
                medidor._inicio_total = time.perf_counter()

            def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
                medidor._inicio_epoca = time.perf_counter()

            def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
                duracion = time.perf_counter() - medidor._inicio_epoca
                medidor.tiempos.append(duracion)
                promedio = sum(medidor.tiempos) / len(medidor.tiempos)
                restantes = max(medidor._total_epocas - (epoch + 1), 0)
                eta_min = promedio * restantes / 60.0
                print(
                    f"    [tiempo] epoca {epoch + 1}: {duracion:6.1f} s  |  "
                    f"promedio: {promedio:6.1f} s  |  ETA: {eta_min:5.1f} min"
                )

        return _CallbackTiempo()

    def resumen(self) -> dict[str, float]:
        """Resume los tiempos medidos.

        Returns:
            Diccionario con ``epocas_medidas``, ``segundos_por_epoca_media``,
            ``segundos_por_epoca_min``, ``segundos_por_epoca_max`` y
            ``segundos_totales``. Devuelve ceros si no se completo ninguna epoca.
        """
        if not self.tiempos:
            return {
                "epocas_medidas": 0,
                "segundos_por_epoca_media": 0.0,
                "segundos_por_epoca_min": 0.0,
                "segundos_por_epoca_max": 0.0,
                "segundos_totales": 0.0,
            }
        return {
            "epocas_medidas": len(self.tiempos),
            "segundos_por_epoca_media": float(sum(self.tiempos) / len(self.tiempos)),
            "segundos_por_epoca_min": float(min(self.tiempos)),
            "segundos_por_epoca_max": float(max(self.tiempos)),
            "segundos_totales": float(sum(self.tiempos)),
        }
