# ==============================
# Tarea Corta 2 — Árboles de Decisión — Inteligencia Artificial
# Integrantes: María Felix Mendez Abarca, Cristhian Rivas y Jozafath Perez
# Descripción: Métricas de clasificación (accuracy, precision, recall, F1)
#              hechas con NumPy y la función validacion_cruzada con k folds.
# ==============================
import numpy as np

from src.datos import SEMILLA, filas

# Todas las métricas toman la clase 1 (>50K) como la positiva.


def _conteos(y_true, y_pred):
    """Verdaderos positivos, falsos positivos y falsos negativos."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, fp, fn


def accuracy(y_true, y_pred):
    """Fracción de predicciones correctas."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision(y_true, y_pred):
    """De los que el modelo dijo que eran 1, cuántos lo eran. 0 si nunca predijo 1."""
    tp, fp, _ = _conteos(y_true, y_pred)
    return float(tp / (tp + fp)) if tp + fp else 0.0


def recall(y_true, y_pred):
    """De los que eran 1, cuántos encontró el modelo. 0 si no hay ningún 1."""
    tp, _, fn = _conteos(y_true, y_pred)
    return float(tp / (tp + fn)) if tp + fn else 0.0


def f1(y_true, y_pred):
    """Media armónica de precision y recall."""
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if p + r else 0.0


METRICAS = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def validacion_cruzada(modelo_cls, params, X, y, k=5):
    """Devuelve media y desviación estándar de cada métrica.

    Baraja los índices con SEMILLA y los parte en k folds excluyentes. En cada
    vuelta un fold es validación y los otros k-1 son entrenamiento; se crea un
    modelo nuevo con modelo_cls(**params) para que ningún fold vea datos de otro.

    Formato: {"accuracy_media": ..., "accuracy_desv": ..., "f1_media": ..., ...}
    """
    if not 2 <= k <= len(y):
        raise ValueError("k debe estar entre 2 y la cantidad de filas")
    folds = np.array_split(np.random.default_rng(SEMILLA).permutation(len(y)), k)
    valores = {m: [] for m in METRICAS}
    for i, val in enumerate(folds):
        train = np.concatenate(folds[:i] + folds[i + 1:])
        modelo = modelo_cls(**params)
        modelo.fit(filas(X, train), y[train])
        y_pred = modelo.predict(filas(X, val))
        for nombre, metrica in METRICAS.items():
            valores[nombre].append(metrica(y[val], y_pred))
    resultado = {}
    for nombre, v in valores.items():
        resultado[f"{nombre}_media"] = float(np.mean(v))
        resultado[f"{nombre}_desv"] = float(np.std(v))
    return resultado
