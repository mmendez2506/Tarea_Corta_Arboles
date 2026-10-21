# ==============================
# Tarea Corta 2 — Árboles de Decisión — Inteligencia Artificial
# Integrantes: María Felix Mendez Abarca, Cristhian Rivas y Jozafath Perez
# Descripción: Métricas de clasificación (accuracy, precision, recall, F1)
#              hechas con NumPy y la función validacion_cruzada con k folds.
# ==============================
import numpy as np


def accuracy(y_true, y_pred):
    raise NotImplementedError


def precision(y_true, y_pred):
    raise NotImplementedError


def recall(y_true, y_pred):
    raise NotImplementedError


def f1(y_true, y_pred):
    raise NotImplementedError


def validacion_cruzada(modelo_cls, params, X, y, k=5):
    """Devuelve media y desviación estándar de cada métrica."""
    raise NotImplementedError
