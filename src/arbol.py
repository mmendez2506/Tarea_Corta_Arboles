# ==============================
# Tarea Corta 2 — Árboles de Decisión — Inteligencia Artificial
# Integrantes: María Felix Mendez Abarca, Cristhian Rivas y Jozafath Perez
# Descripción: Árbol de decisión hecho desde cero con NumPy. Incluye la clase
#              Nodo, la clase ArbolDecision (fit y predict) y las funciones de
#              impureza (Gini y entropía) usadas para escoger los cortes.
# ==============================
import numpy as np


# ---------------------------------------------------------------------------
# Funciones de impureza
# ---------------------------------------------------------------------------

def _proporciones(labels):
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.array([])
    _, conteos = np.unique(labels, return_counts=True)
    return conteos / labels.size


def gini_impurity(labels):
    """Coeficiente de Gini: 1 - sum(p_i^2)."""
    p = _proporciones(labels)
    if p.size == 0:
        return 0.0
    return float(1.0 - np.sum(p ** 2))


def entropy_impurity(labels):
    """Entropía: -sum(p_i * log2(p_i))."""
    p = _proporciones(labels)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


# Versiones que reciben una matriz de conteos (filas = candidatos de corte,
# columnas = clases). Se usan para evaluar todos los umbrales a la vez.

def _gini_desde_conteos(conteos):
    n = conteos.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, conteos / n, 0.0)
    return 1.0 - np.sum(p ** 2, axis=-1)


def _entropia_desde_conteos(conteos):
    n = conteos.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, conteos / n, 0.0)
        terminos = np.where(p > 0, p * np.log2(p), 0.0)
    return -np.sum(terminos, axis=-1)


_CRITERIOS = {
    "gini": (gini_impurity, _gini_desde_conteos),
    "entropia": (entropy_impurity, _entropia_desde_conteos),
    "entropy": (entropy_impurity, _entropia_desde_conteos),
}
_POR_FUNCION = {
    gini_impurity: _gini_desde_conteos,
    entropy_impurity: _entropia_desde_conteos,
}


def _resolver_criterio(criterio):
    """Acepta "gini" / "entropia" o directamente la función de impureza."""
    if isinstance(criterio, str):
        clave = criterio.lower()
        if clave not in _CRITERIOS:
            raise ValueError(f"Criterio desconocido: {criterio!r}. Use 'gini' o 'entropia'.")
        return _CRITERIOS[clave]
    if criterio in _POR_FUNCION:
        return criterio, _POR_FUNCION[criterio]
    raise ValueError("criterio debe ser 'gini', 'entropia', gini_impurity o entropy_impurity.")


class Nodo:
    def __init__(self, feature=None, umbral=None, gini=None,
                 cantidad_muestras=None, valor=None,
                 izquierda=None, derecha=None):
        self.feature = feature
        self.umbral = umbral
        self.gini = gini
        self.cantidad_muestras = cantidad_muestras
        self.valor = valor
        self.izquierda = izquierda
        self.derecha = derecha


class ArbolDecision:
    def __init__(self, max_depth=5, min_split_samples=2, criterio="gini"):
        self.max_depth = max_depth
        self.min_split_samples = min_split_samples
        self.criterio = criterio
        self.raiz = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
