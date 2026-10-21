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


# ---------------------------------------------------------------------------
# Ganancia de información y búsqueda del mejor umbral
# ---------------------------------------------------------------------------

def ganancia_informacion(y, y_mask, impurity_func=entropy_impurity):
    """Ganancia de información de dividir y según y_mask.

    IG = I(padre) - |izq|/n * I(izq) - |der|/n * I(der)
    (el cuaderno original ponderaba un solo hijo por la proporción de clases).
    """
    y = np.asarray(y)
    y_mask = np.asarray(y_mask, dtype=bool)
    n = y.size
    a = int(y_mask.sum())
    b = n - a
    if a == 0 or b == 0:
        return 0.0
    return (impurity_func(y)
            - (a / n) * impurity_func(y[y_mask])
            - (b / n) * impurity_func(y[~y_mask]))


def buscar_mejor_information_gain_variable(x, y, criterio=entropy_impurity,
                                           min_split_samples=2, n_clases=None):
    """Mejor umbral para una variable numérica.

    Devuelve (max_gain, best_midpoint). Si el nodo tiene menos de
    min_split_samples muestras, o la variable es constante, devuelve (0.0, None).

    Da el mismo resultado que probar cada punto medio con ganancia_informacion,
    pero calcula todos los cortes a la vez con sumas acumuladas.
    y debe venir codificada como enteros 0..k-1 (ArbolDecision lo hace).
    La división es: izquierda = x <= umbral, derecha = x > umbral.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    n = x.size
    if n < min_split_samples or n < 2:
        return 0.0, None

    _, impureza_conteos = _resolver_criterio(criterio)
    if n_clases is None:
        n_clases = int(y.max()) + 1

    orden = np.argsort(x, kind="mergesort")
    x_ord = x[orden]
    y_ord = y[orden]

    # Conteos acumulados por clase: fila i = conteos de las primeras i+1 muestras
    one_hot = np.zeros((n, n_clases))
    one_hot[np.arange(n), y_ord] = 1
    acumulado = np.cumsum(one_hot, axis=0)
    total = acumulado[-1]

    # Cortes válidos: donde cambia el valor de x (entre x_ord[i] y x_ord[i+1])
    cortes = np.nonzero(x_ord[1:] > x_ord[:-1])[0]
    if cortes.size == 0:
        return 0.0, None

    izq = acumulado[cortes]
    der = total - izq
    n_izq = cortes + 1
    n_der = n - n_izq

    impureza_padre = impureza_conteos(total[None, :])[0]
    ganancias = (impureza_padre
                 - (n_izq / n) * impureza_conteos(izq)
                 - (n_der / n) * impureza_conteos(der))

    mejor = int(np.argmax(ganancias))
    max_gain = float(ganancias[mejor])
    if max_gain <= 0:
        return 0.0, None
    i = cortes[mejor]
    best_midpoint = float((x_ord[i] + x_ord[i + 1]) / 2)
    return max_gain, best_midpoint


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
