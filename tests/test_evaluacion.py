# ==============================
# Tarea Corta 2 — Árboles de Decisión — Inteligencia Artificial
# Integrantes: María Felix Mendez Abarca, Cristhian Rivas y Jozafath Perez
# Descripción: Pruebas de src/datos.py y src/evaluacion.py. Usan un modelo que
#              predice la clase mayoritaria, así que no dependen del árbol.
#              Ejecutar desde la raíz:  python -m tests.test_evaluacion
# ==============================
import numpy as np

from src.datos import cargar_datos, train_test_split_manual
from src.evaluacion import accuracy, f1, precision, recall, validacion_cruzada


class ClaseMayoritaria:
    """Predice siempre la clase más común del train. Guarda qué filas vio."""
    vistos = []

    def __init__(self, **params):
        pass

    def fit(self, X, y):
        self.clase = int(np.bincount(y).argmax())
        ClaseMayoritaria.vistos.append(("fit", X[:, 0].copy()))

    def predict(self, X):
        ClaseMayoritaria.vistos.append(("predict", X[:, 0].copy()))
        return np.full(len(X), self.clase)


def test_metricas_a_mano():
    # tp=2 (filas 0 y 4), fp=1 (fila 3), fn=1 (fila 1), tn=1 (fila 2)
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 1]
    assert accuracy(y_true, y_pred) == 3 / 5
    assert precision(y_true, y_pred) == 2 / 3
    assert recall(y_true, y_pred) == 2 / 3
    assert abs(f1(y_true, y_pred) - 2 / 3) < 1e-12


def test_metricas_sin_positivos_no_dividen_por_cero():
    assert precision([1, 0], [0, 0]) == 0.0
    assert recall([0, 0], [1, 0]) == 0.0
    assert f1([1, 0], [0, 0]) == 0.0


def test_split_reproducible_y_sin_repetidos():
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100) % 2
    X_tr, X_te, y_tr, y_te = train_test_split_manual(X, y, test_size=0.2)
    assert len(X_te) == 20 and len(X_tr) == 80
    assert set(X_tr[:, 0]).isdisjoint(X_te[:, 0])
    assert set(X_tr[:, 0]) | set(X_te[:, 0]) == set(range(100))
    assert (y_te == X_te[:, 0] % 2).all()  # X y y se barajan juntos
    assert (train_test_split_manual(X, y)[1] == X_te).all()  # misma semilla, mismo split


def test_validacion_cruzada_folds_excluyentes():
    X = np.arange(103).reshape(-1, 1)
    y = (np.arange(103) < 30).astype(int)
    ClaseMayoritaria.vistos = []
    r = validacion_cruzada(ClaseMayoritaria, {}, X, y, k=5)
    validaciones = [filas for paso, filas in ClaseMayoritaria.vistos if paso == "predict"]
    entrenos = [filas for paso, filas in ClaseMayoritaria.vistos if paso == "fit"]
    assert len(validaciones) == 5
    # Cada fila está en exactamente un fold de validación
    assert sorted(np.concatenate(validaciones)) == list(range(103))
    # Ninguna fila de validación se usó para entrenar ese modelo
    for tr, val in zip(entrenos, validaciones):
        assert set(tr).isdisjoint(val) and len(tr) + len(val) == 103
    # La clase mayoritaria (0) nunca predice 1: precision, recall y F1 son 0
    assert set(r) == {f"{m}_{s}" for m in ("accuracy", "precision", "recall", "f1")
                      for s in ("media", "desv")}
    assert abs(r["accuracy_media"] - 73 / 103) < 0.02 and r["f1_media"] == 0.0


def test_cargar_datos():
    X, y = cargar_datos()
    assert len(X) == len(y) == 48842 and "class" not in X.columns
    assert set(np.unique(y)) == {0, 1} and abs(y.mean() - 0.239) < 0.001
    assert not X["workclass"].str.startswith(" ").any()


if __name__ == "__main__":
    pruebas = [f for nombre, f in sorted(globals().items()) if nombre.startswith("test_")]
    for prueba in pruebas:
        prueba()
        print(f"OK  {prueba.__name__}")
    print(f"\n{len(pruebas)} pruebas pasaron")
