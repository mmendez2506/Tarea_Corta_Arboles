# ==============================
# Tarea Corta 2 — Árboles de Decisión — Inteligencia Artificial
# Integrantes: María Felix Mendez Abarca, Cristhian Rivas y Jozafath Perez
# Descripción: Pruebas del árbol de decisión (src/arbol.py). Usan datos
#              sintéticos, así que corren sin el CSV. Ejecutar desde la raíz:
#                  python -m tests.test_arbol
# ==============================
import numpy as np
import pandas as pd

from src.arbol import (ArbolDecision, Nodo, buscar_mejor_information_gain_variable,
                       entropy_impurity, ganancia_informacion, gini_impurity)


def test_impurezas():
    assert gini_impurity([0, 0, 0, 0]) == 0.0
    assert abs(gini_impurity([0, 0, 1, 1]) - 0.5) < 1e-12
    assert entropy_impurity([1, 1, 1]) == 0.0
    assert abs(entropy_impurity([0, 1]) - 1.0) < 1e-12


def test_ganancia_pondera_ambos_hijos():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    mascara = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    esperado = gini_impurity(y) - 0.5 * gini_impurity(y[mascara]) - 0.5 * gini_impurity(y[~mascara])
    assert abs(ganancia_informacion(y, mascara, gini_impurity) - esperado) < 1e-12


def test_mejor_umbral_igual_a_fuerza_bruta():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 30, 300).astype(float)
    y = (x + rng.normal(0, 8, 300) > 15).astype(int)
    for criterio in (gini_impurity, entropy_impurity):
        ganancia, umbral = buscar_mejor_information_gain_variable(x, y, criterio)
        valores = np.unique(x)
        medios = (valores[:-1] + valores[1:]) / 2
        ganancias = [ganancia_informacion(y, x <= t, criterio) for t in medios]
        assert abs(ganancia - max(ganancias)) < 1e-12
        assert abs(umbral - medios[int(np.argmax(ganancias))]) < 1e-12


def test_min_split_samples_en_busqueda():
    assert buscar_mejor_information_gain_variable([1, 2, 3], [0, 1, 1], min_split_samples=5) == (0.0, None)


def test_aprende_regla_simple():
    X = np.arange(100).reshape(-1, 1)
    y = (X[:, 0] > 49).astype(int)
    modelo = ArbolDecision(max_depth=1).fit(X, y)
    assert (modelo.predict(X) == y).all()
    assert modelo.raiz.umbral == 49.5


def test_respeta_max_depth_y_min_split_samples():
    rng = np.random.default_rng(1)
    X = rng.random((500, 4))
    y = rng.integers(0, 2, 500)
    assert ArbolDecision(max_depth=3).fit(X, y).profundidad() <= 3
    assert ArbolDecision(max_depth=0).fit(X, y).cantidad_hojas() == 1
    modelo = ArbolDecision(max_depth=None, min_split_samples=100).fit(X, y)
    pila = [modelo.raiz]
    while pila:
        n = pila.pop()
        if not n.es_hoja():
            assert n.cantidad_muestras >= 100
            pila += [n.izquierda, n.derecha]


def test_dataframe_con_texto_y_etiquetas_con_espacio():
    df = pd.DataFrame({
        "age": [25, 38, 52, 28, 45, 61, 33, 40],
        "workclass": [" Private", " Self-emp", " Private", " ?", " Gov", " Self-emp", " Private", " Gov"],
    })
    y = np.array([" <=50K", " >50K", " >50K", " <=50K", " >50K", " >50K", " <=50K", " <=50K"])
    modelo = ArbolDecision(max_depth=3).fit(df, y)
    pred = modelo.predict(df)
    assert set(pred) <= {" <=50K", " >50K"}
    assert (pred == y).mean() >= 0.75
    # Una categoría nueva no rompe predict
    nuevo = pd.DataFrame({"age": [30], "workclass": [" Never-worked"]})
    modelo.predict(nuevo)


def test_predict_proba_e_importancia():
    rng = np.random.default_rng(2)
    X = rng.random((400, 3))
    y = (X[:, 1] > 0.5).astype(int)
    modelo = ArbolDecision(max_depth=4).fit(X, y)
    proba = modelo.predict_proba(X)
    assert proba.shape == (400, 2) and np.allclose(proba.sum(axis=1), 1)
    importancia = modelo.importancia_variables()
    assert list(importancia)[0] == "x[1]"
    assert abs(sum(importancia.values()) - 1) < 1e-9


def test_clonar_y_resumen():
    modelo = ArbolDecision(max_depth=2, min_split_samples=10, criterio="entropia")
    copia = modelo.clonar()
    assert copia.get_params() == modelo.get_params() and copia.raiz is None
    modelo.fit(np.random.rand(50, 2), np.random.randint(0, 2, 50))
    r = modelo.resumen()
    assert {"max_depth", "profundidad_real", "hojas", "nodos", "tiempo_entrenamiento_s"} <= set(r)


def test_errores_claros():
    for mal in (dict(max_depth=-1), dict(min_split_samples=1), dict(criterio="otro")):
        try:
            ArbolDecision(**mal)
        except ValueError:
            continue
        raise AssertionError(f"No falló con {mal}")
    try:
        ArbolDecision().predict([[1, 2]])
    except RuntimeError:
        pass
    else:
        raise AssertionError("predict sin fit debería fallar")


def test_predict_usa_columnas_por_nombre():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"age": rng.integers(17, 90, 300), "hours": rng.integers(1, 99, 300)})
    y = (df["age"] > 40).astype(int).to_numpy()
    modelo = ArbolDecision(max_depth=3).fit(df, y)
    assert (modelo.predict(df[["hours", "age"]]) == modelo.predict(df)).all()
    try:
        modelo.predict(df[["hours"]])
    except ValueError:
        pass
    else:
        raise AssertionError("Debería fallar si falta una columna")


if __name__ == "__main__":
    pruebas = [f for nombre, f in sorted(globals().items()) if nombre.startswith("test_")]
    for prueba in pruebas:
        prueba()
        print(f"OK  {prueba.__name__}")
    print(f"\n{len(pruebas)} pruebas pasaron")
