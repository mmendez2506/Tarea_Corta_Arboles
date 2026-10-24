# Tarea Corta 2 — Árboles de Decisión

**Curso:** Inteligencia Artificial — TEC
**Integrantes:** María Felix Mendez Abarca, Cristhian Rivas y Jozafath Perez

Árbol de decisión implementado desde cero con NumPy, validación cruzada propia y comparación de 10 combinaciones de hiperparámetros sobre el dataset `adult-census`.

## Estructura

| Archivo | Responsable | Contenido |
| --- | --- | --- |
| `src/arbol.py` | Persona A | `Nodo`, `ArbolDecision`, impurezas y ganancia de información |
| `src/datos.py` | Persona B | Carga del CSV y split train/test manual |
| `src/evaluacion.py` | Persona B | Métricas con NumPy y `validacion_cruzada` |
| `experimentos.py` | Persona C | Las 10 combinaciones de hiperparámetros |
| `taller.ipynb` | Persona C | Resultados, análisis y evaluación en test |
| `tests/test_arbol.py` | Persona A | Pruebas del árbol |
| `tests/test_evaluacion.py` | Persona B | Pruebas de datos, métricas y validación cruzada |
| `data/adult-census.csv` | — | Dataset del taller |

Correr las pruebas desde la raíz del repo: `python -m tests.test_arbol` y `python -m tests.test_evaluacion`

## Cómo usar el árbol

```python
from src.arbol import ArbolDecision

modelo = ArbolDecision(max_depth=6, min_split_samples=20, criterio="gini")  # o "entropia"
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
```

- **`X`** puede ser el DataFrame tal cual, con las columnas de texto (`workclass`, `education`, …). El árbol las codifica solo y quita los espacios. No hace falta one-hot ni codificar a mano.
- **`y`** puede ser 0/1 o texto. `predict` devuelve el mismo tipo que recibió.
- Si `X` es DataFrame, `predict` toma las columnas **por nombre** y en el orden de `fit`: no importa el orden en que lleguen, pero no puede faltar ninguna.
- Las categorías se dividen en dos grupos y el mejor agrupamiento se recalcula en cada nodo.
- **`max_depth`**: cantidad máxima de variables que se exploran de la raíz a una hoja (= profundidad). `None` es sin límite.
- **`min_split_samples`**: mínimo de muestras para que un nodo se pueda dividir (≥ 2).

Con el dataset completo entrena en menos de 1 s con profundidad limitada y unos 4 s sin límite.

## Datos y evaluación

```python
from src.datos import cargar_datos, train_test_split_manual
from src.evaluacion import validacion_cruzada, f1

X, y = cargar_datos()                                   # X: DataFrame, y: NumPy 0/1 (1 = >50K)
X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2)   # SEMILLA = 42
cv = validacion_cruzada(ArbolDecision, {"max_depth": 10, "min_split_samples": 300, "criterio": "gini"},
                        X_train, y_train, k=5)
```

- `cargar_datos` quita el espacio inicial de las etiquetas y del texto, y pasa la clase a 0/1. Los `?` quedan como una categoría más. Las categóricas se dejan como texto porque el árbol las agrupa mejor que un ordinal o un one-hot (F1 0.667 contra 0.644 y 0.660 en validación cruzada).
- `train_test_split_manual` baraja los índices con `np.random.default_rng(semilla).permutation`.
- `accuracy`, `precision`, `recall` y `f1` toman 1 como clase positiva y devuelven 0 en vez de dividir entre cero.
- `validacion_cruzada` parte los datos en `k` folds excluyentes con la misma semilla, crea un modelo nuevo por fold y devuelve un dict plano:

```python
{"accuracy_media": 0.857, "accuracy_desv": 0.003, "precision_media": ..., "precision_desv": ...,
 "recall_media": ..., "recall_desv": ..., "f1_media": ..., "f1_desv": ...}
```

### Para la Persona C (experimentos y análisis)

| Método | Devuelve | Para qué |
| --- | --- | --- |
| `modelo.resumen()` | dict con hiperparámetros, `profundidad_real`, `hojas`, `nodos`, `tiempo_entrenamiento_s` | Una fila de la tabla de resultados |
| `modelo.importancia_variables()` | dict variable → importancia (suma 1), de mayor a menor | Explicar qué variables usa el modelo |
| `modelo.imprimir(max_niveles=3)` | Imprime los primeros niveles del árbol | Mostrar las reglas en el notebook |
| `modelo.predict_proba(X)` | Probabilidad de cada clase por fila | Opcional, por ejemplo para una curva ROC |

`hojas` y `profundidad_real` sirven para hablar de sobreajuste: más hojas significa un modelo más complejo.

## Experimentos

`python experimentos.py` corre las 10 combinaciones con validación cruzada de 5 folds (~1 min) e imprime la tabla y el modelo escogido. El notebook usa las mismas funciones (`correr_experimentos` y `escoger_mejor`).

`experimentos.py` espera que `validacion_cruzada` devuelva un dict plano con media y desviación de cada métrica:

```python
{"accuracy_media": 0.857, "accuracy_desv": 0.003, "precision_media": ..., "precision_desv": ...,
 "recall_media": ..., "recall_desv": ..., "f1_media": ..., "f1_desv": ...}
```

`src/datos.py` y `src/evaluacion.py` tienen por ahora una versión provisional de la Persona C para poder correr los experimentos; se reemplazan por la de la Persona B al mergear `evaluacion`. Después de eso hay que volver a correr el notebook.

## Flujo de Git

Una rama por persona (`arbol`, `evaluacion`, `experimentos`) y un pull request a `main` cuando la parte funcione. Solo la Persona C edita `taller.ipynb`.
