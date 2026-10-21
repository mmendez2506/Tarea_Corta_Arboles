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
| `data/adult-census.csv` | — | Dataset del taller |

Correr las pruebas desde la raíz del repo: `python -m tests.test_arbol`

## Cómo usar el árbol

```python
from src.arbol import ArbolDecision

modelo = ArbolDecision(max_depth=6, min_split_samples=20, criterio="gini")  # o "entropia"
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
```

- **`X`** puede ser el DataFrame tal cual, con las columnas de texto (`workclass`, `education`, …). El árbol las codifica solo y quita los espacios. No hace falta one-hot ni codificar a mano.
- **`y`** puede ser 0/1 o texto. `predict` devuelve el mismo tipo que recibió.
- **`max_depth`**: cantidad máxima de variables que se exploran de la raíz a una hoja (= profundidad). `None` es sin límite.
- **`min_split_samples`**: mínimo de muestras para que un nodo se pueda dividir (≥ 2).

Con el dataset completo entrena en menos de 1 s con profundidad limitada y unos 4 s sin límite.

### Para la Persona B (datos y evaluación)

`cargar_datos` solo necesita separar la clase y pasarla a 0/1:

```python
df = pd.read_csv(ruta)
y = (df["class"].str.strip() == ">50K").astype(int).to_numpy()   # 1 = gana más de 50K
X = df.drop(columns="class")                                      # el árbol acepta el texto
```

En `validacion_cruzada`, entrenar un modelo nuevo por fold:

```python
modelo = modelo_cls(**params)        # o modelo_base.clonar()
modelo.fit(X.iloc[idx_train], y[idx_train])     # .iloc si X es DataFrame
y_pred = modelo.predict(X.iloc[idx_val])
```

### Para la Persona C (experimentos y análisis)

| Método | Devuelve | Para qué |
| --- | --- | --- |
| `modelo.resumen()` | dict con hiperparámetros, `profundidad_real`, `hojas`, `nodos`, `tiempo_entrenamiento_s` | Una fila de la tabla de resultados |
| `modelo.importancia_variables()` | dict variable → importancia (suma 1), de mayor a menor | Explicar qué variables usa el modelo |
| `modelo.imprimir(max_niveles=3)` | Imprime los primeros niveles del árbol | Mostrar las reglas en el notebook |
| `modelo.predict_proba(X)` | Probabilidad de cada clase por fila | Opcional, por ejemplo para una curva ROC |

`hojas` y `profundidad_real` sirven para hablar de sobreajuste: más hojas significa un modelo más complejo.

## Flujo de Git

Una rama por persona (`arbol`, `evaluacion`, `experimentos`) y un pull request a `main` cuando la parte funcione. Solo la Persona C edita `taller.ipynb`.
