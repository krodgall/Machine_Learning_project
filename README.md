# Machine_Learning_project

## 1. Descripción del problema de negocio

El dataset contiene información detallada de anuncios de coches de segunda mano.  
Nuestro objetivo es estudiar qué factores explican el precio de un vehículo y utilizar estos patrones para construir un modelo que prediga el precio de un coche a partir de sus características.

Este problema es relevante, por ejemplo, para:

- Plataformas de compraventa de coches de segunda mano, que necesitan sugerir precios razonables a los vendedores.
- Concesionarios y particulares que quieren estimar el valor de mercado de un vehículo de forma rápida y basada en datos.
- Sistemas de recomendación que podrían ordenar anuncios según relación calidad–precio.

Frente a soluciones tradicionales (tasaciones manuales, reglas fijas, experiencia individual), un enfoque basado en Aprendizaje Automático permite:

- Incorporar información histórica de cientos de miles de vehículos.
- Ajustar el precio automáticamente a cambios en el mercado (marcas, modelos, años, etc.).
- Reducir sesgos y subjetividad en la valoración.

---

## 2. Origen del dataset

En el presente proyecto se utiliza el siguiente dataset de Kaggle:

Used Cars Dataset:
https://www.kaggle.com/datasets/andreinovikov/used-cars-dataset

El dataset contiene información sobre 762.091 coches de segunda mano extraídos de cars.com.  
Los datos fueron recopilados en abril de 2023.

**Ventajas del dataset:**

- Gran cantidad de observaciones: permite entrenar y evaluar modelos de forma robusta.
- Variables variadas (técnicas, uso del vehículo, valoraciones, etc.).
- Datos relativamente recientes (2023), representativos del mercado actual de segunda mano.

**Desventajas / limitaciones:**

- El mercado está centrado en Estados Unidos, por lo que los resultados pueden no generalizarse directamente a otros países.
- No se incluyen variables económicas que también podrían influir en el precio.

> **Nota:** Los datos no se incluyen en el repositorio por las restricciones de tamaño de GitHub.

---

## 3. Planteamiento del problema en términos de Aprendizaje Automático

El problema se formula como un problema de aprendizaje supervisado de regresión:

- **Tipo de aprendizaje:**
  - Aprendizaje supervisado.
  - Tarea principal: regresión (predicción de una variable continua).

- **Vector de atributos (features):**
  Se utilizarán, entre otras, las siguientes variables del dataset (tras la limpieza y transformación adecuadas):

  - `manufacturer` – nombre del fabricante del coche  
  - `model` – nombre del modelo del coche  
  - `year` – año de fabricación del coche  
  - `mileage` – número de millas recorridas desde su fabricación   
  - `engine_cylindres` – número de cilindros del coche
  - `engine_liters` – número de litros de potencia del coche
  - `transmission_simple` – tipo de transmisión  
  - `drivetrain_simple` – tipo de tracción  
  - `fuel_simple` – tipo de combustible  
  - `mpg_max` y `mpg_min` – rendimiento máximo y mínimo del coche en millas por galón   
  - `accidents_or_damage` – si ha tenido accidentes o daños  
  - `one_owner` – si ha tenido un único propietario  
  - `personal_use_only` – si se ha usado sólo para fines personales   
  - `seller_rating` – valoración del vendedor
  - `driver_rating` – valoración del coche realizada por conductores
  - `driver_reviews_num` – número total de reviews que tiene el anuncio

  Además, se han creado features derivadas:

  - `car_age` – antigüedad del coche
  - `mileage_per_year` – número de millas por años de antigüedad
  - `mpg_mean` – media del rendimiento del coche en millas por galón
  - `seller_rating_cat` – valoración del vendedor categorizada (1-2,2-3, ...)
  - `driver_rating_cat` – valoración del coche realizada por conductores categorizada (1-2,2-3, ...)
  - `log_mileage` – logaritmo del número de millas recorridas desde su fabricación
  - `engine_power_index` – `engine_liters` * `engine_cylindres`
  - `log_driver_reviews_num` – logaritmo del número total de reviews que tiene el anuncio

- **Variable objetivo (target):**

  - `price` – precio del coche
  - `log_price` – logaritmo del precio del coche

---

## 4. Separación del dataset

El dataset se ha dividido en tres subconjuntos:

- **Train**: 60% – usado para entrenar los modelos.
- **Validación**: 20% – usado para comparar modelos y seleccionar
  hiperparámetros.
- **Test**: 20% – usado sólo al final para evaluar el modelo elegido.

La separación se ha realizado de forma aleatoria, fijando una semilla
(`random_state=42`) para asegurar reproducibilidad.

---

## 5. Modelos y algoritmos candidatos

Se ha utilizado como **modelo base** un `DummyRegressor` (predice la mediana del precio), que sirve como referencia mínima.

Como **modelos candidatos** se han entrenado:

1. **Regresión lineal** con todas las variables transformadas.
2. **Random Forest Regressor**.
3. **Gradient Boosting Regressor**.

Todos los modelos comparten el mismo pipeline de preprocesado (`ColumnTransformer`), para asegurar que la comparación sea justa.

Las métricas de evaluación utilizadas han sido:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (coeficiente de determinación)

---

## 6. Selección de hiperparámetros

Sobre el modelo candidato con mejor rendimiento en validación (p. ej., Random Forest) se ha aplicado `RandomizedSearchCV` con validación cruzada de 3 folds, buscando sobre los siguientes hiperparámetros:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- etc.

El criterio de selección ha sido maximizar el score negativo de RMSE (`scoring = 'neg_root_mean_squared_error'`).

---

## 7. Resultados y evaluación final

Comparando los modelos en el conjunto de validación se observa que:

- El modelo base (Dummy) obtiene un MAE y un RMSE elevados, lo que indica que no es útil más allá de servir como referencia.
- La **regresión lineal** mejora significativamente las métricas respecto al Dummy, pero presenta ciertas limitaciones para capturar relaciones no lineales.
- El modelo de **árboles (Random Forest / Gradient Boosting)** obtiene los mejores resultados en validación.

El modelo elegido final es: **\<nombre del modelo elegido\>** con los siguientes resultados aproximados en el conjunto de test:

- MAE (test): **\<valor\>**
- RMSE (test): **\<valor\>**
- R² (test): **\<valor\>**

*(Rellenar con los resultados reales.)*

---

## 8. Conclusiones

- Factores como la **antigüedad (`car_age`)**, el **kilometraje (`mileage`)** y ciertas características técnicas (`engine`, `drivetrain`, `fuel_type`) tienen un impacto claro en el precio.
- Los modelos de tipo **ensemble de árboles** capturan mejor las interacciones entre variables y ofrecen una mejora significativa frente a la regresión lineal.
- El modelo final puede utilizarse para estimar precios razonables de coches de segunda mano en el mercado de EEUU, con las limitaciones de generalización a otros mercados.

Como trabajo futuro podrían explorarse:

- Otros modelos más avanzados (XGBoost, LightGBM).
- Mayor trabajo de selección de variables y tratamiento de outliers.
- Incorporar información económica externa (macro datos, etc.).

---

## 9. Autores

- Carlos Rodríguez Gallardo.
    - c.rodrigurezgallardo@cunef.edu

- Juan José Gómez Romero.
    - juanjose.gomez@cunef.edu

- Jaime Rodríguez Gross.
    - j.rodriguezgross@cunef.edu