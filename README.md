# Machine_Learning_project

https://github.com/krodgall/Machine_Learning_project

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

Se ha establecido un flujo de trabajo riguroso probando modelos de complejidad incremental. Todos los modelos han compartido el mismo preprocesamiento (ColumnTransformer y Pipeline) para asegurar una comparación justa.

- **DummyRegressor**: Modelo base que predice la mediana. Sirve como línea base (baseline) para saber si nuestros modelos realmente aprenden algo.   - **Regresión Lineal**: Modelo lineal clásico. Se observó underfitting (incapacidad de capturar la complejidad del mercado).
- **Random Forest Regressor**: Mostró un excelente rendimiento en entrenamiento (`$R^2 \approx 1.0$`), pero sufría de un sobreajuste (overfitting) severo, generalizando peor en validación y siendo computacionalmente muy costoso debido al tamaño del dataset.
- **XGBoost Regressor (eXtreme Gradient Boosting)**: El candidato final seleccionado. Demostró ser más eficiente y robusto que el Random Forest, manejando mejor la varianza y ofreciendo el mejor equilibrio entre sesgo y varianza.

---

## 6. Selección de hiperparámetros

Dada la magnitud del dataset (más de 760.000 registros), una búsqueda exhaustiva (GridSearchCV) resultaba computacionalmente inviable. Se optó por RandomizedSearchCV sobre el modelo ganador (XGBoost), utilizando validación cruzada de 3 folds (CV=3).

Estrategia de optimización: Se buscó maximizar el rendimiento minimizando el RMSE (Root Mean Squared Error) sobre la variable objetivo logarítmica (`log_price`).

Se ajustaron los siguientes hiperparámetros clave para controlar la complejidad y la velocidad de aprendizaje:

- **n_estimators** (`100, 300, 500`): Número de árboles de decisión.

- **learning_rate** (`0.01, 0.05, 0.1`): Tasa de aprendizaje (paso de actualización de los pesos).

- **max_depth** (`3, 5, 7, 10`): Profundidad máxima para controlar el sobreajuste.

- **subsample y colsample_bytree**: Fracciones de datos y columnas usadas por árbol para añadir aleatoriedad y robustez.

La búsqueda aleatoria convergió en una configuración robusta (500 estimadores, profundidad 10 y learning rate 0.05) que redujo significativamente el error en el conjunto de validación.

---

## 7. Resultados y evaluación final

Tras seleccionar el mejor modelo de XGBoost y re-entrenarlo con los mejores hiperparámetros, se procedió a la evaluación definitiva utilizando el conjunto de TEST (datos nunca vistos por el modelo). Las predicciones (realizadas en escala logarítmica) fueron transformadas nuevamente a escala real (euros/dólares) para su interpretación de negocio.

Métricas finales en TEST:

- **MAE** (Error Medio Absoluto): $2,529.73

Interpretación: De media, el modelo se equivoca en aproximadamente 2.500$ al tasar un coche. Dado que el rango de precios varía desde coches muy baratos a lujo, es una precisión muy competitiva.

- **RMSE** (Raíz del Error Cuadrático Medio): $4,345.06

Interpretación: Esta métrica es más sensible a grandes errores. La diferencia con el MAE indica que existen algunos casos atípicos (coches de muy alta gama o con características inusuales) donde el error es mayor.

- **R²** (Coeficiente de determinación): 0.9475

Interpretación: El modelo es capaz de explicar el 94.7% de la variabilidad del precio de los coches en el mercado.

---

## 8. Conclusiones

El proyecto ha permitido construir un sistema de tasación automática robusto y escalable. Las principales conclusiones extraídas son:

- **Superioridad del Boosting**: Algoritmos como XGBoost superan claramente a enfoques tradicionales (Regresión Lineal) y a bagging clásico (Random Forest) en este tipo de datos tabulares, reduciendo el error a la mitad respecto a los modelos base.

- **Importancia de la Transformación Logarítmica**: La variable price presenta una distribución muy asimétrica. Trabajar con log_price ha sido fundamental para estabilizar el entrenamiento y mejorar la convergencia del modelo.

- **Eficacia del Modelo**: Con un MAE de 2,500 €, el modelo es una herramienta útil para dar referencias de precio rápidas en el mercado de segunda mano, aunque debe supervisarse en vehículos de lujo o coleccionismo (donde se concentran los mayores errores).


Como trabajo futuro podrían explorarse:

- Implementar técnicas de NLP (Procesamiento de Lenguaje Natural) para extraer información de la descripción textual de los anuncios, que actualmente no se usa.

- Desarrollar una interfaz web (Streamlit) para poner el modelo en producción.

- Explorar el ensamblaje (Stacking) de XGBoost con otros modelos como LightGBM o CatBoost para arañar puntos extra de precisión.

---

## 9. Autores

- Carlos Rodríguez Gallardo.
    - c.rodrigurezgallardo@cunef.edu

- Juan José Gómez Romero.
    - juanjose.gomez@cunef.edu

- Jaime Rodríguez Gross.
    - j.rodriguezgross@cunef.edu