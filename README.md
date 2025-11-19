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
  - `engine` – motor del coche  
  - `transmission` – tipo de transmisión  
  - `drivetrain` – tipo de tracción  
  - `fuel_type` – tipo de combustible  
  - `mpg` – rendimiento del coche en millas por galón  
  - `exterior_color` – color exterior  
  - `interior_color` – color interior  
  - `accidents_or_damage` – si ha tenido accidentes o daños  
  - `one_owner` – si ha tenido un único propietario  
  - `personal_use_only` – si se ha usado sólo para fines personales  
  - `seller_name` – nombre del vendedor  
  - `seller_rating` – valoración del vendedor  
  - `driver_rating` – valoración del coche realizada por conductores  
  - `driver_reviews_num` – número de reseñas del coche  
  - `price_drop` – reducción del precio respecto al precio inicial  

  Además, se crearán features derivadas (por ejemplo, antigüedad del coche, rangos de kilometraje, etc.).

- **Variable objetivo (target):**

  - `price` – precio del coche

Opcionalmente, también se podría plantear una variante de clasificación, por ejemplo categorizando el precio en rangos (barato, medio, caro), pero el enfoque principal del proyecto será en principio la regresión.

---

## 4. Autores

- Carlos Rodríguez Gallardo.
    - c.rodrigurezgallardo@cunef.edu

- Juan José Gómez Romero.
    - juanjose.gomez@cunef.edu

- Jaime Rodríguez Gross.
    - j.rodriguezgross@cunef.edu