# 🌿 Clasificación de imágenes de hojas con modelo preentrenado ViT (DeiT) de Hugging Face

Este proyecto aplica técnicas de visión por computadora para clasificar imágenes de hojas de plantas utilizando modelos **Vision Transformer (ViT)**, específicamente el modelo preentrenado `facebook/deit-base-patch16-224` de HuggingFace.

Se comparan dos enfoques:

- Clasificación directa con el modelo preentrenado.
- Clasificación luego de aplicar fine-tuning sobre un dataset específico de enfermedades vegetales.

---

## 📁 Estructura del proyecto

```
.
├── results/                             # Resultados del entrenamiento
├── venv/                                # Entorno virtual (opcional)
├── .env                                 # Token de HuggingFace (no incluido en git)
├── .gitignore                           # Ignora archivos temporales y credenciales
├── README.md                            # Este archivo
├── requirements.txt                     # Requisitos base
├── requirements_torch.txt               # Requisitos de torch con soporte CUDA o CPU
├── vision-transformer-classifier.ipynb  # Notebook principal del proyecto
```

---

## 🚀 Cómo ejecutar el proyecto

### 1. Crear el entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Linux/macOS
venv\Scripts\activate     # En Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

#### Para GPU con CUDA:

```bash
pip install -r requirements_torch.txt --index-url https://download.pytorch.org/whl/cu124
```

#### Para CPU:

```bash
pip install -r requirements_torch.txt
```

### 3. Crear archivo `.env` con tu token de Hugging Face

Crea un archivo `.env` con el siguiente contenido:

```
HUGGINGFACE_HUB_TOKEN=tu_token_aqui
```

Puedes obtener el token en: https://huggingface.co/settings/tokens

---

## 🧠 Modelo utilizado

### [`facebook/deit-base-patch16-224`](https://huggingface.co/facebook/deit-base-patch16-224)

- **Tipo:** Vision Transformer (ViT) con distillation
- **Patch Size:** 16×16
- **Input Size:** 224×224
- **Preentrenado en:** ImageNet
- **Ventajas:**
  - Eficiente en datos
  - Ligero y fácil de fine-tunear

---

## 🌱 Dataset

### [`fakewave07/plant-diseases-dataset`](https://huggingface.co/datasets/fakewave07/plant-diseases-dataset)

- 38 clases (hojas sanas y con enfermedades)
- Imágenes etiquetadas
- Diseñado para diagnóstico agrícola con visión artificial

---

Si bien el dataset ya proveía splits de entrenamiento y test, se decidió combinar ambos en un único conjunto y realizar una nueva partición en entrenamiento, validación y test. Esto permitió controlar mejor el balance de clases mediante **estratificación**, asegurando representatividad en cada subconjunto.

---

## ⚙️ Fine-tuning

- **Modificaciones al modelo:**

  - Reemplazo de la capa de salida para adaptarla al número de clases (38).
  - `ignore_mismatched_sizes=True` para evitar errores.

- **Transformaciones aplicadas:**

  - Aumento de datos (flip horizontal, rotación aleatoria)
  - Preprocesamiento mediante `AutoImageProcessor` (normalización y redimensionamiento)

- **Parámetros de entrenamiento:**

```python
TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    eval_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    ...
)
```

---

## 📈 Resultados

### 🔹 Modelo preentrenado (sin fine-tuning)

| Métrica   | Valor  |
| --------- | ------ |
| Accuracy  | 0.0181 |
| F1 Score  | 0.0091 |
| Precision | 0.0208 |
| Recall    | 0.0219 |

> Desempeño casi aleatorio. El modelo no está adaptado a este tipo de imágenes.

---

### 🔹 Modelo fine-tuned

| Métrica   | Valor  |
| --------- | ------ |
| Accuracy  | 0.9937 |
| F1 Score  | 0.9829 |
| Precision | 0.9839 |
| Recall    | 0.9831 |

> Luego del fine-tuning, el modelo aprende eficazmente las clases y generaliza con precisión, incluso en clases minoritarias.

---

## 📊 Comparación resumen

| Métrica   | Preentrenado | Fine-tuned |
| --------- | ------------ | ---------- |
| Accuracy  | 0.0181       | 0.9937     |
| F1 Macro  | 0.0091       | 0.9829     |
| Precision | 0.0208       | 0.9839     |
| Recall    | 0.0219       | 0.9831     |

---

## ✅ Conclusión

El fine-tuning transforma un modelo generalista en una herramienta altamente especializada. En este caso, se logró una clasificación precisa de enfermedades en plantas con un modelo ViT que inicialmente no ofrecía ningún valor predictivo útil sin entrenamiento adicional.

Este trabajo demuestra el potencial del **transfer learning** con modelos de Hugging Face aplicados a la agricultura inteligente.

---

## 📌 Créditos

- [Hugging Face Models](https://huggingface.co/models)
- [Transformers Library](https://github.com/huggingface/transformers)
- Dataset: [fakewave07/plant-diseases-dataset](https://huggingface.co/datasets/fakewave07/plant-diseases-dataset)

## 💼 Contacto

- 🔗 [LinkedIn](https://www.linkedin.com/in/carolina-omodeo)

---

> 🌟 Gracias por visitar mi perfil. Estoy abierta a nuevas oportunidades y colaboraciones tech.
