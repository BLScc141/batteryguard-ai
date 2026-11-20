# BatteryGuard AI
Proyecto final de Fundamentos de la IA (LITIID-73), UT Cancún.

## Reproducir localmente

Requisitos: Python 3.15.2 y dependencias de `requirements.txt`.

```bash
pip install -r requirements.txt
python src/data/preprocess.py
python src/models/train.py --data data/processed/ev_battery_failure_clean.csv --output .
python src/models/predict.py
python src/models/ablation.py --data data/processed/ev_battery_failure_clean.csv --output .
```

## Contrato de API para integración

La implementación HTTP y la conexión con React están a cargo de otro integrante. El contrato esperado es `POST http://127.0.0.1:8000/predict`, con la solicitud y respuesta de abajo. Al incorporar el módulo de API, el comando de arranque previsto desde la raíz del repositorio es:

```bash
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

Cuando exista la implementación HTTP, la documentación interactiva debe estar en `http://127.0.0.1:8000/docs` y el estado del servicio en `http://127.0.0.1:8000/health`.

El endpoint `POST /predict` debe recibir `preset_id` y un objeto opcional `overrides`. Los overrides permitidos son `cell_voltage_avg`, `cell_temperature_max` y `terrain_type`; la API debe validar tipos y rangos antes de inferir.

Para permitir el origen del frontend, la implementación debe configurar CORS para el dominio que acuerde el equipo. En desarrollo local, el origen esperado es `http://localhost:5173` (o `http://127.0.0.1:5173`).

### Contrato de `POST /predict`

URL local: `http://127.0.0.1:8000/predict` (Swagger: `http://127.0.0.1:8000/docs`). El cuerpo requiere un `preset_id` que exista en `models/presets.json`; `overrides` es opcional.

```json
{
  "preset_id": "EV109069",
  "overrides": {
    "cell_voltage_avg": 3.649,
    "cell_temperature_max": 60.0,
    "terrain_type": "Flat"
  }
}
```

Validaciones de overrides:

- `cell_voltage_avg`: número entre 3.06 y 3.85 V.
- `cell_temperature_max`: número entre −4.95 y 86.98 °C.
- `terrain_type`: `Flat`, `Coastal`, `Mountainous`, `Hilly`, `Desert` o `Desconocido`.
- No se admiten otras claves en `overrides`. Las entradas inválidas responden HTTP 422.

Ejemplo real observado en Swagger para esa solicitud (la latencia varía entre ejecuciones):

```json
{
  "prediccion": 0,
  "etiqueta": "Sin falla detectada",
  "score_riesgo": 0.103261,
  "nivel_riesgo": "bajo",
  "recomendaciones": [
    "Sin alerta del clasificador; mantener el plan de seguimiento preventivo."
  ],
  "rsl_meses": null,
  "tiempo_ms": 113.397,
  "modelo": "Logistic Regression regularized C=0.1",
  "umbral_clasificacion": 0.318373
}
```

`score_riesgo` es el score producido por el clasificador; no se ha validado como probabilidad calibrada.

## Ablación de variables

Para registrar en `results/experiments.csv` la comparación entre las 60 features y la variante conservadora de 52 features, ejecuta:

```bash
python src/models/ablation.py --data data/processed/ev_battery_failure_clean.csv --output .
```

Ambos escenarios usan split estratificado 80/20, semilla 42, validación cruzada de cinco folds y umbral elegido por F1 máximo en predicciones OOF dentro de train. El test no se usa para elegir el umbral.

## Limitaciones conocidas

- El dataset es sintético; la etiqueta `battery_failure` se genera con factores simulados. La ficha no publica la fórmula exacta ni los factores usados.
- El modelo clasifica la etiqueta sintética del dataset. No se ha validado que anticipe fallas futuras en vehículos o baterías reales.
- El modelo elegido es una Regresión Logística regularizada (`C=0.1`). En test obtuvo F1 positivo 0.7131, precisión 0.6731, recall 0.7581 y ROC-AUC 0.9711. No alcanzó la meta anterior F1 >0.85.
- `score_riesgo` no es una probabilidad calibrada. La salida es binaria (`bajo`/`alto`); no hay estimador RSL y `rsl_meses` queda en `null`.
- La ficha de Kaggle describe licencia CC BY-SA 4.0. Fuente: [EV Battery Health Prediction Dataset (20K)](https://www.kaggle.com/datasets/srisyra02/ev-battery-health-prediction-dataset-20k), owner Srimathy Sivanessan. Al compartir el dataset, conserva la atribución y la licencia.

## Datos
- Archivo: `data/raw/ev_battery_failure_prediction_Dataset.csv`; procesado: `data/processed/ev_battery_failure_clean.csv`.
- Volumen medido: 20,000 filas × 70 columnas. Target `battery_failure`: 18,616 clase 0 (93.08%) y 1,384 clase 1 (6.92%).
