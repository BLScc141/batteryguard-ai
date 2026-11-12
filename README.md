# BatteryGuard AI
Proyecto final de Fundamentos de la IA (LITIID-73), UT Cancún.

## Reproducir 

Python 3.15.1. Desde la raíz del repo: pip install -r requirements.txt python src/data/preprocess.py; python src/models/train.py --data data/processed/ev_battery_failure_clean.csv --output . python src/models/predict.py
## Datos
- Fuente: dataset de información histórica sobre baterías de vehículos eléctricos, proporcionado por el profesor Carlos Andre Castro Rodriguez en el canal Proyectos (Teams) del grupo LITIID-73.
- Autor original y licencia: no especificados por el docente.
- Volumen: 20,000 filas x 70 columnas. Target: `battery_failure` (93.08% clase 0, 6.92% clase 1).
- `data/raw/`: original sin modificar. `data/processed/`: versión limpia generada desde el notebook.