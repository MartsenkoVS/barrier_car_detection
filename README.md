# Barrier Car Detection

Система детекции, трекинга и распознавания гос. номеров автомобилей в зоне перед шлагбаумом. 

## Структура проекта

```
├── app.py               # Запуск приложения FastAPI
├── src/                 # Логика приложения
│   ├── roi.py           # Описание полигонов, расчет нахождения авто в зоне
│   ├── ocr.py           # Детекция и распознавание гос. номеров
│   └── pipeline.py      # Основной пайплайн приложения
├── templates/           # Jinja2 (index.html)
├── models/              # Весы моделей детекции
├── services             # Стримминг видео с результатами обработки
└── routers              # Маршрутизация FastAPI
```
