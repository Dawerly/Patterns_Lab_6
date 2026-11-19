# Лабораторна № 6

Чотири Python-файли:

- core.py — дані, обробка, Factory, Repository і PredictionService.
- train.py — навчання та порівняння моделей.
- app.py — FastAPI та схеми запитів.
- test_app.py — 8 тестів.

## Встановлення

Потрібен Python 3.11 або 3.12.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

На Windows активація: `.venv\Scripts\activate`.

Усі команди запускаються з каталогу Lab_6_simple.

## Запуск

Готова модель уже додана. Для сервера:

```sh
python -m uvicorn app:app --reload
```

Відкрити [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Там можна заповнити запит і перевірити всі маршрути.

Для повторного навчання:

```sh
python train.py
```

Для перевірки:

```sh
python -m pytest -q
```

Дані вже містяться в data/ames.csv, тому інтернет для навчання не потрібен. Розбиття — 70/15/15 із random_state=42. Налаштування — у configs/config.yaml. Нові ознаки, заповнення пропусків, кодування, відбір, масштабування та модель зберігаються одним конвеєром.

Звіт — docs/report.pdf та docs/report.md, діаграма — docs/uml.png. Виконаний аналіз train — notebooks/01_eda.ipynb. Готові результати — docs/results.csv. Тест перевіряє збіг п'яти прогнозів після завантаження моделі.

Фінальна модель — RandomForest, MAE test 17266,87 USD, R² test 0,9049. Модель призначена для навчального набору Ames за 2006–2010 роки.

Джерело даних: [OpenML 42165](https://www.openml.org/d/42165).
