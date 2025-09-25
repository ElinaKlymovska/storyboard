# video-shots

CLI для створення серії скріншотів із відео через заданий інтервал.

## Встановлення залежностей

```
pip install -r requirements.txt
```

Знадобляться OpenCV, NumPy, Pillow та Replicate.

## Налаштування AI аналізу

Для використання функції `--analyze` потрібен API токен Replicate:

```bash
export REPLICATE_API_TOKEN=your_token_here
```

Отримати токен можна на [replicate.com](https://replicate.com/collections/vision-models).

## Використання

```
python video_shots.py <video_path> -n <interval>
```

Основні прапорці:

- `-n/--interval` — інтервал між скріншотами (200ms, 0.5s, 00:00:03.5 тощо).
- `--format png|jpg` — формат вихідних файлів (дефолт: png).
- `--outdir <path>` — власна вихідна папка.
- `--prefix <name>` — префікс для назв файлів.
- `--start/--end` — обмеження діапазону часу.
- `--rotate 0|90|180|270` — примусове обертання кадрів.
- `--pdf <path>` — збір усіх скрінів у один PDF.
- `--time-precision auto|ms|sec` — формат часу у підписі.
- `--max-frames <N>` — верхня межа кількості кадрів.
- `--analyze` — аналізувати кожен скріншот за допомогою AI vision model.
- `--analysis-prompt <text>` — промпт для аналізу зображень.

Приклади:

```
# Базове використання
python video_shots.py input.mp4 -n 0.5s --outdir shots --format jpg

# З AI аналізом зображень
python video_shots.py input.mp4 -n 0.2s --analyze --analysis-prompt "Опиши емоції та дії людей на зображенні"
```

## Логи

- Успішне виконання: `Готово. Збережено 6 файлів у 'output/input_shots'.`
- Надто малий інтервал: `ℹ️ Інтервал 0.005000s менший за 1/FPS (0.033333s). Можливе повторення кадрів.`
- Відсутній файл: `video_shots.py: error: Файл відео 'missing.mp4' не існує`

## Тестові сценарії

- 10-секундне відео при `-n 2s` → 6 скріншотів для t = 0..10 включно.
- 10-секундне відео при `-n 200ms` → ~51 скріншот і коректні підписи з мілісекундами.
- Відео з FPS=30 при `-n 5ms` → попередження про обмеження FPS, можливі дублікати кадрів.

