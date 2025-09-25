# Налаштування Notion інтеграції

## Отримання Notion API токена

1. Перейдіть на [notion.so/my-integrations](https://notion.so/my-integrations)
2. Натисніть "New integration"
3. Заповніть назву (наприклад, "Video Storyboard")
4. Виберіть workspace
5. Скопіюйте "Internal Integration Token"

## Налаштування змінних середовища

```bash
export NOTION_API_TOKEN=your_notion_token_here
```

## Отримання ID батьківської сторінки

1. Відкрийте Notion сторінку, куди хочете додати storyboard
2. Скопіюйте URL сторінки
3. ID - це частина після останнього слешу, без дефісів

Приклад:
- URL: `https://notion.so/My-Workspace/My-Page-1234567890abcdef`
- ID: `1234567890abcdef`

## Використання

```bash
# Створити storyboard в Notion
python video_shots.py input.mp4 -n 0.5s --analyze --notion 1234567890abcdef

# З PDF та Notion одночасно
python video_shots.py input.mp4 -n 0.2s --analyze --pdf storyboard.pdf --notion 1234567890abcdef
```

## Результат в Notion

Створиться сторінка з:
- Заголовком "Storyboard: [назва_відео]"
- Описом з метаданими
- Кожним кадром як окремий розділ з:
  - Зображенням
  - AI аналізом у callout блоці
  - Метаданими (час, кадр, індекс)
  - Розділювачем між кадрами
