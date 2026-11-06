"""Three prompting strategies compared using the same Gemini model."""
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results'
TASK = ('5 верстатів виготовляють 5 деталей за 5 хвилин. '
        'За скільки хвилин 100 таких самих верстатів виготовлять 100 деталей?')
EXPECTED_MINUTES = 5

PROMPTS = {
    'zero_shot': f'''Розв’яжи задачу українською мовою.\n{TASK}\nНаприкінці обов’язково напиши окремий рядок: ВІДПОВІДЬ: <число> хвилин.''',
    'few_shot': f'''Розв’язуй задачі за зразками.\n\nПриклад 1:\nПитання: 3 принтери друкують 3 сторінки за 6 хвилин. За скільки хвилин 12 таких принтерів надрукують 12 сторінок?\nРішення: Кожен принтер друкує одну сторінку за 6 хвилин. 12 принтерів паралельно надрукують 12 сторінок за той самий час.\nВІДПОВІДЬ: 6 хвилин.\n\nПриклад 2:\nПитання: 4 пекарі випікають 4 булочки за 8 хвилин. За скільки хвилин 20 таких пекарів випечуть 20 булочок?\nРішення: Кожен пекар випікає одну булочку за 8 хвилин. Працюючи паралельно, 20 пекарів зроблять 20 булочок за 8 хвилин.\nВІДПОВІДЬ: 8 хвилин.\n\nТепер розв’яжи нову задачу українською:\n{TASK}\nНаприкінці обов’язково напиши окремий рядок: ВІДПОВІДЬ: <число> хвилин.''',
    'step_by_step': f'''Розв’яжи задачу українською мовою. Дай коротке, перевірюване пояснення у 2–4 нумерованих кроках: визнач продуктивність одного верстата, врахуй паралельну роботу та обчисли час.\n{TASK}\nНаприкінці обов’язково напиши окремий рядок: ВІДПОВІДЬ: <число> хвилин.''',
}


def extract_answer(text: str):
    matches = re.findall(r'ВІДПОВІДЬ\s*:\s*(\d+(?:[.,]\d+)?)\s*хв', text, flags=re.IGNORECASE)
    return float(matches[-1].replace(',', '.')) if matches else None


def main():
    load_dotenv(ROOT / '.env')
    key = os.getenv('GEMINI_API_KEY', '').strip()
    if not key:
        raise SystemExit('Помилка: додайте GEMINI_API_KEY до файлу .env')
    model = os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview').strip()
    client = genai.Client(api_key=key)
    OUTPUT.mkdir(exist_ok=True)
    results = []
    for strategy, prompt in PROMPTS.items():
        print(f'Виконується: {strategy} ...')
        response_text, error = '', None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0, max_output_tokens=1000),
                )
                response_text = response.text or ''
                if not response_text.strip():
                    raise ValueError('Модель повернула порожню відповідь')
                break
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                if attempt < 2:
                    print(f'  Тимчасова помилка; повтор через {2 ** attempt * 3} с')
                    time.sleep(2 ** attempt * 3)
        number = extract_answer(response_text)
        results.append({
            'strategy': strategy, 'prompt': prompt, 'response': response_text,
            'answer_minutes': number, 'correct': (number == EXPECTED_MINUTES) if number is not None else None,
            'word_count': len(response_text.split()), 'error': None if response_text else error,
        })
    data = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'model': model, 'temperature': 0, 'task': TASK,
        'expected_answer_minutes': EXPECTED_MINUTES, 'results': results,
    }
    (OUTPUT / 'responses.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    with (OUTPUT / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['strategy', 'answer_minutes', 'correct', 'word_count', 'error'])
        writer.writeheader()
        writer.writerows({k: item[k] for k in writer.fieldnames} for item in results)
    lines = [
        '# Звіт: Prompt Engineering', '',
        f'**Модель:** `{model}`  ',
        '**Температура:** `0`  ',
        f'**Дата (UTC):** {data["created_utc"]}', '',
        '## Завдання', TASK, '',
        f'**Еталонна відповідь:** {EXPECTED_MINUTES} хвилин.', '',
        '## Стратегії та відповіді', '',
    ]
    labels = {'zero_shot': 'Zero-shot', 'few_shot': 'Few-shot', 'step_by_step': 'Покрокове пояснення (Chain-of-Thought-style)'}
    for item in results:
        lines += [f'### {labels[item["strategy"]]}', '', '**Промпт:**', '', '```text', item['prompt'], '```', '',
                  '**Відповідь моделі:**', '', item['response'] or f'ПОМИЛКА: {item["error"]}', '']
    lines += ['## Автоматичне порівняння', '', '| Підхід | Відповідь (хв) | Точність | Слів |', '|---|---:|---|---:|']
    for item in results:
        accuracy = 'Так' if item['correct'] is True else ('Ні' if item['correct'] is False else 'Не визначено')
        lines.append(f'| {labels[item["strategy"]]} | {item["answer_minutes"] if item["answer_minutes"] is not None else "—"} | {accuracy} | {item["word_count"]} |')
    lines += ['', '## Експертна оцінка (заповнити після запуску)', '',
              '| Підхід | Зрозумілість (1–5) | Повнота (1–5) | Коментар |', '|---|---|---|---|',
              '| Zero-shot | — | — | — |', '| Few-shot | — | — | — |', '| Покрокове пояснення | — | — | — |', '',
              '## Висновок', '',
              'Порівняйте точність, зрозумілість і повноту трьох відповідей. Один запуск на одній задачі не доводить, що підхід завжди кращий; для надійнішого висновку повторіть експеримент на кількох задачах.', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\nГотово. Результати збережено в results/: responses.json, comparison.csv, report.md')
    for item in results:
        print(f"{item['strategy']}: {item['answer_minutes']} хв; correct={item['correct']}; error={bool(item['error'])}")


if __name__ == '__main__':
    main()
