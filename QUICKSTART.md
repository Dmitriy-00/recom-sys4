# Быстрый Старт

## Установка

```bash
# Установить зависимости
pip install -r requirements.txt
```

## Запуск Примера

```bash
# Запустить демонстрацию
python examples/basic_usage.py
```

## Минимальный Пример

```python
from recom_system.recommendation_engine import RecommendationEngine
from recom_system.models import *

# 1. Создать систему
engine = RecommendationEngine()

# 2. Создать материал
material = Material(
    id="matrix_1999",
    metadata=MaterialMetadata(
        title="The Matrix",
        year=1999,
        type=MaterialType.FILM
    )
)

# Установить базовые параметры
material.difficulty.difficulty_level = 6.0
material.transformative.transformative_score = 8.5

# Добавить тропы
material.tropes = [
    TropeUsage(
        trope_id="chosen_one",
        trope_name="The Chosen One",
        usage_type=UsageType.STRAIGHT,
        execution=9.0,
        centrality=0.9
    )
]

engine.add_material(material)

# 3. Создать пользователя
user = User(id="user_001", username="alice")
user.cognitive.current_complexity_comfort = 6.0
user.cognitive.meta_awareness_level = 7.0
engine.add_user(user)

# 4. Получить рекомендации
recommendations = engine.get_recommendations(
    user_id="user_001",
    top_k=5
)

# 5. Показать результаты
for rec in recommendations:
    print(f"{rec['material'].metadata.title}: {rec['final_score']:.2f}/10")
    print(f"  {rec['explanation']}")
```

## Структура Проекта

```
recom-sys4/
├── recom_system/              # Основной пакет
│   ├── __init__.py
│   ├── models.py              # Модели данных
│   ├── trope_map.py           # Карта тропов
│   ├── scoring_components.py  # Компоненты скоринга
│   └── recommendation_engine.py  # Главный движок
│
├── examples/
│   └── basic_usage.py         # Полный пример
│
├── requirements.txt
├── QUICKSTART.md              # Этот файл
└── PROJECT_README.md          # Подробная документация
```

## Основные Концепции

### 1. Материалы
Объекты с 89+ параметрами: сложность, трансформативность, тропы, темы и т.д.

### 2. Пользователи
Многомерный профиль: когнитивный стиль, предпочтения, история просмотров.

### 3. Тропы
Нарративные паттерны с разными типами использования (straight, deconstruction, meta).

### 4. Зона Ближайшего Развития
Система рекомендует материалы на уровне `current_level + 1` для оптимального роста.

### 5. Трансформативность
Фокус на материалах, способных изменить мировоззрение.

## Следующие Шаги

1. Изучите `examples/basic_usage.py` для понимания API
2. Прочитайте `PROJECT_README.md` для деталей
3. Посмотрите оригинальный алгоритм в `Максимально полный алгоритм рекомендаций трансформативных медиа.md`
4. Добавьте свои материалы и тропы
5. Экспериментируйте с весами компонентов

## Помощь

- Документация: `PROJECT_README.md`
- Примеры: `examples/`
- Алгоритм: `Максимально полный алгоритм рекомендаций трансформативных медиа.md`
