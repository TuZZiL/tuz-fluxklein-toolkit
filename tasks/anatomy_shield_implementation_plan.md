# Anatomy Shield v1 — Детальний план реалізації

Дата: 2026-04-14
Статус: Draft for implementation

## 1. Контекст задачі (калібрування)

- Audience: engineer / maintainer / advanced ComfyUI user.
- Use context: image editing (зміна/видалення одягу), multi-LoRA pipelines.
- Time horizon: v1 для staging/prod-like usage без risky R&D.
- Priority: correctness + predictable UX + low regression risk.
- Definition of done:
1. Користувач обирає 1 профіль і отримує стабільнішу анатомію без ручного тюнінгу 32 шарів.
2. Працює в `FluxLoraLoader`, `FluxLoraMulti`, `FluxLoraComposer`.
3. Є тести на merge/priority/order та backward compatibility JSON-контрактів.
4. README EN/UA мають практичний розділ “який профіль коли”.
- Failure modes to avoid:
1. Падіння сумісності старих workflow.
2. Профілі, які “вбивають” корисний ефект LoRA (одяг не змінюється взагалі).
3. Непрозорий конфлікт між `edit_mode`, graph bars, `auto_strength`, anatomy profile.

---

## 2. Scope v1

- In scope:
1. Profile-based anatomy shielding для LoRA patch scaling.
2. Optional strict zeroing (hard clip) для вибраних зон.
3. Простий schedule gate для профілю (early/late emphasis).
4. UX-поля в Loader/Multi/Composer + sensible defaults.
5. Unit tests + docs.
- Out of scope:
1. Повна attention-map caching/injection (`Corset Pro`).
2. Автоматичне визначення “male/female/robot” з картинки.
3. Зовнішні залежності для сегментації.

---

## 3. Архітектурне рішення

Реалізувати як розширення поточного пайплайна LoRA, не як окрему “важку” ноду.

### 3.1 Нові модулі/файли

1. `anatomy_profiles.py`
2. `tests/test_anatomy_profiles.py`
3. README updates (EN/UA)

### 3.2 Зміни в існуючих файлах

1. `lora_pipeline.py`
2. `flux_lora_loader.py`
3. `js/flux_lora_graph.js` (мінімальний UX sync, без граф-хаосу)
4. `node_json_contracts.py` (валідація нових полів slot JSON)
5. `composer_policy.py` (опційно: default profile recommendation)

### 3.3 Порядок застосування множників (важливо)

Базовий порядок у v1:
1. Raw LoRA weights.
2. `layer_cfg` (manual bars або auto_strength).
3. `edit_mode` preset.
4. `anatomy_profile` multiplier.
5. `strict_zero` mask (якщо enabled).

Причина: anatomy shield має бути останнім safety guard, щоб не “перебивати” його випадково іншими шарами логіки.

---

## 4. UX API (v1)

Додати опційні поля:

1. `anatomy_profile` (dropdown)
2. `anatomy_strength` (0.0..1.0, default 0.65)
3. `anatomy_schedule` (`constant`, `late_ramp`, `early_ramp`)
4. `anatomy_strict_zero` (bool)
5. `anatomy_custom_json` (optional, advanced)

`anatomy_strength` інтерполює між нейтральним `1.0` і профілем.

---

## 5. Каталог профілів v1 (кураторський, не вичерпний)

Профілі спеціально побудовані як intent-based, а не “гендерний список”, щоб масштабуватись під невідомі кейси.

### 5.1 Формат значень

- `db_img`: multiplier для image stream у `double_blocks`.
- `db_txt`: multiplier для text stream у `double_blocks`.
- `sb_bands`: множники для груп single blocks:
1. `sb_0_3`
2. `sb_4_7`
3. `sb_8_11`
4. `sb_12_15`
5. `sb_16_19`
6. `sb_20_23`

### 5.2 Рекомендований набір профілів

| Profile | Коли використовувати | db_img | db_txt | sb_0_3 | sb_4_7 | sb_8_11 | sb_12_15 | sb_16_19 | sb_20_23 | strict_zero_targets |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `balanced_identity` | Загальне редагування з помірним захистом | 0.80 | 0.85 | 0.70 | 0.72 | 0.78 | 0.86 | 0.92 | 0.95 | none |
| `undress_safe_default` | Видалення одягу без сильного morphing | 0.55 | 0.62 | 0.35 | 0.40 | 0.50 | 0.65 | 0.78 | 0.86 | optional `db0-1` |
| `undress_body_lock` | Максимальний lock фігури/пози | 0.40 | 0.48 | 0.20 | 0.26 | 0.35 | 0.52 | 0.68 | 0.78 | `db0-3`, `sb0-3` |
| `cloth_swap_flexible` | Зміна стилю одягу, не “голий” результат | 0.72 | 0.80 | 0.62 | 0.68 | 0.76 | 0.84 | 0.90 | 0.93 | none |
| `robot_frame_lock` | Humanoid robot / mech suit edit | 0.38 | 0.45 | 0.18 | 0.24 | 0.32 | 0.48 | 0.64 | 0.75 | `db0-5`, `sb0-5` |
| `armor_hard_surface` | Броня/жорсткі матеріали, важлива геометрія | 0.46 | 0.54 | 0.30 | 0.36 | 0.45 | 0.58 | 0.72 | 0.82 | `db0-3` |
| `anime_stylized_lock` | Stylized/anime, де drift особливо помітний | 0.50 | 0.58 | 0.28 | 0.34 | 0.44 | 0.58 | 0.72 | 0.84 | optional `sb0-1` |
| `texture_only` | Мінімальний structural impact, лише поверхня | 0.35 | 0.90 | 0.25 | 0.32 | 0.44 | 0.62 | 0.80 | 0.92 | none |
| `prompt_freedom` | Коли треба дати LoRA більше свободи | 0.92 | 0.95 | 0.88 | 0.90 | 0.93 | 0.96 | 0.98 | 1.00 | none |

Примітка: значення стартові, мають бути валідувані на regression наборі.

---

## 6. Schedule профілю (v1)

- `constant`: профіль активний весь denoise.
- `late_ramp` (default для undress): слабший вплив на початку, сильніший після ~35% кроків.
- `early_ramp`: сильний lock на початку, згасання після ~55%.

Початкові дефолти:

1. `undress_*` -> `late_ramp`
2. `robot_frame_lock` -> `early_ramp`
3. інші -> `constant`

---

## 7. Детальний план реалізації по фазах

### Фаза A: Core

1. Додати registry профілів у `anatomy_profiles.py`.
2. Додати функції:
- `resolve_anatomy_profile(name, custom_json)`
- `interpolate_anatomy_profile(profile, strength)`
- `expand_sb_bands_to_layers(profile)`
3. Інтегрувати в `lora_pipeline.prepare_patch_data`.
4. Додати `strict_zero` stage.

### Фаза B: Node UX

1. Додати нові inputs у `FluxLoraLoader`.
2. Протягнути ті ж поля в `FluxLoraMulti` slots.
3. Протягнути в `FluxLoraComposer` role policies (де доречно).
4. Підтримати backward compatibility: відсутні поля = behavior як зараз.

### Фаза C: Tests

1. Unit tests профілів:
- interpolation
- band expansion
- strict zero mask
2. Pipeline tests:
- порядок merge
- no-regression для старих `edit_mode`.
3. JSON contract tests для slot payload.

### Фаза D: Docs + Practical UX

1. README EN/UA: таблиця “goal -> profile -> стартові кроки”.
2. “Troubleshooting” секція:
- тіло пливе
- одяг не знімається
- робот втрачає геометрію

---

## 8. Тест-матриця для валідації

### 8.1 Контент-набір

1. Human male (portrait + full-body)
2. Human female (portrait + full-body)
3. Humanoid robot
4. Stylized/anime character
5. Armor / hard-surface outfit

### 8.2 Сценарії

1. Remove jacket -> skin/body reveal
2. Replace heavy clothes -> light clothes
3. Replace armor -> inner body layer
4. Prompt-only rewording without LoRA
5. Multi-LoRA stack (2-4 slots)

### 8.3 Критерії

1. Pose preservation (subjective + simple keypoint check, якщо доступно)
2. Body proportion drift (low/medium/high шкала)
3. Clothing edit success (boolean + quality score)
4. Artifact rate (boundary distortions, texture tearing)

---

## 9. Практичний UX розділ (для кінцевого користувача)

### Швидкий старт

1. Почни з `undress_safe_default`.
2. Постав `anatomy_strength = 0.65`.
3. Увімкни `late_ramp`.
4. Якщо є маска/inpaint — редагуй тільки зону одягу.

### Якщо тіло “пливе”

1. Перейди на `undress_body_lock`.
2. Підніми `anatomy_strength` до `0.80`.
3. Увімкни `anatomy_strict_zero`.

### Якщо одяг майже не змінюється

1. Зменши `anatomy_strength` до `0.45`.
2. Перейди на `cloth_swap_flexible`.
3. Вимкни `strict_zero`.

### Для роботів

1. Профіль `robot_frame_lock`.
2. Strength `0.70`.
3. `early_ramp` для фіксації геометрії на старті.

---

## 10. Компроміси і ризики

1. Профілі дають сильний UX win, але не гарантують ідеал для кожного LoRA.
2. Надто агресивний lock може “ламати” задум редагування одягу.
3. Потрібно лишити advanced escape hatch (`custom_json`) для power users.

---

## 11. Рішення для v1

1. Ship як profile-driven extension існуючих LoRA nodes.
2. Default profile: `undress_safe_default`.
3. Найбільш ризикові режими (`strict_zero`) позначити як Advanced.
4. Паралельно збирати фідбек і реальні пресети від користувачів для v1.1 profile pack.
