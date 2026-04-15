# TUZ FluxKlein Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Завантажувач LoRA для **FLUX.2 Klein** (9B) у ComfyUI.

[English version](README.md)

---

## Проблема

Більшість LoRA натреновані інструментами HuggingFace і збережені у **diffusers форматі**. Стандартний LoRA loader ComfyUI **мовчки губить** більшу частину цих ваг на Klein 9B — LoRA ніби завантажується, але ледь працює або виглядає неправильно.

Крім того, при **редагуванні зображень** через LoRA (зміна одягу, аксесуарів, стилю) LoRA часто **руйнує обличчя** або змінює пропорції тіла.

## Рішення

Цей пакет нод робить дві речі:

1. **Автоматично конвертує** будь-який формат LoRA для коректної роботи з Klein 9B
2. **Захищає те, що ви хочете зберегти** — обличчя, тіло, або стиль — одним dropdown

Просто оберіть пресет, підключіть LoRA і працюйте.

---

## Швидкий старт (2 хвилини)

### Крок 1: Встановлення

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/TuZZiL/tuz-fluxklein-toolkit.git
# Перезапустіть ComfyUI
```

Потребує `numpy` (зазвичай вже є з ComfyUI).
Для `TUZ Klein Edit Composite` ще: `pip install opencv-python-headless`

### Крок 2: Додайте ноду

Знайдіть **TUZ FLUX LoRA Loader** в меню нод у категорії `loaders/FLUX`. Підключіть вхід `MODEL`.

### Крок 3: Виберіть LoRA і генеруйте

1. Оберіть файл LoRA
2. Поставте `edit_mode` → **Auto** (рекомендований старт)
3. Поставте `strength` → **0.7** (безпечний default)
4. Генеруйте!

Все. Auto режим аналізує ваги LoRA і сам обирає найкращий рівень захисту. Далі підлаштовуйте під себе.

---

## Огляд нод

Пакет містить **7 нод** у 3 групах:

### Завантаження LoRA (ядро)

| Нода | Призначення | Коли використовувати |
|---|---|---|
| **TUZ FLUX LoRA Loader** | Одна LoRA + інтерактивний граф | Ваш щоденний інструмент для 1 LoRA |
| **TUZ FLUX LoRA Multi** | Кілька LoRA, динамічні слоти | Стекінг 2-4 LoRA (edit + consistency + style) |
| **TUZ FLUX LoRA Scheduled** | Криві сили по кроках | Коли LoRA має затухати/наростати під час семплінгу |

### Conditioning (допоміжні інструменти)

| Нода | В одному реченні |
|---|---|
| **Ref Latent Controller** | "Зробити reference image сильнішим/слабшим в attention" |
| **Text/Ref Balance** | "Дати prompt більше влади або міцніше тримати reference" |
| **Mask Ref Controller** | "Захистити або послабити конкретні області reference" |
| **Color Anchor** | "Не дати кольорам пливти під час семплінгу" |

### Аналіз та постобробка

| Нода | Призначення |
|---|---|
| **TUZ FLUX Preflight Advisor** | Аналіз LoRA перед запуском — отримати рекомендовані налаштування |
| **TUZ Klein Edit Composite** | Злити згенерований edit назад на оригінальне зображення (після decode) |

---

## Пресети редагування — ключова концепція

Думайте про `edit_mode` як про **регулятор захисту**, а не як про категорію LoRA:

| Пресет | Що захищає | Найкраще для |
|---|---|---|
| **Auto** ⭐ | Аналізує ваги LoRA, обирає сам | Починайте тут — працює для більшості LoRA |
| **Preserve Face** | Обличчя та ідентичність | Одяг/аксесуари/макіяж |
| **Preserve Body** | Обличчя + пропорції тіла | Try-on, заміна outfit |
| **Style Only** | Структуру (зменшує image stream) | Aesthetic/painterly LoRA |
| **Edit Subject** | Помірний захист ідентичності | Зміна об'єктів зі збереженням identity |
| **Boost Prompt** | Нічого (підсилює текст натомість) | Коли prompt занадто слабкий |
| **None** | Нічого | "Сира" LoRA, повна свобода |

### Діал захисту (`balance`)

Працює як діал захисту: `0.0` = "сира" LoRA, `1.0` = повний захист пресету.

```
0.0 ◄━━━━━━━━━━━━━━━━━━━━━► 1.0
"Сира" LoRA (без захисту)   Повний захист пресету
```

**Правило:**
- LoRA перезаписує обличчя? → Збільшуйте `balance` до 1.0
- Edit занадто слабкий або "перестрахований"? → Зменшуйте до 0.0

### Edit vs Generate (`use_case`)

Впливає лише на **Auto** режим:

- **Edit** → Auto більш консервативний (береже ідентичність/структуру)
- **Generate** → Auto дає LoRA більше свободи (text-to-image, restyle)

Ручні пресети завжди працюють рівно так, як ви їх обрали, незалежно від `use_case`.

---

## Детальний довідник нод

### TUZ FLUX LoRA Loader

Завантажувач однієї LoRA з інтерактивним per-layer графом та optional auto-strength.

| Вхід | Тип | Опис |
|---|---|---|
| `model` | MODEL | Модель FLUX.2 Klein / FLUX.1 |
| `lora_name` | dropdown | Файл LoRA з `models/loras` |
| `strength` | float | Глобальна сила LoRA (-5.0 до 5.0) |
| `use_case` | dropdown | Підказує Auto: ви редагуєте reference чи генеруєте вільно |
| `auto_convert` | boolean | Конвертувати diffusers-формат у нативний FLUX |
| `auto_strength` | boolean | Авто-обчислення per-layer strength з ΔW аналізу |
| `edit_mode` | dropdown | Рівень захисту — `Auto` рекомендований старт |
| `balance` | float | Діал захисту: 0.0 = "сира" LoRA, 1.0 = повний захист пресету |
| `anatomy_profile` | dropdown | Opt-in профіль збереження тіла для clothing / body edit |
| `anatomy_strength` | float | Наскільки сильно anatomy profile має захищати структуру |
| `anatomy_strict_zero` | boolean | Advanced. Жорстко вимикає найчутливіші блоки профілю |

**Граф-віджет:** 8 double-block колонок (img фіолетові / txt бірюзові) + 24 single-block колонки (зелені).
- Перетягуйте для налаштування
- Shift+drag рухає всі бари в секції
- Клік перемикає бар вкл/викл

**Auto-strength:** Аналізує тензори ваг LoRA і автоматично заповнює оптимальні per-layer strengths. Після цього ви все ще можете вручну підправити.

**Anatomy Shield:** Використовуйте, коли LoRA постійно змінює форму тіла, а вам треба рухати лише одяг або поверхневі деталі.
- `Undress Safe` - стартовий профіль для зняття одягу.
- `Undress Body Lock` - жорсткіший режим, сильніше тримає силует.
- `Robot Frame Lock` - найкращий старт для humanoid robot або mech edits.
- Залишайте `anatomy_profile=None`, якщо хочете стару поведінку без змін.
- Починайте з `anatomy_strength=0.60–0.70`; піднімайте лише якщо тіло все ще пливе.

**Практичне правило:** `edit_mode` керує тим, наскільки LoRA захищена по шарах, а `anatomy_profile` додає поверх цього високорівневий пресет збереження тіла. Використовуйте обидва лише тоді, коли це справді потрібно.

### TUZ FLUX LoRA Multi

Динамічний multi-LoRA завантажувач. Натисніть **"+ Add LoRA"** для нового слоту, **"✕"** для видалення.

Кожен слот має: Enabled перемикач, LoRA dropdown, Strength, Use case, Edit mode, Protection.

| Вхід | Тип | Опис |
|---|---|---|
| `model` | MODEL | Модель FLUX.2 Klein / FLUX.1 |
| `auto_convert` | boolean | Конвертувати diffusers-формат |

**Рекомендоване налаштування для редагування зображень:**

```
Слот 1: editing LoRA       → edit_mode=Auto, strength=0.6–0.8
Слот 2: consistency LoRA   → edit_mode=Auto, strength=0.4–0.6
Слот 3: enhancer LoRA      → edit_mode=None, strength=0.2–0.4
```

Якщо ви робите body-preserving edit або знімаєте одяг, anatomy profile ставте лише на головний слот. Стартові варіанти:
- `Undress Safe` для звичайного зняття одягу.
- `Undress Body Lock`, якщо силует має лишатися стабільним.
- `Armor Hard Surface` для броні або жорсткого одягу.

### TUZ FLUX LoRA Scheduled

Per-step контроль сили LoRA через Hook Keyframes ComfyUI. Ефект LoRA змінюється протягом кроків семплінгу. Приймає conditioning і повертає модифікований — окрема утилітна нода не потрібна.

| Вхід | Тип | Опис |
|---|---|---|
| `model` | MODEL | Модель FLUX.2 Klein |
| `conditioning` | CONDITIONING | Базовий conditioning |
| `lora_name` | dropdown | Файл LoRA |
| `strength` | float | Базова сила LoRA (0.0–2.0) |
| `schedule` | dropdown | Профіль кривої сили |
| `edit_mode` | dropdown | Рівень захисту (підтримує Auto) |
| `balance` | float | "Сира" LoRA ↔ повний захист пресету |
| `keyframes` | int | Кількість keyframes (2–10) |

**Повертає:** `MODEL` + `CONDITIONING`

| Розклад | Крива | Найкраще для |
|---|---|---|
| **Constant** | `1.0 → 1.0 → 1.0` | Стандартна поведінка |
| **Fade Out** | `1.0 → 0.7 → 0.3 → 0.0` | Зміни на початку, reference в кінці |
| **Fade In** | `0.0 → 0.3 → 0.7 → 1.0` | Спочатку reference, потім LoRA |
| **Strong Start** | `1.0 → 0.5 → 0.2 → 0.0` | Агресивне затухання, макс збереження |
| **Pulse** | `0.3 → 1.0 → 1.0 → 0.3` | Пік в середніх кроках |

### TUZ FLUX Preflight Advisor

Аналізує файл LoRA + сумісність з моделлю і повертає рекомендації **без зміни** чого-небудь.

| Вихід | Тип | Опис |
|---|---|---|
| `report` | STRING | Людиночитаний підсумок із warnings |
| `recommended_edit_mode` | STRING | Рекомендований пресет |
| `recommended_balance` | FLOAT | Рекомендований рівень захисту |
| `recommended_strength` | FLOAT | Безпечний стартовий strength |
| `compat_status` | STRING | `ok`, `partial` або `failed` |
| `matched_modules` | INT | Модулі LoRA, сумісні з моделлю |
| `total_modules` | INT | Загальна кількість модулів LoRA |

> **Важливо:** `recommended_edit_mode=None` означає "Raw / No Protection", а не "нічого не обрано".

Multi-версія (`TUZ FLUX Multi Preflight Advisor`) приймає той самий JSON-формат слотів, що й `TUZ FLUX LoRA Multi`.

---

## Companion conditioning-ноди

Це **невеликі коригуючі інструменти**, які сидять поруч з LoRA loader у графі. Вони не замінюють loader. Вони уточнюють взаємодію reference image та prompt.

**Базовий flow:**
```
reference image → VAE Encode → ReferenceLatent
text prompt → conditioning
LoRA loader → model
companion nodes → conditioning/model
sampler → decode → TUZ Klein Edit Composite
```

<details>
<summary><b>Практичний гайд (натисніть, щоб розгорнути)</b></summary>

### Коли яку ноду брати

| Проблема | Брати |
|---|---|
| Prompt ігнорується reference-ом | `Text/Ref Balance` |
| Лише одна область має лишитися reference-правдою | `Mask Ref Controller` |
| Увесь reference занадто жорсткий або занадто слабкий | `Ref Latent Controller` |
| Композиція або поза пливе | `Structure Lock` |
| Кольори пливуть, але структура правильна | `Color Anchor` |

### Ref Latent Controller

Ця нода змінює, наскільки сильно reference tokens лишаються в attention path. Це глобальний діал “тиску reference”.

Використовуйте її, коли reference надто домінує, надто слабкий або занадто жорсткий загалом.

Стартові значення: `strength=1.0`, `appearance_scale=1.15`, `detail_scale=0.75`, `blur_radius=2`.

Як крутити:
- Піднімайте `strength` або `appearance_scale`, якщо prompt занадто легко перетискає reference.
- Зменшуйте `detail_scale`, якщо identity правильна, але текстури й дрібні деталі занадто “заблоковані”.
- Ставте `reference_index=-1`, коли треба однаково обробити всі reference latent.
- Використовуйте `spatial_fade`, коли лише частина reference має лишатися сильнішою.

Практичне застосування: зберегти обличчя або позу, але послабити жорсткість тканин чи дрібних деталей.

### Text/Ref Balance

Ця нода зміщує контроль між prompt і reference. `attn_patch` змінює attention напряму, а `latent_mix` послаблює reference ще до семплінгу.

Використовуйте її, коли prompt не проходить або reference перемагає.

Стартові значення: `balance=0.45` для prompt-led edit, `balance=0.65`, коли prompt має бути сильнішим.

Як крутити:
- Зменшуйте `balance`, щоб reference лишався сильнішим.
- Збільшуйте `balance`, щоб text take over був агресивнішим.
- `attn_patch` залишайте як default.
- `latent_mix` використовуйте тільки коли prompt і далі недопрацьовує після attention patch.

Практичне застосування: змусити впертий reference підкоритися prompt, не прибираючи reference повністю.

### Mask Ref Controller

Ця нода застосовує просторову маску до reference latent. Вона не локалізує prompt. Вона локалізує саме reference influence.

Використовуйте її, коли одна зона має лишатися reference-правдою, а інша має змінюватися.

Стартові значення: `mask_action=scale`, `strength=0.8`, `feather=12`.

Як крутити:
- `scale` послаблює reference під маскою.
- `mix` замінює masked regions іншим latent-сигналом.
- `invert_mask=True` міняє місцями захищену й змінену область.
- `reference_keep` має значення лише в режимі `mix`.

Практичне застосування: захистити обличчя і змінити одяг або фон, або інвертувати маску, щоб пушити edit у сам subject.

### Structure Lock

Ця нода тримає грубу просторову структуру reference, але все ще дозволяє prompt змінювати текстуру й деталі.

Використовуйте її, коли поза, кадрування або розташування об'єкта пливуть під час edit.

Стартові значення: `strength=0.35`, `blur_radius=6`, `ramp_start=0.0`, `ramp_end=0.5`.

Як крутити:
- Піднімайте `strength`, якщо композиція все ще занадто дрейфує.
- Піднімайте `blur_radius`, якщо lock стає занадто липким і починає заморожувати деталі.
- Скорочуйте `ramp_end`, якщо хочете, щоб структура трималась тільки на ранніх кроках.
- Ставте `reference_index=-1`, коли однаковий lock має поширюватися на кілька reference.
- Додавайте маску, якщо треба заякорити тільки обличчя, subject або background.

Практичне застосування: стабілізувати позу в портретах, зберегти framing товару або утримати subject на місці під час заміни background.

### Color Anchor

Ця нода робить color correction під час семплінгу, щоб результат лишався ближчим до source palette.

Використовуйте її, коли композиція правильна, але палітра пливе.

Стартові значення: `strength=0.35`, `ramp_curve=1.5`, `channel_weights=uniform`.

Як крутити:
- Піднімайте `strength`, якщо результат все ще надто відхиляється від reference palette.
- Піднімайте `ramp_curve`, якщо корекція стартує занадто рано і робить картинку плоскою.
- Використовуйте `by_variance`, коли частина каналів стабільна, а частина шумна.

Практичне застосування: зберігати skin tones, кольори одягу або background hues ближчими до reference без жорсткого color match.

### Типові зв'язки

| Ціль | Пара |
|---|---|
| Локальний контроль identity по області | `Ref Latent Controller` + `Mask Ref Controller` |
| Prompt проходить, але кольори пливуть | `Text/Ref Balance` + `Color Anchor` |
| Переписати треба лише частину кадру | `Mask Ref Controller` + `Color Anchor` |

</details>

---

## TUZ Klein Edit Composite

Postprocess-нода для злиття згенерованого edit назад на оригінальне зображення. Стоїть **після VAE Decode**, не всередині LoRA pipeline.

```
оригінал + generated edit → VAE Decode → TUZ Klein Edit Composite → save
```

**Потрібно:** `pip install opencv-python-headless`

<details>
<summary><b>Повний довідник полів (натисніть, щоб розгорнути)</b></summary>

| Поле | Тип | Що робить |
|---|---|---|
| `generated_image` | IMAGE | Змінене зображення для композиту |
| `original_image` | IMAGE | Чисте джерело |
| `delta_e_threshold` | FLOAT | Чутливість до змін (-1 = авто) |
| `flow_quality` | choice | `medium`, `fast`, `ultrafast` |
| `use_occlusion` | BOOLEAN | Consistency перевірка flow |
| `occlusion_threshold` | FLOAT | Чутливість occlusion (-1 = авто) |
| `noise_removal_pct` | FLOAT | Прибрати шум з маски (% діагоналі) |
| `close_radius_pct` | FLOAT | Morphological close (% діагоналі) |
| `fill_holes` | BOOLEAN | Заповнити дірки в масці |
| `fill_borders` | BOOLEAN | Розтягти маску в warped-border |
| `max_islands` | INT | Тільки N найбільших regions (0 = всі) |
| `grow_mask_pct` | FLOAT | Збільшити/зменшити маску (% діагоналі) |
| `feather_pct` | FLOAT | Пом'якшити край blend (% діагоналі) |
| `color_match_blend` | FLOAT | Підтягти кольори до original |
| `poisson_blend_edges` | BOOLEAN | Seamless blending країв |
| `custom_mask` | MASK | Зовнішня маска (optional) |
| `custom_mask_mode` | choice | `replace`, `add`, `subtract` |
| `enable_debug` | BOOLEAN | Debug gallery + детальний report |

</details>

---

## Практичні рецепти

### Рецепт 1: Базова зміна одягу зі збереженням обличчя

```
LoRA Loader: edit_mode=Preserve Face, strength=0.7, balance=0.7
```

### Рецепт 2: Стилізація без структурних пошкоджень

```
LoRA Loader: edit_mode=Style Only, strength=0.5, balance=0.5
```

### Рецепт 3: Multi-LoRA з фіксацією ідентичності

```
Слот 1: clothing LoRA  → edit_mode=Preserve Body, strength=0.7
Слот 2: enhancer       → edit_mode=None, strength=0.3
+ Text/Ref Balance: balance=0.6 (трохи підсилити prompt)
+ Color Anchor: strength=0.3 (запобігти color shift)
```

### Рецепт 4: Prompt занадто слабкий

```
LoRA Loader: edit_mode=Boost Prompt, strength=0.8, balance=0.6
```

### Рецепт 5: Зберегти look, але дати edit "дихати"

```
LoRA Loader: edit_mode=Auto
+ Ref Latent Controller: appearance_scale=1.15, detail_scale=0.7
```

---

## Як це працює всередині

<details>
<summary><b>Конвертація форматів (натисніть, щоб розгорнути)</b></summary>

| Формат | Джерело | Приклад ключів | Обробка |
|---|---|---|---|
| **Native** | ComfyUI, kohya | `diffusion_model.double_blocks.0.img_attn.qkv` | Прямий прохід |
| **Diffusers** | HuggingFace | `transformer_blocks.0.attn.to_q` | Block-diagonal QKV fusion |
| **Musubi Tuner / PEFT** | Modelscope, MuseAI | `single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.default` | Ремапінг ключів |

Всі формати автовизначаються коли `auto_convert` увімкнено.

**Суть проблеми:**
LoRA поставляються з окремими `to_q` / `to_k` / `to_v` проєкціями. FLUX зберігає їх як єдину fused QKV матрицю. Без конвертації більшість attention ваг просто не потрапляють у модель.

**Математика (block-diagonal fusion):**
Для fused ваги `W = [W_q; W_k; W_v]` та окремих LoRA `B_i @ A_i`:

```
A_fused = cat([A_q, A_k, A_v], dim=0)         [3r × in]
B_fused = block_diag(B_q, B_k, B_v)           [3·out × 3r]
```

Alpha/rank scaling вбудовано в B_fused.

</details>

<details>
<summary><b>Як працює auto-strength (натисніть, щоб розгорнути)</b></summary>

Для кожної пари шарів у файлі:

```
ΔW = lora_B @ lora_A
scaled_norm = frobenius_norm(ΔW) × (alpha / rank)
strength = clamp(global × (mean_norm / layer_norm), floor=0.30, ceiling=1.50)
```

Double blocks обробляються з img та txt потоками незалежно. Середній шар потрапляє на `global_strength`.

</details>

<details>
<summary><b>Чому Auto обирає саме такий пресет (натисніть, щоб розгорнути)</b></summary>

Auto аналізує розподіл ваг LoRA по шарах архітектури:

- **Високий сигнал у пізніх single blocks** (editing LoRA) → **Preserve Body**
- **Помірний сигнал** → **Preserve Face**
- **Рівномірний** (enhancer-и, слайдери) → **None**

Protection теж обчислюється автоматично. В консолі видно рішення:
```
[FLUX LoRA Multi slot 1] Auto → Preserve Body (protection=0.75)
[FLUX LoRA Multi slot 2] Auto → Preserve Face (protection=0.60)
```

</details>

<details>
<summary><b>Архітектура FLUX.2 Klein 9B (натисніть, щоб розгорнути)</b></summary>

```
Double blocks (8 шарів)
  img потік:
    img_attn.qkv    [12288, 4096]  (об'єднані Q+K+V)
    img_attn.proj   [4096, 4096]
    img_mlp.0       [24576, 4096]
    img_mlp.2       [4096, 12288]
  txt потік:
    txt_attn.qkv    [12288, 4096]
    txt_attn.proj   [4096, 4096]
    txt_mlp.0       [24576, 4096]
    txt_mlp.2       [4096, 12288]

Single blocks (24 шари)
  linear1    [36864, 4096]  (об'єднані Q+K+V+proj_mlp)
  linear2    [4096, 16384]

dim=4096  double_blocks=8  single_blocks=24
```

- **Double blocks (0-7):** Потоки image та text ізольовані — крос-модальна корупція неможлива.
- **Single blocks (0-23):** Спільна обробка — текстовий prompt може перезаписати reference. Пізні single blocks (12-23) найагресивніші.

</details>

---

## FAQ

**Q: Чи потрібен цей пакет для FLUX.1?**
A: Працює і з FLUX.1, але основна цінність — для Klein 9B, де невідповідність архітектури найчастіша.

**Q: Що означає `None` в edit_mode?**
A: "Raw / No Protection" — не "нічого не обрано". LoRA працює з усіма шарами на рівній силі.

**Q: Моя LoRA ніби не має ефекту.**
A: Перевірте що `auto_convert` увімкнено. З GGUF-моделлю переконайтеся, що [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) встановлено.

**Q: Auto режим обирає неправильний пресет.**
A: Auto не читає ваш намір. Переключіться на ручний: `Preserve Face` для identity-роботи, `None` для повної свободи. Підлаштовуйте `balance` як діал захисту.

**Q: Як дізнатися який пресет обрав Auto?**
A: Дивіться консоль ComfyUI. Там логується: `Auto → Preserve Body (protection=0.75)`.

**Q: Чи можна conditioning ноди без LoRA loader?**
A: Так. Це незалежні ноди на `MODEL` та `CONDITIONING` — корисні у будь-якому FLUX workflow.

---

## Подяки

- Оригінальний пакет нод від [capitan01R](https://github.com/capitan01R/Comfyui-flux2klein-Lora-loader)
- Пресети редагування базуються на дослідженні архітектури з [comfyUI-Realtime-Lora](https://github.com/shootthesound/comfyUI-Realtime-Lora) (Klein 9B debiaser / layer mapping)
- GGUF підтримка через [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
