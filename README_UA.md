# ComfyUI FLUX.2 Klein LoRA Loader
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Завантажувач LoRA для **FLUX.2 Klein** (9B) у ComfyUI.

[English version](README.md)

## Навіщо це замість стандартного LoRA loader?

Більшість LoRA, які ви завантажуєте, натреновані інструментами HuggingFace і збережені у **diffusers форматі**. Стандартний ComfyUI LoRA loader **мовчки губить** більшу частину цих ваг на Klein 9B — LoRA ніби завантажується, але ледь працює або виглядає неправильно.

Цей пакет **автоматично конвертує** будь-який формат LoRA для коректної роботи з Klein 9B. Просто підключіть свою LoRA — і вона працює на повну.

Але це не все. Коли ви використовуєте LoRA для **редагування зображень** (зміна одягу, додавання аксесуарів, перенос стилю на reference фото), LoRA часто **руйнує обличчя** або змінює пропорції тіла. Цей пакет вирішує це через **пресети редагування** — один dropdown, який вказує загрузчику які частини зображення захищати:

- **Preserve Face** — редагуйте вільно, зберігаючи обличчя
- **Preserve Body** — захист і обличчя, і пропорцій тіла (фігура, поза)
- **Auto** — загрузчик аналізує вашу LoRA і сам обирає найкращий захист

Результат: ваші зміни застосовуються там де потрібно, а все інше залишається недоторканим.

## Можливості

- **3 сфокусовані ноди**: Loader (одна LoRA + граф), Multi (динамічні слоти), Scheduled (темпоральний контроль)
- **Автовизначення формату**: Підтримка native, diffusers та Musubi Tuner (PEFT) LoRA
- **Block-diagonal QKV fusion**: Коректне об'єднання окремих Q/K/V проєкцій у fused матриці
- **Пресети редагування**: Збереження обличчя, тіла або стилю при редагуванні через LoRA
- **Auto режим**: Аналізує ваги LoRA і автоматично обирає найкращий пресет
- **Auto-strength**: Вбудована per-layer калібрація сили через ΔW аналіз
- **Динамічний multi-slot**: Додавання/видалення слотів LoRA з per-slot edit_mode (rgthree-стиль)
- **Scheduling по кроках**: Зміна сили LoRA протягом sampling steps
- **GGUF сумісність**: Працює з квантованими моделями через ComfyUI-GGUF

## Передумови

LoRA натреновані для FLUX моделей часто поставляються у diffusers форматі — окремі `to_q`, `to_k`, `to_v` проєкції. Нативна архітектура FLUX зберігає їх як єдину QKV матрицю, а single blocks об'єднують attention і MLP в один `linear1`. Завантаження таких LoRA без конвертації означає що більшість attention ваг просто не потрапляють у модель.

| Що в LoRA | Що очікує FLUX | Що робить цей пакет |
|---|---|---|
| Окремі `to_q` / `to_k` / `to_v` | Об'єднана `img_attn.qkv` / `txt_attn.qkv` | Block-diagonal fusion при завантаженні |
| Окремі компоненти single block | Об'єднана `linear1` `[36864, 4096]` | Правильне злиття `[q, k, v, proj_mlp]` |
| Musubi Tuner `.default.` ключі | Стандартні LoRA ключі | Авто-стрипання та ремапінг |
| Тільки глобальна сила | Незалежні img/txt + per-single-block | Інтерактивний віджет + авто-калібрація |
| LoRA впливає на все однаково | Різні шари контролюють різні аспекти | **Пресети** для селективного контролю |

## Встановлення

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/TuZZiL/Comfyui-flux2klein-Lora-loader.git
```

Потребує `numpy` (зазвичай вже встановлено з ComfyUI).

## Ноди

### TUZ FLUX LoRA Loader

Завантажувач однієї LoRA з інтерактивним графом per-layer strength та опціональним auto-strength.

| Вхід | Тип | Опис |
|---|---|---|
| `model` | MODEL | Модель FLUX.2 Klein / FLUX.1 |
| `lora_name` | dropdown | Файл LoRA з `models/loras` |
| `strength` | float | Глобальна сила LoRA (-5.0 до 5.0) |
| `use_case` | dropdown | Підказує Auto, чи ви редагуєте reference-зображення, чи робите вільнішу генерацію |
| `auto_convert` | boolean | Конвертувати diffusers-формат у нативний FLUX |
| `auto_strength` | boolean | Увімкнено: вирівнює нерівномірні LoRA через авто per-layer strength на основі ΔW аналізу |
| `edit_mode` | dropdown | Наскільки “захисно” поводитися з LoRA; `Auto` — рекомендований старт |
| `balance` | float | 0.0 = найсильніший ефект пресету, 1.0 = “сирa” поведінка LoRA |

**Граф-віджет:** Показує double blocks (8 колонок, img фіолетові / txt бірюзові, розділені верх/низ) та single blocks (24 колонки, зелені). Перетягуйте для налаштування. Shift+drag рухає всі бари в секції. Клік перемикає бар вкл/викл.

**Auto-strength:** Коли увімкнено, нода аналізує тензори ваг LoRA і обчислює оптимальні per-layer strengths автоматично. Бари графа заповнюються автоматично — ви все ще можете вручну підправити їх.

### TUZ FLUX LoRA Multi

**Динамічний multi-LoRA завантажувач** з per-slot контролем. Натисніть **"+ Add LoRA"** щоб додати слот, **"✕"** щоб видалити.

Кожен слот має:
- **Enabled** перемикач
- **LoRA** dropdown
- **Strength** (-5.0 до 5.0)
- **Use case** (Edit або Generate)
- **Edit mode** (None, Preserve Face, Preserve Body, Style Only, Edit Subject, Boost Prompt, Auto)
- **Balance** (0.0 до 1.0)

| Вхід | Тип | Опис |
|---|---|---|
| `model` | MODEL | Модель FLUX.2 Klein / FLUX.1 |
| `auto_convert` | boolean | Конвертувати diffusers-формат у нативний FLUX |

Рекомендоване налаштування для редагування:
```
Слот 1: editing LoRA       → edit_mode=Auto (або Preserve Body), strength=0.6-0.8
Слот 2: consistency LoRA   → edit_mode=Auto (або None),          strength=0.4-0.6
Слот 3: enhancer LoRA      → edit_mode=Auto (або None),          strength=0.2-0.4
```

### TUZ FLUX LoRA Scheduled

**Per-step контроль сили LoRA** через вбудовану систему Hook Keyframes ComfyUI. Замість постійної сили, ефект LoRA змінюється протягом кроків. Приймає conditioning на вхід і повертає модифікований conditioning — окрема утилітна нода не потрібна.

| Вхід | Тип | Опис |
|---|---|---|
| `model` | MODEL | Модель FLUX.2 Klein |
| `conditioning` | CONDITIONING | Базовий conditioning (hooks прикріплюються автоматично) |
| `lora_name` | dropdown | Файл LoRA |
| `strength` | float | Базова сила LoRA (0.0–2.0) |
| `use_case` | dropdown | Підказує Auto, чи важливіше зберегти reference, чи дати LoRA більше свободи |
| `schedule` | dropdown | Профіль кривої сили |
| `edit_mode` | dropdown | Наскільки “захисно” поводитися з LoRA (підтримує Auto) |
| `balance` | float | 0.0 = найсильніший ефект пресету, 1.0 = “сирa” поведінка LoRA |
| `keyframes` | int | Кількість keyframes (2-10, за замовч. 5) |

**Повертає:** `MODEL` + `CONDITIONING`

```
FluxLoraScheduled → MODEL → CFGGuider
                  → CONDITIONING → ReferenceLatent → CFGGuider
```

Доступні розклади:

| Розклад | Крива | Коли використовувати |
|---|---|---|
| **Constant** | `1.0 → 1.0 → 1.0 → 1.0` | Стандартна поведінка (без scheduling) |
| **Fade Out** | `1.0 → 0.7 → 0.3 → 0.0` | Editing: зміни на початку, відновлення reference в кінці |
| **Fade In** | `0.0 → 0.3 → 0.7 → 1.0` | Деталізація: спочатку reference, потім LoRA |
| **Strong Start** | `1.0 → 0.5 → 0.2 → 0.0` | Агресивний fade-out для максимального збереження |
| **Pulse** | `0.3 → 1.0 → 1.0 → 0.3` | Піковий ефект в середніх кроках |

### Companion conditioning-ноди

Ці ноди керують reference-latent та conditioning поведінкою, не втручаючись у pipeline LoRA loader.

| Нода | Що робить |
|---|---|
| `TUZ FLUX.2 Klein Ref Latent Controller` | Керує силою окремого reference image в attention path. |
| `TUZ FLUX.2 Klein Text/Ref Balance` | Балансує текст і reference одним повзунком. |
| `TUZ FLUX.2 Klein Mask Ref Controller` | Використовує маску, щоб захищати або відпускати області reference latent. |
| `TUZ FLUX.2 Klein Color Anchor` | Тримає кольори reference ближче до джерела під час семплінгу. |

Ці ноди зроблені в простому ComfyUI-стилі: одна нода = одна задача, стандартні поля, без зайвої візуальної обгортки.

## Пресети редагування

Краще думати про `edit_mode` як про **рівень захисту**, а не як про “тип LoRA”. Різні edit LoRA на Klein можуть поводитися дуже по-різному:

- одні здебільшого міняють одяг, аксесуари або макіяж
- інші сильно штовхають позу, пропорції тіла або композицію
- частина є стилістичними LoRA, які люди просто використовують у режимі edit
- частина є consistency / enhancer LoRA і нормально працюють майже без захисту

При використанні LoRA для редагування зображень (наприклад, зміна одягу на фото) LoRA може пошкодити частини зображення, які ви хочете зберегти — найчастіше обличчя та ідентичність. Це відбувається тому, що FLUX.2 Klein обробляє зображення та текст по-різному на різних шарах:

- **Double blocks (0-7):** Потоки зображення та тексту ізольовані — вони не можуть самі по собі спричинити текст-керовану корупцію.
- **Single blocks (0-23):** Спільна крос-модальна обробка — саме тут текстовий промпт перезаписує reference зображення. Пізні single blocks (12-23) найагресивніші.

### Доступні пресети

| Пресет | Що робить | Коли використовувати |
|---|---|---|
| **None** | Стандартна LoRA (всі шари рівні) | Поведінка за замовчуванням |
| **Preserve Face** | Послаблює пізні single blocks | Редагування зі збереженням обличчя |
| **Preserve Body** | Агресивно послаблює середні+пізні single blocks | Збереження обличчя + пропорцій тіла |
| **Style Only** | Знижує img потік у double blocks | Зміна стилю без структурних змін |
| **Edit Subject** | Помірний захист пізніх блоків, легке підсилення txt | Зміна одягу/об'єктів зі збереженням ідентичності |
| **Boost Prompt** | Підсилює txt потік і середні single blocks | Коли промпт недостатньо виконується |
| **Auto** | Аналізує ваги LoRA, автоматично обирає пресет + balance | Без налаштувань — рекомендовано для більшості |

### З Чого Почати Для Різних LoRA

| Якщо ваша LoRA більше схожа на... | Початковий режим | Чому |
|---|---|---|
| Редагування одягу / аксесуарів / волосся / макіяжу | **Auto** або **Preserve Face** | Зазвичай краще тримає ідентичність, але дає локальні зміни |
| Заміна одягу / try-on / body-sensitive edit | **Auto** або **Preserve Body** | Кращий старт, коли пливуть лице, силует або пропорції |
| Стиль / aesthetic / painterly | **Auto** або **Style Only** | Дозволяє міняти вигляд, але менше ламає структуру |
| Consistency / enhancer / “fixer” LoRA | **None** або **Auto** | Такі LoRA часто й так поводяться акуратно |
| LoRA занадто слабо слухається промпта | **Boost Prompt** | Дає текстовим змінам більше ваги |
| Ви навмисно хочете дати LoRA вільно міняти лице/тіло | **None** | “Сира” LoRA без додаткового захисту |

### Use Case: Edit vs Generate

`use_case` впливає лише на **Auto**. Ручні режими завжди працюють рівно так, як ви їх вибрали.

- **Edit**: Найкраще для reference-driven редагування. Auto поводиться обережніше й сильніше береже ідентичність та структуру.
- **Generate**: Найкраще для text-to-image, вільного restyle або коли немає сильного reference, який треба берегти. Auto дає LoRA більше свободи.

Це важливо, бо edit LoRA для Klein не однакові. Style LoRA, clothing edit LoRA і consistency LoRA можуть бути однаково валідними, але їм потрібні різні стартові припущення.

### Auto режим

Коли `edit_mode` встановлено в **Auto**, нода аналізує розподіл ваг кожної LoRA і обирає оптимальний пресет:

- Високий тренувальний сигнал у пізніх single blocks (editing LoRA) → **Preserve Body**
- Помірний сигнал → **Preserve Face**
- Рівномірний розподіл (слайдери, enhancer-и) → **None**

Auto — сильний стартовий режим, але він все одно не читає ваш намір. Якщо ви спеціально хочете сильніше міняти позу, лице або тіло, переходьте на **None** або піднімайте `balance` ближче до `1.0`.

Balance також обчислюється автоматично. В консолі видно який пресет було обрано:
```
[FLUX LoRA Multi slot 1] Auto → Preserve Body (balance=0.25)
[FLUX LoRA Multi slot 2] Auto → Preserve Face (balance=0.40)
```

### Повзунок Balance

Повзунок `balance` інтерполює між пресетом та стандартною поведінкою:
- **0.0** — повний ефект пресету (максимальний захист)
- **0.5** — середина між пресетом і стандартом
- **1.0** — стандартна LoRA (пресет не діє)

Практичне правило:
- Зменшуйте `balance`, якщо LoRA продовжує перезаписувати обличчя, тіло або структуру reference.
- Збільшуйте `balance`, якщо редагування відчувається занадто слабким або “занадто обережним”.

## Як працює Auto Strength

Для кожної пари шарів у файлі:

```
ΔW = lora_B @ lora_A
scaled_norm = frobenius_norm(ΔW) * (alpha / rank)
strength = clamp(global * (mean_norm / layer_norm), floor=0.30, ceiling=1.50)
```

Double blocks обробляються з img та txt потоками незалежно. Середній шар потрапляє на `global_strength`.

## Підтримувані формати LoRA

| Формат | Джерело | Приклад ключів | Обробка |
|---|---|---|---|
| **Native** | ComfyUI, kohya | `diffusion_model.double_blocks.0.img_attn.qkv` | Прямий прохід |
| **Diffusers** | HuggingFace | `transformer_blocks.0.attn.to_q` | Block-diagonal QKV fusion |
| **Musubi Tuner / PEFT** | Modelscope, MuseAI | `single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.default` | Ремапінг ключів |

Всі формати автовизначаються коли `auto_convert` увімкнено.

## Архітектура FLUX.2 Klein 9B

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

## Подяки

- Оригінальний пакет нод від [capitan01R](https://github.com/capitan01R/Comfyui-flux2klein-Lora-loader)
- Пресети редагування базуються на дослідженні архітектури з [comfyUI-Realtime-Lora](https://github.com/shootthesound/comfyUI-Realtime-Lora) (Klein 9B debiaser / layer mapping)
- GGUF підтримка через [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
