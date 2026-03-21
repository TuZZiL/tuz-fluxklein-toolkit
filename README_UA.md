# ComfyUI FLUX.2 Klein LoRA Loader
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Завантажувач LoRA для **FLUX.2 Klein** (9B) у ComfyUI з автоматичною калібрацією сили по шарах, **семантичними пресетами** для збереження ідентичності при редагуванні зображень та **Auto-режимом**, який сам обирає найкращий пресет.

[English version](README.md)

![](images/nodes.png)

## Можливості

- **Автовизначення формату**: Підтримка native, diffusers та Musubi Tuner (PEFT) LoRA
- **Block-diagonal QKV fusion**: Коректне об'єднання окремих Q/K/V проєкцій у fused матриці
- **Пресети редагування**: Збереження обличчя, тіла або стилю при редагуванні через LoRA
- **Auto режим**: Аналізує ваги LoRA і автоматично обирає найкращий пресет
- **Per-slot контроль**: FluxLoraQuad нода з незалежним edit_mode для кожної LoRA
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

## Ноди

### FLUX LoRA Loader

Завантажувач однієї LoRA з підтримкою edit-mode.

| Вхід | Тип | Опис |
|---|---|---|
| `model` | MODEL | Модель FLUX.2 Klein / FLUX.1 |
| `lora_name` | dropdown | Файл LoRA з `models/loras` |
| `strength_model` | float | Глобальна сила LoRA (-20.0 до 20.0) |
| `auto_convert` | boolean | Конвертувати diffusers-формат у нативний FLUX |
| `edit_mode` | dropdown | Пресет редагування (див. нижче) |
| `balance` | float | 0.0 = повний ефект пресету, 1.0 = стандартна LoRA |
| `lora_name_override` | string (link) | Опціонально — перевизначає dropdown |
| `layer_strengths` | string (link) | Опціонально — per-layer JSON від Auto Strength |

### FLUX LoRA Quad

**4 слоти LoRA з незалежним edit_mode та balance для кожного.** Створена для робочих процесів редагування, де різні LoRA потребують різного рівня захисту.

Кожен слот має: `lora_name`, `strength`, `edit_mode`, `balance`, `enabled`, `auto_convert`.

Рекомендоване налаштування для редагування зображень:
```
Слот 1: editing LoRA       → edit_mode=Auto (або Preserve Body), strength=0.6-0.8
Слот 2: consistency LoRA   → edit_mode=Auto (або None),          strength=0.4-0.6
Слот 3: enhancer LoRA      → edit_mode=Auto (або None),          strength=0.2-0.4
Слот 4: (опціонально)      → за потребою
```

### FLUX LoRA Stack

До 10 LoRA послідовно з незалежною силою, перемикачем та авто-конвертацією на кожен слот. Підтримує глобальний `edit_mode` та `balance`. При Auto кожна LoRA аналізується окремо.

### FLUX LoRA Auto Strength

Зчитує тензори ваг LoRA та обчислює per-layer strength з реального тренувального сигналу у файлі. Double blocks аналізуються з img та txt потоками незалежно. Один регулятор: `global_strength`.

### FLUX LoRA Auto Loader

Самодостатня версія — аналіз і застосування в одній ноді. `model` на вхід, пропатчена `model` на вихід.

## Пресети редагування

При використанні LoRA для редагування зображень (наприклад, зміна одягу на фото) LoRA може пошкодити частини зображення, які ви хочете зберегти — найчастіше обличчя та ідентичність. Це відбувається тому, що FLUX.2 Klein обробляє зображення та текст по-різному на різних шарах:

- **Double blocks (0-7):** Потоки зображення та тексту ізольовані — вони не можуть самі по собі спричинити текст-керовану корупцію.
- **Single blocks (0-23):** Спільна крос-модальна обробка — саме тут текстовий промпт перезаписує reference зображення. Пізні single blocks (12-23) найагресивніші.

### Доступні пресети

| Пресет | Що робить | Коли використовувати |
|---|---|---|
| **None** | Стандартна LoRA (всі шари рівні) | Поведінка за замовчуванням |
| **Preserve Face** | Послаблює пізні single blocks | Редагування зі збереженням обличчя |
| **Preserve Body** | Агресивно послаблює середні+пізні single blocks | Збереження обличчя + пропорцій тіла (фігура, розмір грудей, талія) |
| **Style Only** | Знижує img потік у double blocks | Зміна стилю без структурних змін |
| **Edit Subject** | Помірний захист пізніх блоків, легке підсилення txt | Зміна одягу/об'єктів зі збереженням ідентичності |
| **Boost Prompt** | Підсилює txt потік і середні single blocks | Коли промпт недостатньо виконується |
| **Auto** | Аналізує ваги LoRA, автоматично обирає пресет + balance | Без налаштувань — рекомендовано для більшості |

### Auto режим

Коли `edit_mode` встановлено в **Auto**, нода аналізує розподіл ваг кожної LoRA і обирає оптимальний пресет:

- Високий тренувальний сигнал у пізніх single blocks (editing LoRA) → **Preserve Body**
- Помірний сигнал → **Preserve Face**
- Рівномірний розподіл (слайдери, enhancer-и) → **None**

Balance також обчислюється автоматично. В консолі видно який пресет було обрано:
```
[FLUX LoRA Quad] Slot 1: Auto → Preserve Body (balance=0.25)
[FLUX LoRA Quad] Slot 2: Auto → Preserve Face (balance=0.40)
```

### Повзунок Balance

Повзунок `balance` інтерполює між пресетом та стандартною поведінкою:
- **0.0** — повний ефект пресету (максимальний захист)
- **0.5** — середина між пресетом і стандартом
- **1.0** — стандартна LoRA (пресет не діє)

Edit mode працює поверх Auto Strength — можна комбінувати автоматичну ΔW-калібрацію з семантичними пресетами.

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
