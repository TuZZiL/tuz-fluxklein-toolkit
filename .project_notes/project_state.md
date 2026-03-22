---
name: klein9b-lora-project-state
description: FLUX.2 Klein 9B LoRA loader project - architecture knowledge, format compatibility, edit presets, scheduling, and current state
type: project
---

## Project: Comfyui-flux2klein-Lora-loader (fork)

**Repo:** https://github.com/TuZZiL/Comfyui-flux2klein-Lora-loader
**Original:** https://github.com/capitan01R/Comfyui-flux2klein-Lora-loader
**Local:** `F:\x_other_mat\AI_project\Comfyui-flux2klein-Lora-loader\repo_clean-claude`
**Remote ComfyUI:** SSH до ephemeral GPU pod (credentials змінюються щоразу)

---

## FLUX.2 Klein 9B Architecture (verified)

```
8 double_blocks (0-7) — img і txt потоки ІЗОЛЬОВАНІ
  img: img_attn.qkv [12288,4096], img_attn.proj [4096,4096], img_mlp.0/2
  txt: txt_attn.qkv [12288,4096], txt_attn.proj [4096,4096], txt_mlp.0/2

24 single_blocks (0-23) — СПІЛЬНА обробка img+txt (concatenated)
  linear1 [36864,4096] = fused Q+K+V+proj_mlp
  linear2 [4096,16384] = proj_out

dim=4096
```

**Ключовий інсайт (з comfyUI-Realtime-Lora дослідження):**
- Double blocks НЕ спричиняють текст-керовану корупцію (потоки ізольовані)
- Single blocks — де текст перезаписує reference image
- **Пізні single blocks (12-23) найагресивніше впливають на ідентичність/reference**
- Середні single blocks (8-15) — пропорції тіла
- Послаблення sb12-sb23 зберігає обличчя при editing

**НЕ плутати зі стандартним FLUX:** Klein має 8+24 блоки, стандартний FLUX 19+38

---

## Чотири формати / варіанти LoRA ключів (всі підтримуються)

### 1. Native формат (ComfyUI)
```
diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight
```
Проходить напряму. Приклад: `klein_slider_anatomy.safetensors`

### 2. Standard diffusers формат
```
transformer_blocks.0.attn.to_q.lora_A.weight  (окремі Q/K/V)
```
Потребує block-diagonal QKV fusion.

### 3. Musubi Tuner / PEFT формат (Modelscope)
```
single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.default.weight
transformer_blocks.0.ff.linear_in.lora_A.default.weight
```
Відмінності: `.default.` в іменах, `to_qkv_mlp_proj` (вже fused), `ff.linear_in/out`.
Приклади: `removedress6000_6.safetensors`, `ChrisHendriks_v3_3.safetensors`

### 4. Alias dotted формат
```
transformer.transformer_blocks.0.attn.to_q.lora.down.weight
```
Відмінності: `.lora.down/.lora.up` замість `.lora_down/.lora_up` або `.lora_A/.lora_B`.
Приклад: `f2k_consis.safetensors`

---

## Наші ноди (поточний стан — consolidated v2)

### 1. FluxLoraLoader — single LoRA з повним контролем
- Інтерактивний граф-віджет (per-layer strength bars)
- `use_case`: `Edit` / `Generate` — окрема вісь поведінки для `Auto`
- `auto_strength` toggle — автоматичний розрахунок per-layer strengths через ΔW аналіз
- `auto_strength` тепер шле layer JSON назад у frontend і заповнює граф автоматично
- compatibility badge `Compat X/Y` прямо на graph widget + деталі в логах
- graph preset buttons: `Reset`, `Global`, `Face`, `Body`, `Style`
- edit_mode dropdown (6 пресетів + Auto) + balance slider
- Підтримує всі 4 формати/варіанти LoRA
- Edit presets застосовуються і до native `lora_up/lora_down`, не лише до `.lora_B`
- `strength` діапазон: -5.0..5.0

### 2. FluxLoraMulti — dynamic multi-slot (rgthree-style)
- "+" Add LoRA кнопка для додавання слотів
- Per-slot: enabled, lora, strength, `use_case`, edit_mode, balance
- Duplicate / Collapse / Reorder (`up/down`) прямо в header слота
- Fully JS-driven slots — slot_data серіалізується як JSON
- `collapsed` теж серіалізується в slot_data і відновлюється після reload
- Замінює Stack + Quad

### 3. FluxLoraScheduled — per-step temporal scheduling
- Використовує ComfyUI Hook Keyframes (native API)
- Приймає MODEL + CONDITIONING, повертає MODEL + CONDITIONING
- Має `use_case`: `Edit` / `Generate` для `Auto` пресету
- Вбудований SetCondHooks — хуки прикріплюються до conditioning автоматично
- 5 профілів: Constant, Fade Out, Fade In, Strong Start, Pulse

### Видалені ноди (consolidated):
- ~~FluxLoraStack~~ → FluxLoraMulti
- ~~FluxLoraQuad~~ → FluxLoraMulti
- ~~FluxLoraAutoStrength~~ → FluxLoraLoader (auto_strength toggle)
- ~~FluxLoraAutoLoader~~ → FluxLoraLoader (auto_strength toggle)
- ~~FluxSetCondHooks~~ → FluxLoraScheduled (вбудовано)

---

## Auto-Preset евристика (edit_presets.py → auto_select_preset)

Поточна евристика стала **use-case aware**: `Edit` і `Generate` мають різну філософію вибору `Auto`.

### `use_case = Edit`
- сильна домінація пізніх `single_blocks` → `Preserve Body`
- помірна домінація late/mid `single_blocks` → `Preserve Face`
- `img >> txt` у double blocks при спокійних singles → `Style Only`
- рівномірна full-coverage LoRA → `Preserve Face`
- дуже м’яка / sparse structural LoRA → `None`
- інакше → `Preserve Face`

### `use_case = Generate`
- style-like (`img >> txt`, calm singles) → `Style Only`
- дуже aggressive late profile → `Preserve Face`
- broad / full-coverage / uniform LoRA → `None`
- інакше → `None`

Balance = `max(0.20, min(0.60, 0.70 - 0.25 * max_ratio))`

Додатково:
- для `None` balance піднімається мінімум до `0.55`
- для `Style Only` balance піднімається мінімум до `0.35`
- для `Generate + None` balance піднімається мінімум до `0.50`

Практичний ефект:
- broad text2image LoRA, які юзер застосовує в editing, тепер рідше класифікуються як `None`
- `Auto` краще страхує reference/identity без окремих warning'ів у UI
- той самий файл LoRA може поводитись вільніше в text-to-image і консервативніше в edit workflow

---

## GGUF сумісність

ComfyUI-GGUF (city96) повністю сумісний:
- `GGUFModelPatcher` наслідує стандартний `ModelPatcher`
- `add_patches()` і `add_hook_patches()` працюють
- Патчі застосовуються відкладено при dequantization
- Обмеження: кілька LoRA сповільнюють інференс (~3x при 4 LoRA)

---

## Архітектурні рішення

1. **`_apply_edit_multipliers` vs `_apply_layer_strengths`**: Окремий метод для edit presets бо `_apply_layer_strengths` ділить на `global_strength` (баг при strength≠1.0). Edit multipliers множать напряму: `tensor * multiplier`.

2. **`_resolve_edit_mode` helper**: Єдина точка входу для resolve Auto → конкретний пресет. Використовується в Loader, Stack, Quad.

3. **Hook-based scheduling**: Використовує native ComfyUI `WeightHook` + `HookKeyframeGroup` замість хакерських re-patch підходів. Patches реєструються через `add_hook_patches`, keyframes визначають strength curve.

4. **Shared compatibility helpers**: Винесено `lora_compat.py` для єдиної нормалізації ключів, inventory LoRA modules і compatibility report.

5. **`lora_meta.py` fixes**: `parse_lora_key` винесено в shared helper, додано підтримку `.lora.down/.lora.up`, а `analyse_for_node` тепер має in-memory cache по `(path, size, mtime_ns)`.

6. **Compatibility report**: Loader будує inventory complete/incomplete LoRA modules, звіряє їх з `key_map`, логуючи `matched/skipped/incomplete`, і показує компактний badge в Loader UI.

7. **Use case routing**: `use_case` проходить через Loader, Scheduled і per-slot Multi, впливаючи тільки на `edit_mode=Auto`; ручні пресети не залежать від нього.

8. **Graph preset masks**: кнопки `Face` / `Body` / `Style` пишуть прямо в `layer_strengths` і повторно використовують shape існуючих пресетів без нового storage format.

9. **Multi rerender strategy**: `FluxLoraMulti` тепер перебудовує slot widgets із `node._slots`, що спрощує duplicate/collapse/reorder і гарантує коректне restore після workflow reload.

10. **Auto-strength UI sync**: Loader використовує hidden `UNIQUE_ID` і `PromptServer.send_sync("flux_lora.auto_strength", ...)`, а фронтенд оновлює hidden `layer_strengths` widget і граф без ручного refresh.

11. **Repo metadata alignment**: `pyproject.toml` і publish workflow вирівняні під фактичний `origin` (`TuZZiL`). `PublisherId` поки залишено як `capitan01r`, бо це може бути окремий Comfy Registry ідентифікатор.

---

## Recent fixes (2026-03-22)

- Виправлено застосування `edit_mode` до native LoRA з ключами `lora_up/lora_down`
- Виправлено `auto_strength` UX: граф тепер може авто-заповнюватися після backend аналізу
- Додано підтримку alias формату `.lora.down/.lora.up`
- Додано compatibility report + badge `Compat X/Y` у `FluxLoraLoader`
- Додано in-memory cache для `analyse_for_node()`
- Зроблено `Auto` більш консервативним для broad text2image LoRA в edit use-case
- Додано `use_case = Edit / Generate` у Loader + Scheduled і per-slot у Multi
- Додано graph preset buttons: `Reset`, `Global`, `Face`, `Body`, `Style`
- Додано `Duplicate / Collapse / Reorder` у `FluxLoraMulti`
- Вирівняно repo URL metadata і GitHub publish owner check під `TuZZiL`

---

## Корисні посилання

- **comfyUI-Realtime-Lora** — Klein 9B layer mapping, debiaser: https://github.com/shootthesound/comfyUI-Realtime-Lora
- **ComfyUI-LoRABlockWeight** — per-block weight control: https://github.com/bhvbhushan/ComfyUI-LoRABlockWeight
- **ComfyUI-GGUF** — GGUF loader: https://github.com/city96/ComfyUI-GGUF
- **FLUX-Makeup** — identity preservation: https://arxiv.org/html/2508.05069v1

---

## TODO

- [x] Консолідувати 7 нод → 3 (Loader, Multi, Scheduled)
- [x] Виправити `edit_mode` для native LoRA (`lora_up/lora_down`)
- [x] Повернути `auto_strength` значення назад у UI граф
- [x] Додати alias support `.lora.down/.lora.up`
- [x] Додати compatibility report у Loader
- [x] Додати cache для `analyse_for_node()`
- [x] Зробити `Auto` консервативнішим для t2i LoRA в editing
- [x] Додати `use_case = Edit / Generate` у всі 3 ноди
- [x] Додати graph preset buttons у Loader
- [x] Додати Duplicate / Collapse / Reorder у `FluxLoraMulti`
- [x] Узгодити repo metadata (`README`/`pyproject`/workflow) з фактичним remote
- [ ] Протестувати FluxLoraMulti JS widget в ComfyUI (duplicate/collapse/reorder/save/restore)
- [ ] Емпірично протестувати пресети та scheduling на реальних генераціях
- [ ] Перевірити нову `Auto` евристику на реальних t2i LoRA у ComfyUI edit workflow
- [ ] Перевірити `use_case=Generate` vs `Edit` на реальних LoRA в Loader / Multi / Scheduled
- [ ] Перевірити FluxLoraScheduled з GGUF (hook patches + dequantization)
- [ ] Оновити gen_workflow.py під нову структуру нод
- [ ] Оновити README.md та README_UA.md
- [ ] Підтримка Klein 4B (5 double + 20 single)
- [ ] Per-Step Edit Mode (різний edit_mode на різних кроках — потребує custom hook)
