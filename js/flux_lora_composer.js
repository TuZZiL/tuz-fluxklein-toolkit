import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const GOALS = ["Edit", "Restyle", "Generate"];
const SAFETY_LEVELS = ["Safe", "Balanced", "Strong"];
const ROLES = ["Main Edit", "Style", "Detail", "Identity", "Prompt Boost"];

const MIN_WIDTH = 500;
const PAD = 10;
const HEADER_H = 48;
const GLOBALS_H = 34;
const TOOLBAR_H = 28;
const SLOT_H = 56;
const EMPTY_H = 58;
const FOOTER_H = 20;
const ROW_GAP = 8;
const RADIUS = 8;

const STRENGTH_RANGE = { min: -5.0, max: 5.0, step: 0.05, precision: 2 };

const THEME = {
    canvas: "#0a0f17",
    surface0: "#101721",
    surface1: "#141d29",
    surface2: "#1a2433",
    surface3: "#233247",
    border: "#334355",
    borderStrong: "#4c657d",
    text: "#eef4ff",
    textSoft: "#afbdd0",
    textMuted: "#76879a",
    accent: "#8eb7e8",
    success: "#8bd3a8",
    successBg: "#163024",
    danger: "#ef92a4",
    dangerBg: "#341821",
};

const ROLE_BADGES = {
    "Main Edit": { fill: "#24364d", stroke: "#5579a1", text: "#dcecff", label: "Main" },
    "Style": { fill: "#3a2d1b", stroke: "#9b7c3d", text: "#ffe4ad", label: "Style" },
    "Detail": { fill: "#1f2a24", stroke: "#55795f", text: "#d2f2d7", label: "Detail" },
    "Identity": { fill: "#2f2245", stroke: "#7d67b8", text: "#e3d8ff", label: "ID" },
    "Prompt Boost": { fill: "#3a2130", stroke: "#a16188", text: "#ffd7eb", label: "Prompt" },
};

let _loraListCache = null;
let _loraListPromise = null;

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

function hideWidget(node, widget) {
    if (!widget) return;
    if (!widget.origType) widget.origType = widget.type;
    const originalSerialize = widget.serializeValue?.bind(widget);
    widget.type = "converted-widget";
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    widget.mouse = () => false;
    widget.serializeValue = () => originalSerialize ? originalSerialize() : widget.value;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function fitText(ctx, text, maxWidth) {
    if (!text || maxWidth <= 0) return "";
    if (ctx.measureText(text).width <= maxWidth) return text;
    let out = text;
    while (out.length > 1 && ctx.measureText(`${out}...`).width > maxWidth) {
        out = out.slice(0, -1);
    }
    return `${out}...`;
}

function snapToStep(value, step, min = -Infinity, max = Infinity, precision = 2) {
    const snapped = Math.round(Number(value) / step) * step;
    const clamped = Math.max(min, Math.min(max, snapped));
    return Number(clamped.toFixed(precision));
}

function normalizeSlot(initial) {
    return {
        enabled: initial?.enabled ?? true,
        lora: initial?.lora ?? "None",
        strength: typeof initial?.strength === "number" ? initial.strength : 1.0,
        role: ROLES.includes(initial?.role) ? initial.role : "Main Edit",
    };
}

function makeDefaultSlot(overrides = {}) {
    return normalizeSlot({
        enabled: true,
        lora: "None",
        strength: 1.0,
        role: "Main Edit",
        ...overrides,
    });
}

function parseSlotData(raw) {
    if (!raw || raw === "[]") return [];
    try {
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed.map((slot) => normalizeSlot(slot)) : [];
    } catch (e) {
        console.warn("[FluxLoraComposer] Failed to parse slot_data:", e);
        return [];
    }
}

function serializeSlots(slots) {
    return JSON.stringify(slots.map((slot) => ({
        enabled: slot.enabled,
        lora: slot.lora,
        strength: slot.strength,
        role: slot.role,
    })));
}

function updateWidgetValue(widget, value) {
    if (!widget) return;
    widget.value = value;
    widget.callback?.(value);
}

function pointInRect(x, y, rect) {
    return !!rect && x >= rect.x && x <= rect.x + rect.w && y >= rect.y && y <= rect.y + rect.h;
}

function menuItemsFromValues(values, currentValue, onSelect) {
    return values.map((value) => ({
        content: value === currentValue ? `• ${value}` : value,
        callback: () => onSelect(value),
    }));
}

function openMenu(event, items) {
    if (!event || typeof LiteGraph === "undefined" || !LiteGraph.ContextMenu) return;
    LiteGraph.closeAllContextMenus?.(window);
    new LiteGraph.ContextMenu(items, { event, scale: 1.0 }, window);
}

function strengthRatio(value) {
    return clamp((value - STRENGTH_RANGE.min) / (STRENGTH_RANGE.max - STRENGTH_RANGE.min), 0, 1);
}

function drawField(ctx, rect, label, value, alignRight = false) {
    roundRect(ctx, rect.x, rect.y, rect.w, rect.h, 6);
    ctx.fillStyle = THEME.surface2;
    ctx.fill();
    ctx.strokeStyle = THEME.border;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = THEME.textMuted;
    ctx.font = "500 10px 'IBM Plex Sans', ui-sans-serif";
    ctx.fillText(label, rect.x + 10, rect.y + 11);
    ctx.fillStyle = THEME.text;
    ctx.font = "600 11px 'JetBrains Mono', monospace";
    if (alignRight) {
        ctx.textAlign = "right";
        ctx.fillText(value, rect.x + rect.w - 18, rect.y + 25);
        ctx.textAlign = "left";
    } else {
        ctx.fillText(value, rect.x + 10, rect.y + 25);
    }
    ctx.fillStyle = THEME.textMuted;
    ctx.textAlign = "right";
    ctx.fillText("v", rect.x + rect.w - 10, rect.y + 25);
    ctx.textAlign = "left";
}

function drawToggle(ctx, rect, enabled) {
    roundRect(ctx, rect.x, rect.y, rect.w, rect.h, 6);
    ctx.fillStyle = enabled ? THEME.successBg : THEME.surface2;
    ctx.fill();
    ctx.strokeStyle = enabled ? "#4d8a66" : THEME.border;
    ctx.lineWidth = 1;
    ctx.stroke();
    const dotX = enabled ? rect.x + rect.w - rect.h + 4 : rect.x + 4;
    roundRect(ctx, dotX, rect.y + 4, rect.h - 8, rect.h - 8, 5);
    ctx.fillStyle = enabled ? THEME.success : THEME.textMuted;
    ctx.fill();
}

function drawActionPill(ctx, rect, label, tone = "neutral") {
    const palette = tone === "danger"
        ? { fill: THEME.dangerBg, stroke: "#7c3950", text: THEME.danger }
        : tone === "success"
            ? { fill: THEME.successBg, stroke: "#4d7d61", text: THEME.success }
            : { fill: THEME.surface2, stroke: THEME.border, text: THEME.textSoft };
    roundRect(ctx, rect.x, rect.y, rect.w, rect.h, 6);
    ctx.fillStyle = palette.fill;
    ctx.fill();
    ctx.strokeStyle = palette.stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = palette.text;
    ctx.font = "600 10px 'IBM Plex Sans', ui-sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, rect.x + rect.w / 2, rect.y + rect.h / 2 + 0.5);
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
}

function drawBadge(ctx, rect, role) {
    const badge = ROLE_BADGES[role] || ROLE_BADGES["Main Edit"];
    roundRect(ctx, rect.x, rect.y, rect.w, rect.h, rect.h / 2);
    ctx.fillStyle = badge.fill;
    ctx.fill();
    ctx.strokeStyle = badge.stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = badge.text;
    ctx.font = "600 10px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(badge.label, rect.x + rect.w / 2, rect.y + rect.h / 2 + 0.5);
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
}

function drawStrengthBar(ctx, rect, value, interactive = false) {
    const centerX = rect.x + rect.w / 2;
    const valueX = rect.x + rect.w * strengthRatio(value);
    roundRect(ctx, rect.x, rect.y, rect.w, rect.h, rect.h / 2);
    ctx.fillStyle = interactive ? "#101d31" : THEME.surface2;
    ctx.fill();
    ctx.strokeStyle = THEME.border;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.beginPath();
    ctx.moveTo(centerX, rect.y + 3);
    ctx.lineTo(centerX, rect.y + rect.h - 3);
    ctx.stroke();

    if (value >= 0) {
        const fillW = Math.max(0, valueX - centerX);
        if (fillW > 0) {
            roundRect(ctx, centerX, rect.y + 2, fillW, rect.h - 4, Math.max(3, (rect.h - 4) / 2));
            ctx.fillStyle = "#8eb7e8";
            ctx.fill();
        }
    } else {
        const fillW = Math.max(0, centerX - valueX);
        if (fillW > 0) {
            roundRect(ctx, valueX, rect.y + 2, fillW, rect.h - 4, Math.max(3, (rect.h - 4) / 2));
            ctx.fillStyle = "#ef92a4";
            ctx.fill();
        }
    }
}

async function getLoraList() {
    if (_loraListCache) return _loraListCache;
    if (_loraListPromise) return _loraListPromise;

    _loraListPromise = (async () => {
        try {
            const resp = await api.fetchApi("/object_info/FluxLoraLoader");
            const data = await resp.json();
            const info = data?.FluxLoraLoader;
            if (info?.input?.required?.lora_name) {
                _loraListCache = info.input.required.lora_name[0];
                return _loraListCache;
            }
        } catch (e) {
            // fall through
        }
        _loraListCache = ["None"];
        return _loraListCache;
    })();

    return _loraListPromise;
}

app.registerExtension({
    name: "Comfy.FluxLoraComposer",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FluxLoraComposer") return;

        const _onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = async function () {
            const result = _onNodeCreated?.apply(this, arguments);
            const node = this;
            const W = (name) => node.widgets?.find((widget) => widget.name === name);

            node._slots = parseSlotData(W("slot_data")?.value);
            node._loraList = await getLoraList();
            node._composerState = {
                bounds: {
                    header: {},
                    globals: {},
                    toolbar: {},
                    slots: [],
                },
                slider: null,
            };

            function refreshNodeSize() {
                const currentWidth = Math.max(node.size?.[0] ?? 0, MIN_WIDTH);
                const [, computedHeight] = node.computeSize();
                node.setSize([currentWidth, computedHeight]);
            }

            function markDirty() {
                refreshNodeSize();
                node.setDirtyCanvas(true, true);
            }

            function syncSlotData() {
                updateWidgetValue(W("slot_data"), serializeSlots(node._slots));
            }

            function setSlots(slots) {
                node._slots = slots.map((slot) => normalizeSlot(slot));
                syncSlotData();
                markDirty();
            }

            function updateSlot(index, patch) {
                if (index < 0 || index >= node._slots.length) return;
                node._slots[index] = normalizeSlot({ ...node._slots[index], ...patch });
                syncSlotData();
                markDirty();
            }

            function addSlot(initial = {}) {
                setSlots([...node._slots, makeDefaultSlot(initial)]);
            }

            function removeSlot(index) {
                setSlots(node._slots.filter((_, slotIndex) => slotIndex !== index));
            }

            function toggleAll(enabled) {
                setSlots(node._slots.map((slot) => ({ ...slot, enabled })));
            }

            function resetSlots() {
                setSlots(node._slots.map(() => makeDefaultSlot()));
            }

            function setGlobal(name, value) {
                updateWidgetValue(W(name), value);
                markDirty();
            }

            function openSelector(index, field, values, event) {
                const slot = node._slots[index];
                if (!slot) return;
                openMenu(event, menuItemsFromValues(values, slot[field], (value) => {
                    updateSlot(index, { [field]: value });
                }));
            }

            function startSlider(index, rect, mouseX) {
                node._composerState.slider = { index, rect };
                updateSlider(mouseX);
            }

            function updateSlider(mouseX) {
                const slider = node._composerState.slider;
                if (!slider) return;
                const ratio = clamp((mouseX - slider.rect.x) / slider.rect.w, 0, 1);
                const raw = STRENGTH_RANGE.min + ratio * (STRENGTH_RANGE.max - STRENGTH_RANGE.min);
                updateSlot(slider.index, {
                    strength: snapToStep(raw, STRENGTH_RANGE.step, STRENGTH_RANGE.min, STRENGTH_RANGE.max, STRENGTH_RANGE.precision),
                });
            }

            function endSlider() {
                if (!node._composerState.slider) return false;
                node._composerState.slider = null;
                node.setDirtyCanvas(true, true);
                return true;
            }

            function activeSlots() {
                return node._slots.filter((slot) => slot.enabled && slot.lora !== "None" && Math.abs(slot.strength) > 1e-8);
            }

            function currentSummary() {
                const active = activeSlots();
                const main = active.find((slot) => slot.role === "Main Edit")
                    || active.slice().sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength))[0];
                return {
                    active: active.length,
                    main: main?.lora ?? "None",
                    support: Math.max(0, active.length - (main ? 1 : 0)),
                };
            }

            function computeWidgetHeight() {
                let height = PAD + HEADER_H + ROW_GAP + GLOBALS_H + ROW_GAP + TOOLBAR_H + ROW_GAP;
                height += node._slots.length ? node._slots.length * (SLOT_H + ROW_GAP) - ROW_GAP : EMPTY_H;
                height += ROW_GAP + FOOTER_H + PAD;
                return height;
            }

            function drawHeader(ctx, x, y, w) {
                roundRect(ctx, x, y, w, HEADER_H, RADIUS);
                ctx.fillStyle = THEME.surface0;
                ctx.fill();
                ctx.strokeStyle = THEME.border;
                ctx.lineWidth = 1;
                ctx.stroke();

                const summary = currentSummary();
                ctx.fillStyle = THEME.text;
                ctx.font = "700 12px 'IBM Plex Sans', ui-sans-serif";
                ctx.fillText("FLUX LoRA Composer", x + 14, y + 17);

                ctx.fillStyle = THEME.textSoft;
                ctx.font = "600 10px 'JetBrains Mono', monospace";
                ctx.fillText(
                    `Main: ${fitText(ctx, summary.main, w - 220)}`,
                    x + 14,
                    y + 35
                );

                ctx.textAlign = "right";
                ctx.fillText(
                    `Active ${summary.active} | Support ${summary.support}`,
                    x + w - 14,
                    y + 17
                );
                ctx.fillText(
                    `${W("goal")?.value ?? "Edit"} / ${W("safety")?.value ?? "Balanced"}`,
                    x + w - 14,
                    y + 35
                );
                ctx.textAlign = "left";
            }

            function drawGlobals(ctx, x, y, w) {
                roundRect(ctx, x, y, w, GLOBALS_H, RADIUS);
                ctx.fillStyle = THEME.surface0;
                ctx.fill();
                ctx.strokeStyle = THEME.border;
                ctx.lineWidth = 1;
                ctx.stroke();

                const goalRect = { x: x + 10, y: y + 5, w: 124, h: 24 };
                const safetyRect = { x: x + 142, y: y + 5, w: 124, h: 24 };
                const normalizeRect = { x: x + w - 160, y: y + 5, w: 150, h: 24 };
                node._composerState.bounds.globals = { goal: goalRect, safety: safetyRect, normalize: normalizeRect };

                drawField(ctx, goalRect, "Goal", W("goal")?.value ?? "Edit");
                drawField(ctx, safetyRect, "Safety", W("safety")?.value ?? "Balanced");

                roundRect(ctx, normalizeRect.x, normalizeRect.y, normalizeRect.w, normalizeRect.h, 6);
                ctx.fillStyle = THEME.surface2;
                ctx.fill();
                ctx.strokeStyle = THEME.border;
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.fillStyle = THEME.textMuted;
                ctx.font = "500 10px 'IBM Plex Sans', ui-sans-serif";
                ctx.textBaseline = "middle";
                ctx.fillText("Auto-normalize", normalizeRect.x + 10, normalizeRect.y + normalizeRect.h / 2 + 0.5);
                ctx.textBaseline = "alphabetic";
                drawToggle(ctx, { x: normalizeRect.x + normalizeRect.w - 42, y: normalizeRect.y + 4, w: 30, h: 16 }, !!W("auto_normalize")?.value);
            }

            function drawToolbar(ctx, x, y, w) {
                roundRect(ctx, x, y, w, TOOLBAR_H, RADIUS);
                ctx.fillStyle = THEME.surface0;
                ctx.fill();
                ctx.strokeStyle = THEME.border;
                ctx.lineWidth = 1;
                ctx.stroke();

                const buttons = [
                    { key: "add", label: "+ Add LoRA", tone: "neutral", w: 98 },
                    { key: "enableAll", label: "Enable All", tone: "success", w: 82 },
                    { key: "disableAll", label: "Disable All", tone: "neutral", w: 88 },
                    { key: "reset", label: "Reset", tone: "danger", w: 58 },
                ];

                node._composerState.bounds.toolbar = {};
                let bx = x + 10;
                for (const button of buttons) {
                    const rect = { x: bx, y: y + 4, w: button.w, h: 20 };
                    node._composerState.bounds.toolbar[button.key] = rect;
                    drawActionPill(ctx, rect, button.label, button.tone);
                    bx += button.w + 8;
                }
            }

            function drawSlot(ctx, slot, index, x, y, w) {
                const rowRect = { x, y, w, h: SLOT_H };
                roundRect(ctx, x, y, w, SLOT_H, RADIUS);
                ctx.fillStyle = THEME.surface1;
                ctx.fill();
                ctx.strokeStyle = slot.enabled ? THEME.border : "#2a3441";
                ctx.lineWidth = 1;
                ctx.stroke();

                const toggleRect = { x: x + 10, y: y + 18, w: 34, h: 18 };
                const loraRect = { x: x + 54, y: y + 9, w: 180, h: 24 };
                const roleRect = { x: x + 242, y: y + 9, w: 112, h: 24 };
                const badgeRect = { x: x + 364, y: y + 12, w: 66, h: 18 };
                const strengthRect = { x: x + 54, y: y + 35, w: w - 140, h: 14 };
                const removeRect = { x: x + w - 56, y: y + 18, w: 36, h: 20 };

                node._composerState.bounds.slots[index] = {
                    row: rowRect,
                    toggle: toggleRect,
                    lora: loraRect,
                    role: roleRect,
                    strength: strengthRect,
                    remove: removeRect,
                };

                drawToggle(ctx, toggleRect, slot.enabled);
                drawField(ctx, loraRect, "LoRA", fitText(ctx, slot.lora || "None", 140));
                drawField(ctx, roleRect, "Role", slot.role);
                drawBadge(ctx, badgeRect, slot.role);
                drawActionPill(ctx, removeRect, "Remove", "danger");

                ctx.fillStyle = THEME.textMuted;
                ctx.font = "500 10px 'IBM Plex Sans', ui-sans-serif";
                ctx.fillText("Strength", x + 10, y + 46);
                drawStrengthBar(ctx, strengthRect, slot.strength, node._composerState.slider?.index === index);
                ctx.fillStyle = THEME.text;
                ctx.font = "600 10px 'JetBrains Mono', monospace";
                ctx.textAlign = "right";
                ctx.fillText(slot.strength.toFixed(2), strengthRect.x + strengthRect.w, y + 47);
                ctx.textAlign = "left";
            }

            function drawEmpty(ctx, x, y, w) {
                roundRect(ctx, x, y, w, EMPTY_H, RADIUS);
                ctx.fillStyle = THEME.surface1;
                ctx.fill();
                ctx.strokeStyle = THEME.border;
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.fillStyle = THEME.textSoft;
                ctx.font = "600 11px 'IBM Plex Sans', ui-sans-serif";
                ctx.fillText("Add 2-4 LoRAs, assign roles, then let Composer handle the mix.", x + 14, y + 25);
                ctx.fillStyle = THEME.textMuted;
                ctx.font = "500 10px 'IBM Plex Sans', ui-sans-serif";
                ctx.fillText("Main Edit drives the change. Style/Detail/Identity/Prompt Boost stay supportive.", x + 14, y + 43);
            }

            function drawFooter(ctx, x, y) {
                ctx.fillStyle = THEME.textMuted;
                ctx.font = "500 10px 'IBM Plex Sans', ui-sans-serif";
                ctx.fillText("Compact role-based composition. Internal routing stays hidden by design.", x, y + 14);
            }

            function drawPanel(ctx, nodeRef, width, y) {
                const w = Math.max(width - PAD * 2, MIN_WIDTH - PAD * 2);
                const x = PAD;
                let cursorY = y + PAD;

                roundRect(ctx, x - 2, cursorY - 2, w + 4, computeWidgetHeight() - PAD * 2 + 4, 12);
                ctx.fillStyle = THEME.canvas;
                ctx.fill();

                drawHeader(ctx, x, cursorY, w);
                cursorY += HEADER_H + ROW_GAP;

                drawGlobals(ctx, x, cursorY, w);
                cursorY += GLOBALS_H + ROW_GAP;

                drawToolbar(ctx, x, cursorY, w);
                cursorY += TOOLBAR_H + ROW_GAP;

                node._composerState.bounds.slots = [];
                if (!node._slots.length) {
                    drawEmpty(ctx, x, cursorY, w);
                    cursorY += EMPTY_H;
                } else {
                    node._slots.forEach((slot, index) => {
                        drawSlot(ctx, slot, index, x, cursorY, w);
                        cursorY += SLOT_H + ROW_GAP;
                    });
                    cursorY -= ROW_GAP;
                }

                cursorY += ROW_GAP;
                drawFooter(ctx, x, cursorY);
            }

            function handleToolbarClick(key) {
                if (key === "add") addSlot();
                else if (key === "enableAll") toggleAll(true);
                else if (key === "disableAll") toggleAll(false);
                else if (key === "reset") resetSlots();
            }

            function widgetMouse(event, pos) {
                const [mx, my] = pos;

                if (event.type === "pointermove") {
                    if (node._composerState.slider) {
                        updateSlider(mx);
                        return true;
                    }
                    return false;
                }

                if (event.type === "pointerup" || event.type === "pointercancel") {
                    return endSlider();
                }

                if (event.type !== "pointerdown") return false;

                const globals = node._composerState.bounds.globals || {};
                const toolbar = node._composerState.bounds.toolbar || {};

                if (pointInRect(mx, my, globals.goal)) {
                    openMenu(event, menuItemsFromValues(GOALS, W("goal")?.value, (value) => setGlobal("goal", value)));
                    return true;
                }
                if (pointInRect(mx, my, globals.safety)) {
                    openMenu(event, menuItemsFromValues(SAFETY_LEVELS, W("safety")?.value, (value) => setGlobal("safety", value)));
                    return true;
                }
                if (pointInRect(mx, my, globals.normalize)) {
                    setGlobal("auto_normalize", !W("auto_normalize")?.value);
                    return true;
                }

                for (const [key, rect] of Object.entries(toolbar)) {
                    if (pointInRect(mx, my, rect)) {
                        handleToolbarClick(key);
                        return true;
                    }
                }

                for (let index = 0; index < node._composerState.bounds.slots.length; index++) {
                    const bounds = node._composerState.bounds.slots[index];
                    const slot = node._slots[index];
                    if (!slot) continue;

                    if (pointInRect(mx, my, bounds.toggle)) {
                        updateSlot(index, { enabled: !slot.enabled });
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.lora)) {
                        openSelector(index, "lora", ["None", ...(node._loraList || [])], event);
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.role)) {
                        openSelector(index, "role", ROLES, event);
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.strength)) {
                        startSlider(index, bounds.strength, mx);
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.remove)) {
                        removeSlot(index);
                        return true;
                    }
                }

                return false;
            }

            const uiWidget = {
                name: "flux_lora_composer_ui",
                type: "flux_lora_composer_ui",
                options: { serialize: false },
                computeSize(width) {
                    return [Math.max(width, MIN_WIDTH), computeWidgetHeight()];
                },
                draw(ctx, nodeRef, width, y) {
                    drawPanel(ctx, nodeRef, width, y);
                },
                mouse(event, pos) {
                    return widgetMouse(event, pos);
                },
            };

            node.addCustomWidget(uiWidget);

            setTimeout(() => {
                hideWidget(node, W("goal"));
                hideWidget(node, W("safety"));
                hideWidget(node, W("auto_normalize"));
                hideWidget(node, W("auto_convert"));
                hideWidget(node, W("slot_data"));
                refreshNodeSize();
                node.setDirtyCanvas(true, true);
            }, 0);

            const slotWidget = W("slot_data");
            if (slotWidget) {
                let internalValue = slotWidget.value ?? "[]";
                const originalCallback = slotWidget.callback?.bind(slotWidget);
                Object.defineProperty(slotWidget, "value", {
                    get() {
                        return internalValue;
                    },
                    set(value) {
                        internalValue = value;
                        originalCallback?.(value);
                        node._slots = parseSlotData(value);
                        markDirty();
                    },
                    configurable: true,
                });
            }

            const origConfigure = node.onConfigure?.bind(node);
            node.onConfigure = function (config) {
                origConfigure?.(config);
                setTimeout(async () => {
                    if (!node._loraList) node._loraList = await getLoraList();
                    node._slots = parseSlotData(W("slot_data")?.value);
                    markDirty();
                }, 50);
            };

            if (node.size[0] < MIN_WIDTH) node.size[0] = MIN_WIDTH;
            markDirty();

            return result;
        };
    },
});
