// FLUX LoRA Multi — Card UI widget
//
// Replaces the widget-per-field stack with a compact card layout that keeps
// the original slot_data JSON contract intact.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EDIT_MODES = ["None", "Preserve Face", "Preserve Body", "Style Only", "Edit Subject", "Boost Prompt", "Auto"];
const USE_CASES = ["Edit", "Generate"];

const MIN_WIDTH = 520;
const PAD = 10;
const HEADER_H = 48;
const TOOLBAR_H = 28;
const HINT_H = 20;
const CARD_SUMMARY_H = 68;
const CARD_EXPANDED_H = 256;
const EMPTY_H = 70;
const ROW_GAP = 8;
const CARD_GAP = 8;
const RADIUS = 8;

const STRENGTH_RANGE = { min: -5.0, max: 5.0, step: 0.05, precision: 2 };
const BALANCE_RANGE = { min: 0.0, max: 1.0, step: 0.05, precision: 2 };
const ANATOMY_STRENGTH_RANGE = { min: 0.0, max: 1.0, step: 0.05, precision: 2 };
const DEFAULT_ANATOMY_PROFILES = [
    "None",
    "Balanced Identity",
    "Undress Safe",
    "Undress Body Lock",
    "Cloth Swap Flexible",
    "Robot Frame Lock",
    "Armor Hard Surface",
    "Anime Stylized Lock",
    "Texture Only",
    "Prompt Freedom",
    "Custom",
];

const THEME = {
    canvas: "#151515",
    surface0: "#181818",
    surface1: "#202020",
    surface2: "#252525",
    surface3: "#2b2b2b",
    border: "#333333",
    borderStrong: "#474747",
    text: "#d7d7d7",
    textSoft: "#b8b8b8",
    textMuted: "#8e8e8e",
    accent: "#d0d0d0",
    accentSoft: "#f0f0f0",
    success: "#d6d6d6",
    successBg: "#242424",
    warning: "#d4d4d4",
    warningBg: "#242424",
    danger: "#e8e8e8",
    dangerBg: "#242424",
    shadow: "rgba(0,0,0,0.14)",
};

const BADGES = {
    "None": { label: "NONE", fill: "#232323", stroke: "#3a3a3a", text: "#9d9d9d" },
    "Preserve Face": { label: "FACE", fill: "#252525", stroke: "#4b4b4b", text: "#d7d7d7" },
    "Preserve Body": { label: "BODY", fill: "#252525", stroke: "#4b4b4b", text: "#d7d7d7" },
    "Style Only": { label: "STYLE", fill: "#252525", stroke: "#4b4b4b", text: "#d7d7d7" },
    "Edit Subject": { label: "EDIT", fill: "#252525", stroke: "#4b4b4b", text: "#d7d7d7" },
    "Boost Prompt": { label: "PROMPT", fill: "#252525", stroke: "#4b4b4b", text: "#d7d7d7" },
    "Auto": { label: "AUTO", fill: "#2a2a2a", stroke: "#5a5a5a", text: "#f0f0f0" },
};

let _loraListCache = null;
let _loraListPromise = null;
let _anatomyProfilesCache = null;
let _anatomyProfilesPromise = null;

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

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

function hideWidget(node, widget, { preserveType = false } = {}) {
    if (!widget) return;
    if (!widget.origType) widget.origType = widget.type;
    const originalSerialize = widget.serializeValue?.bind(widget);
    // preserveType=true keeps the widget in widgets_values during workflow save.
    // Use for hidden data widgets like slot_data that must survive reload.
    if (!preserveType) {
        widget.type = "converted-widget";
    }
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    widget.mouse = () => false;
    widget.serializeValue = () => originalSerialize ? originalSerialize() : widget.value;
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
        use_case: initial?.use_case ?? "Edit",
        edit_mode: initial?.edit_mode ?? "None",
        balance: typeof initial?.balance === "number" ? initial.balance : 0.5,
        anatomy_profile: initial?.anatomy_profile ?? "None",
        anatomy_strength: typeof initial?.anatomy_strength === "number" ? initial.anatomy_strength : 0.65,
        anatomy_strict_zero: initial?.anatomy_strict_zero ?? false,
        anatomy_custom_json: initial?.anatomy_custom_json ?? "",
        collapsed: initial?.collapsed ?? true,
    };
}

function makeDefaultSlot(overrides = {}) {
    return normalizeSlot({
        enabled: true,
        lora: "None",
        strength: 1.0,
        use_case: "Edit",
        edit_mode: "None",
        balance: 0.5,
        anatomy_profile: "None",
        anatomy_strength: 0.65,
        anatomy_strict_zero: false,
        anatomy_custom_json: "",
        collapsed: true,
        ...overrides,
    });
}

function parseSlotData(raw) {
    if (!raw || raw === "[]") return [];
    try {
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) return [];
        return parsed.map((slot) => normalizeSlot(slot));
    } catch (e) {
        console.warn("[FluxLoraMulti] Failed to parse slot_data:", e);
        return [];
    }
}

function serializeSlots(slots) {
    return JSON.stringify(slots.map((slot) => ({
        enabled: slot.enabled,
        lora: slot.lora,
        strength: slot.strength,
        use_case: slot.use_case,
        edit_mode: slot.edit_mode,
        balance: slot.balance,
        anatomy_profile: slot.anatomy_profile,
        anatomy_strength: slot.anatomy_strength,
        anatomy_strict_zero: slot.anatomy_strict_zero,
        anatomy_custom_json: slot.anatomy_custom_json,
        collapsed: slot.collapsed,
    })));
}

function shortLoraName(name) {
    if (!name || name === "None") return "No LoRA selected";
    return name;
}

function activeSlotCount(slots) {
    return slots.filter((slot) => slot.enabled && slot.lora !== "None").length;
}

function anatomySummary(slot, short = false) {
    const profile = slot?.anatomy_profile || "None";
    if (profile === "None") return short ? "Anat none" : "Anatomy: None";
    const strict = slot?.anatomy_strict_zero ? " strict" : "";
    const base = `${profile} ${Number(slot?.anatomy_strength ?? 0.65).toFixed(2)}${strict}`;
    return short ? `Anat ${base}` : `Anatomy: ${base}`;
}

function badgeForMode(mode) {
    return BADGES[mode] ?? BADGES["None"];
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

function drawBadge(ctx, rect, badge) {
    roundRect(ctx, rect.x, rect.y, rect.w, rect.h, rect.h / 2);
    ctx.fillStyle = badge.fill;
    ctx.fill();
    ctx.strokeStyle = badge.stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = badge.text;
    ctx.font = "600 10px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(badge.label, rect.x + rect.w / 2, rect.y + rect.h / 2 + 0.5);
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
}

function drawToggle(ctx, rect, enabled) {
    roundRect(ctx, rect.x, rect.y, rect.w, rect.h, 6);
    ctx.fillStyle = enabled ? THEME.surface3 : THEME.surface2;
    ctx.fill();
    ctx.strokeStyle = enabled ? THEME.borderStrong : THEME.border;
    ctx.lineWidth = 1;
    ctx.stroke();
    const dotX = enabled ? rect.x + rect.w - rect.h + 4 : rect.x + 4;
    roundRect(ctx, dotX, rect.y + 4, rect.h - 8, rect.h - 8, 5);
    ctx.fillStyle = enabled ? THEME.text : THEME.textMuted;
    ctx.fill();
}

function drawSegmentButton(ctx, rect, label, active = false) {
    roundRect(ctx, rect.x, rect.y, rect.w, rect.h, 6);
    ctx.fillStyle = active ? THEME.surface3 : THEME.surface1;
    ctx.fill();
    ctx.strokeStyle = active ? THEME.borderStrong : THEME.border;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = active ? THEME.accentSoft : THEME.textSoft;
    ctx.font = active ? "700 10px sans-serif" : "600 10px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, rect.x + rect.w / 2, rect.y + rect.h / 2 + 0.5);
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
}

function drawMutedPill(ctx, rect, label, value) {
    roundRect(ctx, rect.x, rect.y, rect.w, rect.h, 6);
    ctx.fillStyle = THEME.surface1;
    ctx.fill();
    ctx.strokeStyle = THEME.border;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = THEME.textMuted;
    ctx.font = "500 10px sans-serif";
    ctx.fillText(label, rect.x + 10, rect.y + 8);
    ctx.fillStyle = THEME.text;
    ctx.font = "600 10px monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(value, rect.x + rect.w - 10, rect.y + rect.h / 2 + 3);
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
}

function drawActionPill(ctx, rect, label, tone = "neutral") {
    const palette = tone === "danger"
        ? { fill: THEME.dangerBg, stroke: THEME.borderStrong, text: THEME.text }
        : tone === "success"
            ? { fill: THEME.surface3, stroke: THEME.borderStrong, text: THEME.text }
            : { fill: THEME.surface2, stroke: THEME.border, text: THEME.textSoft };
    roundRect(ctx, rect.x, rect.y, rect.w, rect.h, 6);
    ctx.fillStyle = palette.fill;
    ctx.fill();
    ctx.strokeStyle = palette.stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = palette.text;
    ctx.font = "600 10px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, rect.x + rect.w / 2, rect.y + rect.h / 2 + 0.5);
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
}

function drawIconButton(ctx, rect, glyph) {
    roundRect(ctx, rect.x, rect.y, rect.w, rect.h, 6);
    ctx.fillStyle = THEME.surface2;
    ctx.fill();
    ctx.strokeStyle = THEME.border;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = THEME.textMuted;
    ctx.font = "600 11px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(glyph, rect.x + rect.w / 2, rect.y + rect.h / 2 + 0.5);
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
}

function strengthRatio(value) {
    return clamp((value - STRENGTH_RANGE.min) / (STRENGTH_RANGE.max - STRENGTH_RANGE.min), 0, 1);
}

function drawStrengthBar(ctx, rect, value, interactive = false) {
    const centerX = rect.x + rect.w / 2;
    const ratio = strengthRatio(value);
    const valueX = rect.x + rect.w * ratio;
    roundRect(ctx, rect.x, rect.y, rect.w, rect.h, rect.h / 2);
    ctx.fillStyle = interactive ? THEME.surface1 : THEME.surface2;
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
            ctx.fillStyle = "#c4c4c4";
            ctx.fill();
        }
    } else {
        const fillX = valueX;
        const fillW = Math.max(0, centerX - valueX);
        if (fillW > 0) {
            roundRect(ctx, fillX, rect.y + 2, fillW, rect.h - 4, Math.max(3, (rect.h - 4) / 2));
            ctx.fillStyle = "#9e9e9e";
            ctx.fill();
        }
    }
}

function drawBalanceBar(ctx, rect, value, interactive = false) {
    const ratio = clamp(value, 0, 1);
    roundRect(ctx, rect.x, rect.y, rect.w, rect.h, rect.h / 2);
    ctx.fillStyle = interactive ? THEME.surface1 : THEME.surface2;
    ctx.fill();
    ctx.strokeStyle = THEME.border;
    ctx.lineWidth = 1;
    ctx.stroke();
    if (ratio > 0) {
        roundRect(ctx, rect.x + 2, rect.y + 2, Math.max(0, (rect.w - 4) * ratio), rect.h - 4, Math.max(3, (rect.h - 4) / 2));
        ctx.fillStyle = "#b9b9b9";
        ctx.fill();
    }
}

function drawGrip(ctx, rect) {
    ctx.fillStyle = THEME.textMuted;
    for (let row = 0; row < 3; row++) {
        for (let col = 0; col < 2; col++) {
            const x = rect.x + 4 + col * 6;
            const y = rect.y + 4 + row * 5;
            ctx.beginPath();
            ctx.arc(x, y, 1.2, 0, Math.PI * 2);
            ctx.fill();
        }
    }
}

function drawCardField(ctx, rect, label, value) {
    roundRect(ctx, rect.x, rect.y, rect.w, rect.h, 6);
    ctx.fillStyle = THEME.surface2;
    ctx.fill();
    ctx.strokeStyle = THEME.border;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = THEME.textMuted;
    ctx.font = "500 10px sans-serif";
    ctx.fillText(label, rect.x + 10, rect.y + rect.h * 0.42);
    ctx.fillStyle = THEME.text;
    ctx.font = "600 11px monospace";
    ctx.fillText(value, rect.x + 10, rect.y + rect.h * 0.73);
    ctx.fillStyle = THEME.textMuted;
    ctx.textAlign = "right";
    ctx.fillText("v", rect.x + rect.w - 10, rect.y + rect.h * 0.73);
    ctx.textAlign = "left";
}

async function getLoraList() {
    if (_loraListCache) return _loraListCache;
    if (_loraListPromise) return _loraListPromise;

    _loraListPromise = (async () => {
        try {
            const resp2 = await api.fetchApi("/object_info/FluxLoraLoader");
            const data2 = await resp2.json();
            const loaderInfo = data2?.FluxLoraLoader;
            if (loaderInfo?.input?.required?.lora_name) {
                _loraListCache = loaderInfo.input.required.lora_name[0];
                return _loraListCache;
            }
        } catch (e) {
            // fall through
        }

        try {
            const resp = await api.fetchApi("/object_info");
            const allNodes = await resp.json();
            for (const info of Object.values(allNodes)) {
                const req = info?.input?.required || {};
                for (const [key, value] of Object.entries(req)) {
                    if (key.includes("lora") && Array.isArray(value) && Array.isArray(value[0]) && value[0].length > 1) {
                        _loraListCache = value[0];
                        return _loraListCache;
                    }
                }
            }
        } catch (e) {
            console.warn("[FluxLoraMulti] Failed to fetch LoRA list:", e);
        }

        _loraListCache = ["None"];
        return _loraListCache;
    })();

    return _loraListPromise;
}

async function getAnatomyProfiles() {
    if (_anatomyProfilesCache) return _anatomyProfilesCache;
    if (_anatomyProfilesPromise) return _anatomyProfilesPromise;

    _anatomyProfilesPromise = (async () => {
        try {
            const resp = await api.fetchApi("/object_info/FluxLoraLoader");
            const data = await resp.json();
            const values = data?.FluxLoraLoader?.input?.optional?.anatomy_profile?.[0];
            if (Array.isArray(values) && values.length) {
                _anatomyProfilesCache = values;
                return _anatomyProfilesCache;
            }
        } catch (e) {
            console.warn("[FluxLoraMulti] Failed to fetch anatomy profiles:", e);
        }

        _anatomyProfilesCache = [...DEFAULT_ANATOMY_PROFILES];
        return _anatomyProfilesCache;
    })();

    return _anatomyProfilesPromise;
}

app.registerExtension({
    name: "Comfy.FluxLoraMulti",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FluxLoraMulti") return;

        const _onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = async function () {
            const result = _onNodeCreated?.apply(this, arguments);
            const node = this;
            const W = (name) => node.widgets?.find((widget) => widget.name === name);

            node._slots = parseSlotData(W("slot_data")?.value);
            const [loraList, anatomyProfiles] = await Promise.all([getLoraList(), getAnatomyProfiles()]);
            node._loraList = loraList;
            node._anatomyProfiles = anatomyProfiles;
            node._multiState = {
                viewMode: "compact",
                bounds: {
                    header: {},
                    toolbar: {},
                    cards: [],
                    hint: null,
                    empty: null,
                },
                drag: null,
                slider: null,
                dragSlotIndex: null,
            };

            function refreshNodeSize() {
                const currentWidth = Math.max(node.size?.[0] ?? 0, MIN_WIDTH);
                const [, computedHeight] = node.computeSize();
                node.setSize([currentWidth, computedHeight]);
            }

            function ensureProperties() {
                if (!node.properties) node.properties = {};
                return node.properties;
            }

            function persistSlotData(raw) {
                ensureProperties().slot_data_json = raw;
            }

            function readSlotDataWithFallback(config) {
                const widgetRaw = W("slot_data")?.value;
                if (typeof widgetRaw === "string" && widgetRaw.trim() && widgetRaw.trim() !== "[]") {
                    return widgetRaw;
                }
                const configRaw = config?.properties?.slot_data_json;
                if (typeof configRaw === "string" && configRaw.trim()) {
                    return configRaw;
                }
                const nodeRaw = node.properties?.slot_data_json;
                if (typeof nodeRaw === "string" && nodeRaw.trim()) {
                    return nodeRaw;
                }
                return "[]";
            }

            function syncSlotData() {
                const raw = serializeSlots(node._slots);
                persistSlotData(raw);
                updateWidgetValue(W("slot_data"), raw);
            }

            function markDirty() {
                refreshNodeSize();
                node.setDirtyCanvas(true, true);
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

            function toggleAutoConvert() {
                const widget = W("auto_convert");
                if (!widget) return;
                updateWidgetValue(widget, !widget.value);
                node.setDirtyCanvas(true, true);
            }

            function setViewMode(mode) {
                node._multiState.viewMode = mode;
                markDirty();
            }

            function addSlot(initial = {}) {
                setSlots([...node._slots, makeDefaultSlot(initial)]);
            }

            function duplicateSlot(index) {
                if (index < 0 || index >= node._slots.length) return;
                const slot = normalizeSlot(node._slots[index]);
                setSlots([
                    ...node._slots.slice(0, index + 1),
                    { ...slot },
                    ...node._slots.slice(index + 1),
                ]);
            }

            function removeSlot(index) {
                if (index < 0 || index >= node._slots.length) return;
                setSlots(node._slots.filter((_, slotIndex) => slotIndex !== index));
            }

            function moveSlotTo(fromIndex, toIndex) {
                if (fromIndex === toIndex) return;
                if (fromIndex < 0 || fromIndex >= node._slots.length) return;
                if (toIndex < 0 || toIndex >= node._slots.length) return;
                const next = [...node._slots];
                const [slot] = next.splice(fromIndex, 1);
                next.splice(toIndex, 0, slot);
                node._slots = next;
                syncSlotData();
                markDirty();
            }

            function resetSlots() {
                setSlots(node._slots.map(() => makeDefaultSlot()));
            }

            function toggleAll(enabled) {
                setSlots(node._slots.map((slot) => ({ ...slot, enabled })));
            }

            function toggleCollapsed(index) {
                if (node._multiState.viewMode !== "compact") return;
                const current = node._slots[index];
                if (!current) return;
                updateSlot(index, { collapsed: !current.collapsed });
            }

            function openSlotMenu(index, event) {
                const slot = node._slots[index];
                if (!slot) return;
                openMenu(event, [
                    {
                        content: slot.enabled ? "Disable slot" : "Enable slot",
                        callback: () => updateSlot(index, { enabled: !slot.enabled }),
                    },
                    {
                        content: node._multiState.viewMode === "compact"
                            ? (slot.collapsed ? "Expand card" : "Collapse card")
                            : "Expanded view active",
                        callback: () => {
                            if (node._multiState.viewMode === "compact") toggleCollapsed(index);
                        },
                    },
                    {
                        content: "Duplicate",
                        callback: () => duplicateSlot(index),
                    },
                    {
                        content: "Remove",
                        callback: () => removeSlot(index),
                    },
                ]);
            }

            function openSelector(index, field, values, event) {
                const slot = node._slots[index];
                if (!slot) return;
                openMenu(event, menuItemsFromValues(values, slot[field], (value) => {
                    updateSlot(index, { [field]: value });
                }));
            }

            function openLoraSelector(index, event) {
                const values = ["None", ...(node._loraList || [])];
                openSelector(index, "lora", values, event);
            }

            function openUseCaseSelector(index, event) {
                openSelector(index, "use_case", USE_CASES, event);
            }

            function openEditModeSelector(index, event) {
                openSelector(index, "edit_mode", EDIT_MODES, event);
            }

            function openAnatomyProfileSelector(index, event) {
                const values = node._anatomyProfiles?.length ? node._anatomyProfiles : DEFAULT_ANATOMY_PROFILES;
                openSelector(index, "anatomy_profile", values, event);
            }

            function openAnatomyCustomEditor(index) {
                const slot = node._slots[index];
                if (!slot) return;
                const seed = slot.anatomy_custom_json || "{}";
                const next = window.prompt("Custom anatomy profile JSON", seed);
                if (next === null) return;
                updateSlot(index, { anatomy_custom_json: String(next) });
            }

            function computeCardHeight(slot) {
                if (node._multiState.viewMode === "expanded") return CARD_EXPANDED_H;
                return slot.collapsed ? CARD_SUMMARY_H : CARD_EXPANDED_H;
            }

            function computeWidgetHeight() {
                let height = PAD + HEADER_H + ROW_GAP + TOOLBAR_H + ROW_GAP;
                if (!node._slots.length) {
                    height += EMPTY_H;
                } else {
                    height += node._slots.reduce((sum, slot, index) => (
                        sum + computeCardHeight(slot) + (index ? CARD_GAP : 0)
                    ), 0);
                }
                height += ROW_GAP + HINT_H + PAD;
                return height;
            }

            function drawHeader(ctx, x, y, w) {
                roundRect(ctx, x, y, w, HEADER_H, RADIUS);
                ctx.fillStyle = THEME.surface0;
                ctx.fill();
                ctx.strokeStyle = THEME.border;
                ctx.lineWidth = 1;
                ctx.stroke();

                ctx.fillStyle = THEME.text;
                ctx.font = "700 12px sans-serif";
                ctx.fillText("FLUX LoRA Multi", x + 14, y + 17);

                const active = activeSlotCount(node._slots);
                const counterText = `Active: ${active} / ${node._slots.length}`;
                ctx.fillStyle = THEME.textSoft;
                ctx.font = "600 11px monospace";
                ctx.textAlign = "right";
                ctx.fillText(counterText, x + w - 14, y + 17);
                ctx.textAlign = "left";

                const chipY = y + 20;
                const chipH = 20;
                const autoRect = { x: x + 14, y: chipY, w: 116, h: chipH };
                const globalRect = { x: x + 138, y: chipY, w: 96, h: chipH };
                const viewCompactRect = { x: x + w - 142, y: chipY, w: 62, h: chipH };
                const viewExpandedRect = { x: x + w - 74, y: chipY, w: 62, h: chipH };

                node._multiState.bounds.header = {
                    autoConvert: autoRect,
                    viewCompact: viewCompactRect,
                    viewExpanded: viewExpandedRect,
                };

                roundRect(ctx, autoRect.x, autoRect.y, autoRect.w, autoRect.h, 6);
                ctx.fillStyle = THEME.surface1;
                ctx.fill();
                ctx.strokeStyle = THEME.border;
                ctx.stroke();
                ctx.fillStyle = THEME.textMuted;
                ctx.font = "500 10px sans-serif";
                ctx.textBaseline = "middle";
                ctx.fillText("Auto-convert", autoRect.x + 10, autoRect.y + autoRect.h / 2 + 0.5);
                ctx.textBaseline = "alphabetic";
                drawToggle(ctx, { x: autoRect.x + autoRect.w - 42, y: autoRect.y + 3, w: 30, h: 16 }, !!W("auto_convert")?.value);

                drawMutedPill(ctx, globalRect, "Slots", `${active}`);
                drawSegmentButton(ctx, viewCompactRect, "Compact", node._multiState.viewMode === "compact");
                drawSegmentButton(ctx, viewExpandedRect, "Expanded", node._multiState.viewMode === "expanded");
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

                let bx = x + 10;
                node._multiState.bounds.toolbar = {};
                for (const button of buttons) {
                    const rect = { x: bx, y: y + 4, w: button.w, h: 20 };
                    node._multiState.bounds.toolbar[button.key] = rect;
                    drawActionPill(ctx, rect, button.label, button.tone);
                    bx += button.w + 8;
                }
            }

            function drawSummaryCard(ctx, slot, card, badge, isDragging) {
                roundRect(ctx, card.x, card.y, card.w, card.h, RADIUS);
                ctx.fillStyle = THEME.surface1;
                ctx.fill();
                ctx.strokeStyle = isDragging ? THEME.accent : THEME.border;
                ctx.lineWidth = isDragging ? 1.5 : 1;
                ctx.stroke();

                const headerCenterY = card.y + 18;
                const gripRect = { x: card.x + 10, y: headerCenterY - 8, w: 16, h: 16 };
                const toggleRect = { x: card.x + 32, y: headerCenterY - 9, w: 34, h: 18 };
                const chevronRect = { x: card.x + card.w - 56, y: headerCenterY - 9, w: 18, h: 18 };
                const menuRect = { x: card.x + card.w - 32, y: headerCenterY - 9, w: 18, h: 18 };

                drawGrip(ctx, gripRect);
                drawToggle(ctx, toggleRect, slot.enabled);

                ctx.fillStyle = THEME.text;
                ctx.font = "600 12px sans-serif";
                ctx.textBaseline = "middle";
                ctx.fillText(fitText(ctx, shortLoraName(slot.lora), card.w - 190), card.x + 76, headerCenterY + 0.5);
                ctx.textBaseline = "alphabetic";

                const badgeRect = { x: card.x + card.w - 124, y: headerCenterY - 9, w: 60, h: 18 };
                drawBadge(ctx, badgeRect, badge);

                drawIconButton(ctx, chevronRect, ">");
                drawIconButton(ctx, menuRect, "...");

                const barRect = { x: card.x + 16, y: card.y + 36, w: card.w * 0.42, h: 14 };
                drawStrengthBar(ctx, barRect, slot.strength, false);

                ctx.fillStyle = THEME.text;
                ctx.font = "600 10px monospace";
                ctx.fillText(slot.strength.toFixed(2), barRect.x + barRect.w + 10, barRect.y + 11);

                const metaY = card.y + 56;
                ctx.fillStyle = THEME.textSoft;
                ctx.font = "500 10px sans-serif";
                ctx.fillText(slot.use_case, card.x + 16, metaY);
                ctx.fillText(`Prot ${slot.balance.toFixed(2)}`, card.x + 74, metaY);
                const anatomyText = fitText(ctx, anatomySummary(slot, true), card.w - 230);
                ctx.fillText(anatomyText, card.x + 138, metaY);

                return {
                    card,
                    grip: gripRect,
                    toggle: toggleRect,
                    chevron: chevronRect,
                    menu: menuRect,
                    body: card,
                };
            }

            function drawExpandedCard(ctx, slot, card, badge, isDragging) {
                roundRect(ctx, card.x, card.y, card.w, card.h, RADIUS);
                ctx.fillStyle = THEME.surface1;
                ctx.fill();
                ctx.strokeStyle = isDragging ? THEME.accent : THEME.border;
                ctx.lineWidth = isDragging ? 1.5 : 1;
                ctx.stroke();

                const headerCenterY = card.y + 18;
                const gripRect = { x: card.x + 10, y: headerCenterY - 8, w: 16, h: 16 };
                const toggleRect = { x: card.x + 32, y: headerCenterY - 9, w: 34, h: 18 };
                const chevronRect = { x: card.x + card.w - 56, y: headerCenterY - 9, w: 18, h: 18 };
                const menuRect = { x: card.x + card.w - 32, y: headerCenterY - 9, w: 18, h: 18 };
                const badgeRect = { x: card.x + card.w - 124, y: headerCenterY - 9, w: 60, h: 18 };

                drawGrip(ctx, gripRect);
                drawToggle(ctx, toggleRect, slot.enabled);
                ctx.fillStyle = THEME.text;
                ctx.font = "600 12px sans-serif";
                ctx.textBaseline = "middle";
                ctx.fillText(fitText(ctx, shortLoraName(slot.lora), card.w - 200), card.x + 76, headerCenterY + 0.5);
                ctx.textBaseline = "alphabetic";
                drawBadge(ctx, badgeRect, badge);

                drawIconButton(ctx, chevronRect, node._multiState.viewMode === "compact" ? "v" : "-");
                drawIconButton(ctx, menuRect, "...");

                const innerX = card.x + 14;
                const innerW = card.w - 28;
                const fieldY1 = card.y + 34;
                const loraRect = { x: innerX, y: fieldY1, w: innerW, h: 26 };
                drawCardField(ctx, loraRect, "LoRA", fitText(ctx, slot.lora, innerW - 34));

                const halfGap = 8;
                const halfW = (innerW - halfGap) / 2;
                const fieldY2 = card.y + 66;
                const useCaseRect = { x: innerX, y: fieldY2, w: halfW, h: 26 };
                const modeRect = { x: innerX + halfW + halfGap, y: fieldY2, w: halfW, h: 26 };
                drawCardField(ctx, useCaseRect, "Use case", slot.use_case);
                drawCardField(ctx, modeRect, "Edit mode", badge.label);

                const anatomyRect = { x: innerX, y: card.y + 98, w: innerW, h: 26 };
                drawCardField(ctx, anatomyRect, "Anatomy profile", fitText(ctx, slot.anatomy_profile || "None", innerW - 34));

                const balanceLabelY = card.y + 134;
                ctx.fillStyle = THEME.textSoft;
                ctx.font = "500 10px sans-serif";
                ctx.fillText("Protection", innerX, balanceLabelY);
                ctx.textAlign = "right";
                ctx.fillText(slot.balance.toFixed(2), innerX + innerW, balanceLabelY);
                ctx.textAlign = "left";
                const balanceRect = { x: innerX, y: balanceLabelY + 4, w: innerW, h: 14 };
                drawBalanceBar(ctx, balanceRect, slot.balance, true);

                const strengthLabelY = card.y + 158;
                ctx.fillStyle = THEME.textSoft;
                ctx.font = "500 10px sans-serif";
                ctx.fillText("Strength", innerX, strengthLabelY);
                ctx.textAlign = "right";
                ctx.fillText(slot.strength.toFixed(2), innerX + innerW, strengthLabelY);
                ctx.textAlign = "left";
                const strengthRect = { x: innerX, y: strengthLabelY + 4, w: innerW, h: 14 };
                drawStrengthBar(ctx, strengthRect, slot.strength, true);

                const anatomyStrengthLabelY = card.y + 182;
                ctx.fillStyle = THEME.textSoft;
                ctx.font = "500 10px sans-serif";
                ctx.fillText("Anatomy strength", innerX, anatomyStrengthLabelY);
                ctx.textAlign = "right";
                ctx.fillText(slot.anatomy_strength.toFixed(2), innerX + innerW, anatomyStrengthLabelY);
                ctx.textAlign = "left";
                const anatomyStrengthRect = { x: innerX, y: anatomyStrengthLabelY + 4, w: innerW, h: 14 };
                drawBalanceBar(ctx, anatomyStrengthRect, slot.anatomy_strength, true);

                const strictRect = { x: innerX, y: card.y + 202, w: halfW, h: 26 };
                roundRect(ctx, strictRect.x, strictRect.y, strictRect.w, strictRect.h, 6);
                ctx.fillStyle = THEME.surface2;
                ctx.fill();
                ctx.strokeStyle = THEME.border;
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.fillStyle = THEME.textMuted;
                ctx.font = "500 10px sans-serif";
                ctx.fillText("Strict zero", strictRect.x + 10, strictRect.y + strictRect.h * 0.42);
                ctx.fillStyle = THEME.text;
                ctx.font = "600 11px monospace";
                ctx.fillText(slot.anatomy_strict_zero ? "Enabled" : "Disabled", strictRect.x + 10, strictRect.y + strictRect.h * 0.73);
                ctx.textBaseline = "alphabetic";
                drawToggle(
                    ctx,
                    { x: strictRect.x + strictRect.w - 40, y: strictRect.y + 5, w: 30, h: 16 },
                    !!slot.anatomy_strict_zero
                );

                const customRect = { x: innerX + halfW + halfGap, y: card.y + 202, w: halfW, h: 26 };
                drawCardField(
                    ctx,
                    customRect,
                    "Custom JSON",
                    slot.anatomy_profile === "Custom" ? "Edit..." : "Only for Custom"
                );

                const actionY = card.y + card.h - 24;
                const actionRects = {
                    duplicate: { x: innerX, y: actionY, w: 72, h: 18 },
                    toggle: { x: innerX + 80, y: actionY, w: 72, h: 18 },
                    remove: { x: innerX + 160, y: actionY, w: 64, h: 18 },
                };
                drawActionPill(ctx, actionRects.duplicate, "Duplicate");
                drawActionPill(ctx, actionRects.toggle, slot.enabled ? "Disable" : "Enable", slot.enabled ? "neutral" : "success");
                drawActionPill(ctx, actionRects.remove, "Remove", "danger");

                return {
                    card,
                    header: { x: card.x, y: card.y, w: card.w, h: 28 },
                    grip: gripRect,
                    toggle: toggleRect,
                    chevron: chevronRect,
                    menu: menuRect,
                    lora: loraRect,
                    useCase: useCaseRect,
                    editMode: modeRect,
                    anatomyProfile: anatomyRect,
                    balance: balanceRect,
                    strength: strengthRect,
                    anatomyStrength: anatomyStrengthRect,
                    anatomyStrict: strictRect,
                    anatomyCustom: customRect,
                    actions: actionRects,
                };
            }

            function drawCards(ctx, x, y, w) {
                node._multiState.bounds.cards = [];
                if (!node._slots.length) {
                    const emptyRect = { x, y, w, h: EMPTY_H };
                    node._multiState.bounds.empty = emptyRect;
                    roundRect(ctx, emptyRect.x, emptyRect.y, emptyRect.w, emptyRect.h, RADIUS);
                    ctx.fillStyle = THEME.surface0;
                    ctx.fill();
                    ctx.strokeStyle = THEME.border;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                    ctx.fillStyle = THEME.text;
                    ctx.font = "600 12px sans-serif";
                    ctx.fillText("No LoRA slots yet", emptyRect.x + 16, emptyRect.y + 28);
                    ctx.fillStyle = THEME.textMuted;
                    ctx.font = "500 11px sans-serif";
                    ctx.fillText("Use + Add LoRA to create the first slot.", emptyRect.x + 16, emptyRect.y + 48);
                    return EMPTY_H;
                }

                let cursorY = y;
                for (let i = 0; i < node._slots.length; i++) {
                    const slot = node._slots[i];
                    const badge = badgeForMode(slot.edit_mode);
                    const cardH = computeCardHeight(slot);
                    const card = { x, y: cursorY, w, h: cardH };
                    const isExpanded = node._multiState.viewMode === "expanded" || !slot.collapsed;
                    const isDragging = node._multiState.dragSlotIndex === i;
                    const bounds = isExpanded
                        ? drawExpandedCard(ctx, slot, card, badge, isDragging)
                        : drawSummaryCard(ctx, slot, card, badge, isDragging);
                    node._multiState.bounds.cards.push(bounds);
                    cursorY += cardH + CARD_GAP;
                }
                return cursorY - y - CARD_GAP;
            }

            function drawHint(ctx, x, y, w) {
                const hintRect = { x, y, w, h: HINT_H };
                node._multiState.bounds.hint = hintRect;
                roundRect(ctx, hintRect.x, hintRect.y, hintRect.w, hintRect.h, RADIUS);
                ctx.fillStyle = THEME.surface0;
                ctx.fill();
                ctx.strokeStyle = THEME.border;
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.fillStyle = THEME.textMuted;
                ctx.font = "500 10px sans-serif";
                ctx.fillText("Drag grip to reorder • click card to expand • menu for duplicate/remove", hintRect.x + 12, hintRect.y + 15);
            }

            function drawPanel(ctx, nodeRef, width, y) {
                const panelX = PAD;
                const panelW = width - PAD * 2;
                let cursorY = y + PAD;

                roundRect(ctx, panelX, cursorY, panelW, computeWidgetHeight() - PAD * 2, RADIUS + 2);
                ctx.fillStyle = THEME.canvas;
                ctx.fill();

                drawHeader(ctx, panelX, cursorY, panelW);
                cursorY += HEADER_H + ROW_GAP;
                drawToolbar(ctx, panelX, cursorY, panelW);
                cursorY += TOOLBAR_H + ROW_GAP;
                const cardsHeight = drawCards(ctx, panelX, cursorY, panelW);
                cursorY += cardsHeight + ROW_GAP;
                drawHint(ctx, panelX, cursorY, panelW);
            }

            function startSlider(index, field, rect, range, x) {
                node._multiState.slider = { index, field, rect, range };
                updateSlider(x);
            }

            function updateSlider(x) {
                const slider = node._multiState.slider;
                if (!slider) return;
                const ratio = clamp((x - slider.rect.x) / slider.rect.w, 0, 1);
                const rawValue = slider.range.min + ratio * (slider.range.max - slider.range.min);
                const nextValue = snapToStep(rawValue, slider.range.step, slider.range.min, slider.range.max, slider.range.precision);
                updateSlot(slider.index, { [slider.field]: nextValue });
            }

            function endSlider() {
                if (!node._multiState.slider) return false;
                node._multiState.slider = null;
                return true;
            }

            function findReorderTarget(y) {
                const cards = node._multiState.bounds.cards;
                for (let i = 0; i < cards.length; i++) {
                    const card = cards[i].card;
                    if (y < card.y + card.h / 2) return i;
                }
                return cards.length - 1;
            }

            function startDrag(index, y) {
                node._multiState.drag = {
                    index,
                    startY: y,
                    active: false,
                };
                node._multiState.dragSlotIndex = index;
            }

            function updateDrag(y) {
                const drag = node._multiState.drag;
                if (!drag) return;
                if (!drag.active && Math.abs(y - drag.startY) > 5) {
                    drag.active = true;
                }
                if (!drag.active) return;
                const target = findReorderTarget(y);
                if (target !== drag.index) {
                    moveSlotTo(drag.index, target);
                    drag.index = target;
                    node._multiState.dragSlotIndex = target;
                }
            }

            function endDrag() {
                if (!node._multiState.drag) return false;
                node._multiState.drag = null;
                node._multiState.dragSlotIndex = null;
                node.setDirtyCanvas(true, true);
                return true;
            }

            function handleToolbarClick(key) {
                if (key === "add") addSlot();
                else if (key === "enableAll") toggleAll(true);
                else if (key === "disableAll") toggleAll(false);
                else if (key === "reset") resetSlots();
            }

            function widgetMouse(event, pos) {
                const [mx, my] = pos;
                const headerBounds = node._multiState.bounds.header;
                const toolbarBounds = node._multiState.bounds.toolbar;

                if (event.type === "pointermove") {
                    if (node._multiState.slider) {
                        updateSlider(mx);
                        return true;
                    }
                    if (node._multiState.drag) {
                        updateDrag(my);
                        return true;
                    }
                    return false;
                }

                if (event.type === "pointerup" || event.type === "pointercancel") {
                    const sliderHandled = endSlider();
                    const dragHandled = endDrag();
                    return sliderHandled || dragHandled;
                }

                if (event.type !== "pointerdown") return false;

                if (pointInRect(mx, my, headerBounds.autoConvert)) {
                    toggleAutoConvert();
                    return true;
                }
                if (pointInRect(mx, my, headerBounds.viewCompact)) {
                    setViewMode("compact");
                    return true;
                }
                if (pointInRect(mx, my, headerBounds.viewExpanded)) {
                    setViewMode("expanded");
                    return true;
                }

                for (const [key, rect] of Object.entries(toolbarBounds || {})) {
                    if (pointInRect(mx, my, rect)) {
                        handleToolbarClick(key);
                        return true;
                    }
                }

                const cards = node._multiState.bounds.cards || [];
                for (let index = 0; index < cards.length; index++) {
                    const bounds = cards[index];
                    const slot = node._slots[index];
                    const isExpanded = node._multiState.viewMode === "expanded" || !slot.collapsed;

                    if (pointInRect(mx, my, bounds.grip)) {
                        startDrag(index, my);
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.toggle)) {
                        updateSlot(index, { enabled: !slot.enabled });
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.menu)) {
                        openSlotMenu(index, event);
                        return true;
                    }
                    if (node._multiState.viewMode === "compact" && pointInRect(mx, my, bounds.chevron)) {
                        toggleCollapsed(index);
                        return true;
                    }

                    if (!isExpanded) {
                        if (pointInRect(mx, my, bounds.card)) {
                            updateSlot(index, { collapsed: false });
                            return true;
                        }
                        continue;
                    }

                    if (pointInRect(mx, my, bounds.lora)) {
                        openLoraSelector(index, event);
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.useCase)) {
                        openUseCaseSelector(index, event);
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.editMode)) {
                        openEditModeSelector(index, event);
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.anatomyProfile)) {
                        openAnatomyProfileSelector(index, event);
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.balance)) {
                        startSlider(index, "balance", bounds.balance, BALANCE_RANGE, mx);
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.strength)) {
                        startSlider(index, "strength", bounds.strength, STRENGTH_RANGE, mx);
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.anatomyStrength)) {
                        startSlider(index, "anatomy_strength", bounds.anatomyStrength, ANATOMY_STRENGTH_RANGE, mx);
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.anatomyStrict)) {
                        updateSlot(index, { anatomy_strict_zero: !slot.anatomy_strict_zero });
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.anatomyCustom)) {
                        if (slot.anatomy_profile === "Custom") {
                            openAnatomyCustomEditor(index);
                        }
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.actions?.duplicate)) {
                        duplicateSlot(index);
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.actions?.toggle)) {
                        updateSlot(index, { enabled: !slot.enabled });
                        return true;
                    }
                    if (pointInRect(mx, my, bounds.actions?.remove)) {
                        removeSlot(index);
                        return true;
                    }
                    if (node._multiState.viewMode === "compact" && pointInRect(mx, my, bounds.header)) {
                        toggleCollapsed(index);
                        return true;
                    }
                }

                return false;
            }

            const uiWidget = {
                name: "flux_lora_multi_cards",
                type: "flux_lora_multi_cards",
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
                // slot_data must keep its original type so ComfyUI serializes
                // it into widgets_values and restores it on workflow reload.
                hideWidget(node, W("slot_data"), { preserveType: true });
                hideWidget(node, W("auto_convert"));
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
                    if (!node._loraList || !node._anatomyProfiles) {
                        const [loraList, anatomyProfiles] = await Promise.all([getLoraList(), getAnatomyProfiles()]);
                        node._loraList = loraList;
                        node._anatomyProfiles = anatomyProfiles;
                    }
                    // Fallback path: when another ComfyUI install drops hidden widget
                    // values, we recover from node.properties.
                    const restoredRaw = readSlotDataWithFallback(config);
                    updateWidgetValue(W("slot_data"), restoredRaw);
                    persistSlotData(restoredRaw);
                    node._slots = parseSlotData(restoredRaw);
                    markDirty();
                }, 50);
            };

            const origSerialize = node.onSerialize?.bind(node);
            node.onSerialize = function (output) {
                origSerialize?.(output);
                const raw = serializeSlots(node._slots || []);
                output.properties = output.properties || {};
                output.properties.slot_data_json = raw;
                persistSlotData(raw);
            };

            if (node.size[0] < MIN_WIDTH) node.size[0] = MIN_WIDTH;
            markDirty();

            return result;
        };
    },
});
