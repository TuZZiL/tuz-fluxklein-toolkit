// FLUX LoRA Multi — Dynamic Slot Widget (rgthree-style)
//
// Provides "+ Add LoRA" button to dynamically add/remove LoRA slots.
// Each slot: enabled toggle, LoRA dropdown, strength, use_case, edit_mode, balance.
// All slot data serialized to hidden `slot_data` widget as JSON array.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EDIT_MODES = ["None", "Preserve Face", "Preserve Body", "Style Only", "Edit Subject", "Boost Prompt", "Auto"];
const USE_CASES = ["Edit", "Generate"];

let _loraListCache = null;
let _loraListPromise = null;

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
                for (const [k, v] of Object.entries(req)) {
                    if (k.includes("lora") && Array.isArray(v) && Array.isArray(v[0]) && v[0].length > 1) {
                        _loraListCache = v[0];
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

function makeDivider() {
    return {
        name: "divider",
        type: "divider",
        options: { serialize: false },
        computeSize() { return [0, 4]; },
        draw(ctx, node, width, y) {
            ctx.strokeStyle = "#2a2a3a";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(15, y + 2);
            ctx.lineTo(width - 15, y + 2);
            ctx.stroke();
        },
    };
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

function summarizeSlot(data) {
    const state = data.enabled ? "On" : "Off";
    const loraName = data.lora && data.lora !== "None" ? data.lora.replace(/\.[^.]+$/, "") : "No LoRA";
    const mode = data.edit_mode === "None" ? "Standard" : data.edit_mode;
    return `${state} • ${loraName} • ${Number(data.strength ?? 0).toFixed(2)} • ${data.use_case} • ${mode}`;
}

function normalizeSlot(initial) {
    return {
        enabled: initial?.enabled ?? true,
        lora: initial?.lora ?? "None",
        strength: initial?.strength ?? 1.0,
        use_case: initial?.use_case ?? "Edit",
        edit_mode: initial?.edit_mode ?? "None",
        balance: initial?.balance ?? 0.5,
        collapsed: initial?.collapsed ?? false,
    };
}

function moveWidgetTo(node, widget, index) {
    const current = node.widgets.indexOf(widget);
    if (current >= 0) node.widgets.splice(current, 1);
    node.widgets.splice(index, 0, widget);
}

app.registerExtension({
    name: "Comfy.FluxLoraMulti",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FluxLoraMulti") return;

        const _onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = async function () {
            const result = _onNodeCreated?.apply(this, arguments);
            const node = this;
            const W = (name) => node.widgets?.find(widget => widget.name === name);

            node._slots = [];
            node._slotWidgets = [];
            hideWidget(node, W("slot_data"));
            node._loraList = await getLoraList();

            function syncSlotData() {
                const w = node.widgets?.find(widget => widget.name === "slot_data");
                if (!w) return;
                w.value = JSON.stringify(node._slots.map(slot => ({
                    enabled: slot.enabled,
                    lora: slot.lora,
                    strength: slot.strength,
                    use_case: slot.use_case,
                    edit_mode: slot.edit_mode,
                    balance: slot.balance,
                    collapsed: slot.collapsed,
                })));
            }

            function removeRenderedSlotWidgets() {
                for (const group of node._slotWidgets) {
                    for (const widget of group) {
                        const idx = node.widgets.indexOf(widget);
                        if (idx >= 0) node.widgets.splice(idx, 1);
                    }
                }
                node._slotWidgets = [];
            }

            function addSlot(initial, index = node._slots.length) {
                node._slots.splice(index, 0, normalizeSlot(initial));
                renderSlots();
            }

            function duplicateSlot(index) {
                if (index < 0 || index >= node._slots.length) return;
                const copy = { ...node._slots[index] };
                addSlot(copy, index + 1);
            }

            function moveSlot(index, delta) {
                const next = index + delta;
                if (index < 0 || index >= node._slots.length) return;
                if (next < 0 || next >= node._slots.length) return;
                const [slot] = node._slots.splice(index, 1);
                node._slots.splice(next, 0, slot);
                renderSlots();
            }

            function toggleCollapse(index) {
                if (index < 0 || index >= node._slots.length) return;
                node._slots[index].collapsed = !node._slots[index].collapsed;
                renderSlots();
            }

            function removeSlot(index) {
                if (index < 0 || index >= node._slots.length) return;
                node._slots.splice(index, 1);
                renderSlots();
            }

            function renderSlots() {
                removeRenderedSlotWidgets();

                for (let slotIdx = 0; slotIdx < node._slots.length; slotIdx++) {
                    const data = node._slots[slotIdx];
                    const loraValues = ["None", ...(node._loraList || [])];
                    const group = [];
                    let insertAt = node.widgets.findIndex(widget => widget.name === "_add_lora_btn");
                    if (insertAt < 0) insertAt = node.widgets.length;

                    const divider = makeDivider();
                    node.widgets.splice(insertAt, 0, divider);
                    group.push(divider);
                    insertAt += 1;

                    const header = {
                        name: `_slot_header_${slotIdx}`,
                        type: "header",
                        options: { serialize: false },
                        computeSize() { return [0, 24]; },
                        draw(ctx, nodeRef, width, y) {
                            ctx.fillStyle = "#6a6a8a";
                            ctx.font = "bold 10px monospace";
                            ctx.textAlign = "left";
                            ctx.fillText(`LoRA ${slotIdx + 1}`, 15, y + 16);

                            const buttons = [
                                { key: "collapse", label: data.collapsed ? ">" : "v", fill: "#1c2430", stroke: "#49617d", text: "#9ec0ff" },
                                { key: "up", label: "^", fill: "#1d2216", stroke: "#4d6a38", text: "#b8e28f" },
                                { key: "down", label: "v", fill: "#1d2216", stroke: "#4d6a38", text: "#b8e28f" },
                                { key: "duplicate", label: "+", fill: "#1d1d2b", stroke: "#565689", text: "#c9c9ff" },
                                { key: "remove", label: "x", fill: "#2a1a1a", stroke: "#5a2a2a", text: "#cc4444" },
                            ];

                            const bW = 18;
                            const bH = 18;
                            const gap = 4;
                            let bX = width - 12 - ((bW + gap) * buttons.length - gap);
                            const bY = y + 3;
                            header._controls = {};

                            for (const btn of buttons) {
                                ctx.fillStyle = btn.fill;
                                ctx.strokeStyle = btn.stroke;
                                ctx.lineWidth = 1;
                                ctx.beginPath();
                                ctx.roundRect(bX, bY, bW, bH, 3);
                                ctx.fill();
                                ctx.stroke();
                                ctx.fillStyle = btn.text;
                                ctx.font = "bold 10px monospace";
                                ctx.textAlign = "center";
                                ctx.fillText(btn.label, bX + bW / 2, bY + 13);
                                header._controls[btn.key] = { x: bX, y: 3, w: bW, h: bH };
                                bX += bW + gap;
                            }

                            const summaryX = 62;
                            ctx.font = "9px monospace";
                            ctx.fillStyle = "#8a8aa6";
                            ctx.textAlign = "left";
                            const summary = fitText(ctx, summarizeSlot(data), Math.max(30, width - summaryX - 120));
                            ctx.fillText(summary, summaryX, y + 16);
                        },
                        mouse(event, pos) {
                            if (event.type !== "pointerdown") return false;
                            const [mx, my] = pos;
                            for (const [key, bounds] of Object.entries(header._controls || {})) {
                                if (mx >= bounds.x && mx <= bounds.x + bounds.w && my >= bounds.y && my <= bounds.y + bounds.h) {
                                    if (key === "collapse") toggleCollapse(slotIdx);
                                    else if (key === "up") moveSlot(slotIdx, -1);
                                    else if (key === "down") moveSlot(slotIdx, 1);
                                    else if (key === "duplicate") duplicateSlot(slotIdx);
                                    else if (key === "remove") removeSlot(slotIdx);
                                    return true;
                                }
                            }
                            return false;
                        },
                    };
                    node.widgets.splice(insertAt, 0, header);
                    group.push(header);
                    insertAt += 1;

                    if (!data.collapsed) {
                        const enabledW = node.addWidget("toggle", "Enabled", data.enabled, (v) => {
                            data.enabled = v;
                            syncSlotData();
                            node.setDirtyCanvas(true, true);
                        }, {
                            on: "On",
                            off: "Off",
                            serialize: false,
                            tooltip: "Turns this LoRA slot on or off without removing its settings.",
                        });
                        moveWidgetTo(node, enabledW, insertAt++);
                        group.push(enabledW);

                        const loraW = node.addWidget("combo", "LoRA", data.lora, (v) => {
                            data.lora = v;
                            syncSlotData();
                            node.setDirtyCanvas(true, true);
                        }, {
                            values: loraValues,
                            serialize: false,
                            tooltip: "Which LoRA file to load in this slot.",
                        });
                        moveWidgetTo(node, loraW, insertAt++);
                        group.push(loraW);

                        const strengthW = node.addWidget("number", "Strength", data.strength, (v) => {
                            data.strength = v;
                            syncSlotData();
                            node.setDirtyCanvas(true, true);
                        }, {
                            min: -5.0,
                            max: 5.0,
                            step: 0.05,
                            precision: 2,
                            serialize: false,
                            tooltip: "Overall LoRA strength for this slot. Lower it first if the edit is too aggressive.",
                        });
                        moveWidgetTo(node, strengthW, insertAt++);
                        group.push(strengthW);

                        const useCaseW = node.addWidget("combo", "Use case", data.use_case, (v) => {
                            data.use_case = v;
                            syncSlotData();
                            node.setDirtyCanvas(true, true);
                        }, {
                            values: USE_CASES,
                            serialize: false,
                            tooltip: "Tells Auto whether this slot should protect a reference image more (Edit) or allow freer generation (Generate).",
                        });
                        moveWidgetTo(node, useCaseW, insertAt++);
                        group.push(useCaseW);

                        const editW = node.addWidget("combo", "Mode", data.edit_mode, (v) => {
                            data.edit_mode = v;
                            syncSlotData();
                            node.setDirtyCanvas(true, true);
                        }, {
                            values: EDIT_MODES,
                            serialize: false,
                            tooltip: "How protective this slot should be. Auto is the safest starting point; None means raw LoRA behavior.",
                        });
                        moveWidgetTo(node, editW, insertAt++);
                        group.push(editW);

                        const balanceW = node.addWidget("number", "Balance", data.balance, (v) => {
                            data.balance = v;
                            syncSlotData();
                            node.setDirtyCanvas(true, true);
                        }, {
                            min: 0.0,
                            max: 1.0,
                            step: 0.05,
                            precision: 2,
                            serialize: false,
                            tooltip: "How strongly to apply the chosen mode. Lower = safer / more preserving. Higher = closer to raw LoRA.",
                        });
                        moveWidgetTo(node, balanceW, insertAt++);
                        group.push(balanceW);
                    }

                    node._slotWidgets.push(group);
                }

                syncSlotData();
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true, true);
            }

            setTimeout(() => {
                hideWidget(node, W("slot_data"));
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true, true);
            }, 0);

            const addBtn = {
                name: "_add_lora_btn",
                type: "button",
                options: { serialize: false },
                computeSize() { return [0, 32]; },
                draw(ctx, nodeRef, width, y) {
                    const bW = 140;
                    const bH = 26;
                    const bX = (width - bW) / 2;
                    const bY = y + 3;
                    ctx.fillStyle = "#1a2a1a";
                    ctx.strokeStyle = "#3a6a3a";
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.roundRect(bX, bY, bW, bH, 5);
                    ctx.fill();
                    ctx.stroke();
                    ctx.fillStyle = "#5ecc5e";
                    ctx.font = "bold 12px monospace";
                    ctx.textAlign = "center";
                    ctx.fillText("+ Add LoRA", width / 2, bY + 18);
                    ctx.textAlign = "left";
                    addBtn._bounds = { x: bX, y: 3, w: bW, h: bH };
                },
                mouse(event, pos) {
                    if (event.type !== "pointerdown") return false;
                    const [mx, my] = pos;
                    const b = addBtn._bounds;
                    if (b && mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
                        addSlot({ use_case: "Edit" });
                        return true;
                    }
                    return false;
                },
            };
            node.widgets.push(addBtn);

            const MIN_WIDTH = 360;
            if (node.size[0] < MIN_WIDTH) node.size[0] = MIN_WIDTH;
            node.setSize(node.computeSize());

            const origConfigure = node.onConfigure?.bind(node);
            node.onConfigure = function (config) {
                origConfigure?.(config);
                setTimeout(async () => {
                    if (!node._loraList) node._loraList = await getLoraList();
                    const w = node.widgets?.find(widget => widget.name === "slot_data");
                    if (!w || !w.value || w.value === "[]") return;
                    try {
                        const slots = JSON.parse(w.value);
                        if (!Array.isArray(slots)) return;
                        node._slots = slots.map(slot => normalizeSlot(slot));
                        renderSlots();
                    } catch (e) {
                        console.warn("[FluxLoraMulti] Failed to restore slots:", e);
                    }
                }, 50);
            };

            return result;
        };
    },
});
