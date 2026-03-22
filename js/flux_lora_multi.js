// FLUX LoRA Multi — Dynamic Slot Widget (rgthree-style)
//
// Provides "+ Add LoRA" button to dynamically add/remove LoRA slots.
// Each slot: enabled toggle, LoRA dropdown, strength, edit_mode, balance.
// All slot data serialized to hidden `slot_data` widget as JSON array.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Edit-mode preset names (must match Python PRESET_NAMES)
const EDIT_MODES = ["None", "Preserve Face", "Preserve Body", "Style Only", "Edit Subject", "Boost Prompt", "Auto"];

// Cached LoRA file list
let _loraListCache = null;
let _loraListPromise = null;

async function getLoraList() {
    if (_loraListCache) return _loraListCache;
    if (_loraListPromise) return _loraListPromise;

    _loraListPromise = (async () => {
        try {
            // Get LoRA list from FluxLoraLoader which has a lora_name dropdown
            const resp2 = await api.fetchApi("/object_info/FluxLoraLoader");
            const data2 = await resp2.json();
            const loaderInfo = data2?.FluxLoraLoader;
            if (loaderInfo?.input?.required?.lora_name) {
                _loraListCache = loaderInfo.input.required.lora_name[0];
                return _loraListCache;
            }
        } catch (e) {
            // Fallback: try to get from any loaded node
        }

        // Broader fallback: search all node defs
        try {
            const resp = await api.fetchApi("/object_info");
            const allNodes = await resp.json();
            for (const [name, info] of Object.entries(allNodes)) {
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
    widget.type        = "hidden_flux_multi";
    widget.computeSize = () => [0, -4];
}

function makeDivider() {
    return {
        name:  "divider",
        type:  "divider",
        options: { serialize: false },
        computeSize() { return [0, 4]; },
        draw(ctx, node, width, y) {
            ctx.strokeStyle = "#2a2a3a";
            ctx.lineWidth   = 1;
            ctx.beginPath();
            ctx.moveTo(15, y + 2);
            ctx.lineTo(width - 15, y + 2);
            ctx.stroke();
        },
    };
}

app.registerExtension({
    name: "Comfy.FluxLoraMulti",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FluxLoraMulti") return;

        const _onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = async function () {
            const result = _onNodeCreated?.apply(this, arguments);
            const node = this;

            // ── State ─────────────────────────────────────────────────────
            node._slots = [];         // Array of slot data objects
            node._slotWidgets = [];   // Array of widget group arrays (per slot)
            node._loraList = null;

            // ── Hide slot_data widget ─────────────────────────────────────
            setTimeout(() => {
                const w = node.widgets?.find(w => w.name === "slot_data");
                if (w) hideWidget(node, w);
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true, true);
            }, 0);

            // ── Fetch LoRA list ───────────────────────────────────────────
            node._loraList = await getLoraList();

            // ── Sync slots to hidden widget ──────────────────────────────
            function syncSlotData() {
                const w = node.widgets?.find(w => w.name === "slot_data");
                if (!w) return;
                const data = node._slots.map(s => ({
                    enabled:   s.enabled,
                    lora:      s.lora,
                    strength:  s.strength,
                    edit_mode: s.edit_mode,
                    balance:   s.balance,
                }));
                w.value = JSON.stringify(data);
            }

            // ── Add slot ─────────────────────────────────────────────────
            function addSlot(initial) {
                const idx = node._slots.length;
                const data = {
                    enabled:   initial?.enabled   ?? true,
                    lora:      initial?.lora      ?? "None",
                    strength:  initial?.strength  ?? 1.0,
                    edit_mode: initial?.edit_mode ?? "None",
                    balance:   initial?.balance   ?? 0.5,
                };
                node._slots.push(data);

                const loraValues = ["None", ...(node._loraList || [])];

                // Find position: before the "+ Add LoRA" button
                const addBtnIdx = node.widgets.findIndex(w => w.name === "_add_lora_btn");
                const insertAt = addBtnIdx >= 0 ? addBtnIdx : node.widgets.length;

                const widgets = [];

                // Divider
                const div = makeDivider();
                node.widgets.splice(insertAt, 0, div);
                widgets.push(div);

                // Header with slot number and remove button
                const header = {
                    name:    `_slot_header_${idx}`,
                    type:    "header",
                    options: { serialize: false },
                    computeSize() { return [0, 24]; },
                    draw(ctx, node, width, y) {
                        // Slot label
                        ctx.fillStyle = "#6a6a8a";
                        ctx.font      = "bold 10px monospace";
                        ctx.textAlign = "left";
                        const slotNum = node._slots.indexOf(data) + 1;
                        ctx.fillText(`LoRA ${slotNum}`, 15, y + 16);

                        // Remove button [✕]
                        const bW = 22, bH = 18;
                        const bX = width - bW - 12;
                        const bY = y + 3;
                        ctx.fillStyle   = "#2a1a1a";
                        ctx.strokeStyle = "#5a2a2a";
                        ctx.lineWidth   = 1;
                        ctx.beginPath();
                        ctx.roundRect(bX, bY, bW, bH, 3);
                        ctx.fill();
                        ctx.stroke();
                        ctx.fillStyle = "#cc4444";
                        ctx.font      = "bold 11px monospace";
                        ctx.textAlign = "center";
                        ctx.fillText("✕", bX + bW / 2, bY + 14);
                        ctx.textAlign = "left";

                        // Store button bounds for click detection
                        header._removeBounds = { x: bX, y: bY, w: bW, h: bH };
                    },
                    mouse(event, pos) {
                        if (event.type !== "pointerdown") return false;
                        const b = header._removeBounds;
                        if (!b) return false;
                        const [mx, my] = pos;
                        if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
                            removeSlot(node._slots.indexOf(data));
                            return true;
                        }
                        return false;
                    },
                };
                node.widgets.splice(insertAt + 1, 0, header);
                widgets.push(header);

                // Enabled toggle
                const enabledW = node.addWidget("toggle", `_enabled_${idx}`, data.enabled, (v) => {
                    data.enabled = v;
                    syncSlotData();
                }, { on: "On", off: "Off", serialize: false });
                // Move to correct position
                node.widgets.splice(node.widgets.indexOf(enabledW), 1);
                node.widgets.splice(insertAt + 2, 0, enabledW);
                widgets.push(enabledW);

                // LoRA combo
                const loraW = node.addWidget("combo", `_lora_${idx}`, data.lora, (v) => {
                    data.lora = v;
                    syncSlotData();
                }, { values: loraValues, serialize: false });
                node.widgets.splice(node.widgets.indexOf(loraW), 1);
                node.widgets.splice(insertAt + 3, 0, loraW);
                widgets.push(loraW);

                // Strength
                const strengthW = node.addWidget("number", `_strength_${idx}`, data.strength, (v) => {
                    data.strength = v;
                    syncSlotData();
                }, { min: -5.0, max: 5.0, step: 0.05, precision: 2, serialize: false });
                node.widgets.splice(node.widgets.indexOf(strengthW), 1);
                node.widgets.splice(insertAt + 4, 0, strengthW);
                widgets.push(strengthW);

                // Edit mode combo
                const editW = node.addWidget("combo", `_edit_mode_${idx}`, data.edit_mode, (v) => {
                    data.edit_mode = v;
                    syncSlotData();
                }, { values: EDIT_MODES, serialize: false });
                node.widgets.splice(node.widgets.indexOf(editW), 1);
                node.widgets.splice(insertAt + 5, 0, editW);
                widgets.push(editW);

                // Balance
                const balanceW = node.addWidget("number", `_balance_${idx}`, data.balance, (v) => {
                    data.balance = v;
                    syncSlotData();
                }, { min: 0.0, max: 1.0, step: 0.05, precision: 2, serialize: false });
                node.widgets.splice(node.widgets.indexOf(balanceW), 1);
                node.widgets.splice(insertAt + 6, 0, balanceW);
                widgets.push(balanceW);

                node._slotWidgets.push(widgets);
                syncSlotData();
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true, true);
            }

            // ── Remove slot ──────────────────────────────────────────────
            function removeSlot(slotIdx) {
                if (slotIdx < 0 || slotIdx >= node._slots.length) return;

                // Remove widgets for this slot
                const widgetGroup = node._slotWidgets[slotIdx];
                if (widgetGroup) {
                    for (const w of widgetGroup) {
                        const idx = node.widgets.indexOf(w);
                        if (idx >= 0) node.widgets.splice(idx, 1);
                    }
                }

                node._slots.splice(slotIdx, 1);
                node._slotWidgets.splice(slotIdx, 1);

                syncSlotData();
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true, true);
            }

            // ── "+ Add LoRA" button ──────────────────────────────────────
            const addBtn = {
                name: "_add_lora_btn",
                type: "button",
                options: { serialize: false },
                computeSize() { return [0, 32]; },
                draw(ctx, node, width, y) {
                    const bW = 140, bH = 26;
                    const bX = (width - bW) / 2;
                    const bY = y + 3;
                    ctx.fillStyle   = "#1a2a1a";
                    ctx.strokeStyle = "#3a6a3a";
                    ctx.lineWidth   = 1;
                    ctx.beginPath();
                    ctx.roundRect(bX, bY, bW, bH, 5);
                    ctx.fill();
                    ctx.stroke();
                    ctx.fillStyle = "#5ecc5e";
                    ctx.font      = "bold 12px monospace";
                    ctx.textAlign = "center";
                    ctx.fillText("+ Add LoRA", width / 2, bY + 18);
                    ctx.textAlign = "left";
                    addBtn._bounds = { x: bX, y: bY, w: bW, h: bH };
                },
                mouse(event, pos) {
                    if (event.type !== "pointerdown") return false;
                    const b = addBtn._bounds;
                    if (!b) return false;
                    const [mx, my] = pos;
                    if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
                        addSlot(null);
                        return true;
                    }
                    return false;
                },
            };
            node.widgets.push(addBtn);

            // ── Set minimum width ────────────────────────────────────────
            const MIN_WIDTH = 340;
            if (node.size[0] < MIN_WIDTH) {
                node.size[0] = MIN_WIDTH;
            }
            node.setSize(node.computeSize());

            // ── Restore state after workflow reload ──────────────────────
            const origConfigure = node.onConfigure?.bind(node);
            node.onConfigure = function (config) {
                origConfigure?.(config);
                setTimeout(async () => {
                    // Ensure LoRA list is loaded
                    if (!node._loraList) {
                        node._loraList = await getLoraList();
                    }

                    const w = node.widgets?.find(w => w.name === "slot_data");
                    if (!w || !w.value || w.value === "[]") return;

                    try {
                        const slots = JSON.parse(w.value);
                        if (!Array.isArray(slots)) return;

                        // Clear existing dynamic slots
                        while (node._slots.length > 0) {
                            removeSlot(0);
                        }

                        // Rebuild slots from saved data
                        for (const slotData of slots) {
                            addSlot(slotData);
                        }
                    } catch (e) {
                        console.warn("[FluxLoraMulti] Failed to restore slots:", e);
                    }
                }, 50);
            };

            return result;
        };
    },
});
