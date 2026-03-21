// FLUX LoRA Loader — Layer Strength Graph Widget
//
// Two sections side by side:
//   LEFT  — Double blocks (8 cols)
//     TOP half    (purple) = img stream strength  (img_attn + img_mlp)
//     BOTTOM half (teal)   = txt stream strength  (txt_attn + txt_mlp)
//
//   RIGHT — Single blocks (24 cols)
//     ONE bar per block    (green)
//
// Bar height maps 0.0 → 2.0. Global strength shown as a dashed reference line.
//
// Controls:
//   Drag bar up/down  → set strength
//   Click bar         → toggle between 0 and last non-zero value
//   Shift-drag        → move all visible blocks together (within section)
//   Buttons: Reset All | Mirror img→txt | Flatten (all → global)
//
// Serializes to hidden `layer_strengths` widget as JSON:
//   {
//     "db": { "0": { "img": 1.2, "txt": 0.8 }, ... },
//     "sb": { "0": 0.9, "1": 1.0, ... }
//   }

import { app } from "../../scripts/app.js";

const N_DOUBLE  = 8;
const N_SINGLE  = 24;
const STR_MAX   = 2.0;
const STR_MIN   = 0.0;

const PAD       = 10;
const GRAPH_H   = 170;
const BTN_ROW_H = 26;
const LABEL_H   = 18;
const WIDGET_H  = GRAPH_H + BTN_ROW_H + LABEL_H + PAD * 3;

// Fraction of graph width for each section
const DB_FRAC = N_DOUBLE / (N_DOUBLE + N_SINGLE);  // 8/32 = 0.25
const SB_FRAC = N_SINGLE / (N_DOUBLE + N_SINGLE);  // 24/32 = 0.75

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

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
    widget.type        = "hidden_flux_lora";
    widget.computeSize = () => [0, -4];
}

function defaultStrengths() {
    const db = {};
    for (let i = 0; i < N_DOUBLE; i++) db[i] = { img: 1.0, txt: 1.0 };
    const sb = {};
    for (let i = 0; i < N_SINGLE; i++) sb[i] = 1.0;
    return { db, sb };
}

app.registerExtension({
    name: "Comfy.FluxLoraGraph",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FluxLoraLoader") return;

        const _onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = _onNodeCreated?.apply(this, arguments);
            const node   = this;
            const W      = (name) => node.widgets?.find(w => w.name === name);

            setTimeout(() => {
                hideWidget(node, W("layer_strengths"));
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true, true);
            }, 0);

            // ── State ─────────────────────────────────────────────────────────
            let strengths = defaultStrengths();

            // Last non-zero trackers for toggle
            const lastDb = {};
            const lastSb = {};
            for (let i = 0; i < N_DOUBLE; i++) lastDb[i] = { img: 1.0, txt: 1.0 };
            for (let i = 0; i < N_SINGLE; i++) lastSb[i] = 1.0;

            // ── Auto-strength live watcher ─────────────────────────────────────
            // Intercepts the layer_strengths widget callback so that when
            // FluxLoraAutoStrength pushes computed JSON via a node link,
            // the bars update immediately without any manual interaction.
            function applyStrengthJSON(value) {
                try {
                    if (!value || value === "{}") return;
                    const raw = JSON.parse(value);
                    if (typeof raw !== "object") return;
                    const db = raw.db ?? {};
                    for (const [k, v] of Object.entries(db)) {
                        const i = parseInt(k, 10);
                        if (isNaN(i) || i < 0 || i >= N_DOUBLE) continue;
                        if (typeof v.img === "number") {
                            strengths.db[i].img = v.img;
                            if (v.img > 0.001) lastDb[i].img = v.img;
                        }
                        if (typeof v.txt === "number") {
                            strengths.db[i].txt = v.txt;
                            if (v.txt > 0.001) lastDb[i].txt = v.txt;
                        }
                    }
                    const sb = raw.sb ?? {};
                    for (const [k, v] of Object.entries(sb)) {
                        const i = parseInt(k, 10);
                        if (isNaN(i) || i < 0 || i >= N_SINGLE) continue;
                        if (typeof v === "number") {
                            strengths.sb[i] = v;
                            if (v > 0.001) lastSb[i] = v;
                        }
                    }
                    node.setDirtyCanvas(true, true);
                } catch(e) { /* malformed JSON — ignore */ }
            }

            // ComfyUI sets widget.value directly for linked inputs — bypasses
            // callback entirely. Use defineProperty to intercept the setter.
            setTimeout(() => {
                const lsw = W("layer_strengths");
                if (lsw) {
                    let _val = lsw.value ?? "{}";
                    const _origCb = lsw.callback?.bind(lsw);
                    Object.defineProperty(lsw, "value", {
                        get() { return _val; },
                        set(v) {
                            _val = v;
                            _origCb?.(v);
                            applyStrengthJSON(v);
                        },
                        configurable: true,
                    });
                }
            }, 10);
            // ─────────────────────────────────────────────────────────────────

            // Drag state
            let drag = null;
            // drag = { type: 'db'|'sb', idx, comp: 'img'|'txt'|'single',
            //           startY, startVal, shift }

            // Cached geometry (populated in draw)
            let _graphBounds  = null;
            let _btnBounds    = {};
            let _dbColW       = 0;
            let _sbColW       = 0;
            let _dividerX     = 0;

            // ── Helpers ───────────────────────────────────────────────────────

            function syncWidget() {
                const w = W("layer_strengths");
                if (!w) return;
                const out = { db: {}, sb: {} };
                for (let i = 0; i < N_DOUBLE; i++) {
                    out.db[i] = {
                        img: +strengths.db[i].img.toFixed(4),
                        txt: +strengths.db[i].txt.toFixed(4),
                    };
                }
                for (let i = 0; i < N_SINGLE; i++) {
                    out.sb[i] = +strengths.sb[i].toFixed(4);
                }
                const v = JSON.stringify(out);
                w.value = v;
                w.callback?.(v);
            }

            function globalStrength() {
                return W("strength_model")?.value ?? 1.0;
            }

            function hitTest(mx, my, gX, gW, gY, gH) {
                if (mx < gX || mx > gX + gW || my < gY || my > gY + gH) return null;
                const rel = mx - gX;
                const dbW = gW * DB_FRAC;

                if (rel <= dbW) {
                    // Double blocks section
                    const col  = clamp(Math.floor(rel / _dbColW), 0, N_DOUBLE - 1);
                    const comp = (my - gY) < gH / 2 ? "img" : "txt";
                    return { type: "db", idx: col, comp };
                } else {
                    // Single blocks section
                    const col = clamp(Math.floor((rel - dbW) / _sbColW), 0, N_SINGLE - 1);
                    return { type: "sb", idx: col, comp: "single" };
                }
            }

            function getVal(type, idx, comp) {
                return type === "db" ? strengths.db[idx][comp] : strengths.sb[idx];
            }

            function setVal(type, idx, comp, v) {
                const clamped = clamp(v, STR_MIN, STR_MAX);
                if (type === "db") strengths.db[idx][comp] = clamped;
                else               strengths.sb[idx]       = clamped;
            }

            function bumpAll(type, comp, delta) {
                if (type === "db") {
                    for (let i = 0; i < N_DOUBLE; i++) {
                        strengths.db[i][comp] = clamp(strengths.db[i][comp] + delta, STR_MIN, STR_MAX);
                    }
                } else {
                    for (let i = 0; i < N_SINGLE; i++) {
                        strengths.sb[i] = clamp(strengths.sb[i] + delta, STR_MIN, STR_MAX);
                    }
                }
            }

            // ── Widget ────────────────────────────────────────────────────────
            const gw = {
                name:        "flux_lora_graph",
                type:        "flux_lora_graph",
                computeSize(width) { return [width, WIDGET_H]; },

                draw(ctx, node, width, y) {
                    const iW = width - PAD * 2;
                    const gs = globalStrength();

                    // ── Button row ─────────────────────────────────────────────
                    const bY  = y + PAD;
                    const bH  = BTN_ROW_H - 4;
                    const btnW = iW * 0.3;
                    const btns = [
                        { key: "reset",   label: "↺ Reset All",     x: PAD              },
                        { key: "mirror",  label: "⇄ Img→Txt",       x: PAD + iW * 0.34  },
                        { key: "flatten", label: "▬ Flatten",        x: PAD + iW * 0.62  },
                    ];
                    btns.forEach(btn => {
                        _btnBounds[btn.key] = { ...btn, w: btnW, h: bH, y: bY };
                        roundRect(ctx, btn.x, bY, btnW, bH, 3);
                        ctx.fillStyle   = "#1a1a2e"; ctx.fill();
                        ctx.strokeStyle = "#3a3a5a"; ctx.lineWidth = 1; ctx.stroke();
                        ctx.fillStyle   = "#6655aa"; ctx.font = "bold 9px monospace";
                        ctx.textAlign   = "center";
                        ctx.fillText(btn.label, btn.x + btnW / 2, bY + bH * 0.68);
                    });
                    ctx.textAlign = "left";

                    // ── Graph area ─────────────────────────────────────────────
                    const gX = PAD;
                    const gY = bY + bH + 6;
                    const gW = iW;
                    const gH = GRAPH_H;
                    _graphBounds = { x: gX, y: gY, w: gW, h: gH };

                    // Background
                    ctx.fillStyle   = "#0a0a18";
                    ctx.strokeStyle = "#2e2e4a"; ctx.lineWidth = 1;
                    roundRect(ctx, gX, gY, gW, gH, 5); ctx.fill(); ctx.stroke();

                    // Section widths
                    const dbW    = gW * DB_FRAC;
                    const sbW    = gW * SB_FRAC;
                    _dbColW      = dbW / N_DOUBLE;
                    _sbColW      = sbW / N_SINGLE;
                    _dividerX    = gX + dbW;

                    // Section divider
                    ctx.strokeStyle = "#3a3a6a"; ctx.lineWidth = 1;
                    ctx.setLineDash([3, 3]);
                    ctx.beginPath();
                    ctx.moveTo(_dividerX, gY + 2);
                    ctx.lineTo(_dividerX, gY + gH - 2);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    // ── Global strength reference lines ────────────────────────
                    const refNorm = clamp(gs / STR_MAX, 0, 1);
                    // DB: two half-height reference lines (img top, txt bottom)
                    const dbRefY_img = gY        + (gH / 2) * (1 - refNorm);
                    const dbRefY_txt = gY + gH/2 + (gH / 2) * (1 - refNorm);
                    // SB: one full-height reference line
                    const sbRefY     = gY + gH * (1 - refNorm);

                    [dbRefY_img, dbRefY_txt].forEach(ry => {
                        ctx.strokeStyle = "rgba(255,255,255,0.10)"; ctx.lineWidth = 1;
                        ctx.setLineDash([4, 4]);
                        ctx.beginPath();
                        ctx.moveTo(gX + 1, ry);
                        ctx.lineTo(_dividerX - 1, ry);
                        ctx.stroke();
                        ctx.setLineDash([]);
                    });
                    ctx.strokeStyle = "rgba(255,255,255,0.08)"; ctx.lineWidth = 1;
                    ctx.setLineDash([4, 4]);
                    ctx.beginPath();
                    ctx.moveTo(_dividerX + 1, sbRefY);
                    ctx.lineTo(gX + gW - 1, sbRefY);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    // img/txt midline in double blocks section
                    ctx.strokeStyle = "#1c1c30"; ctx.lineWidth = 0.5;
                    ctx.setLineDash([3, 4]);
                    ctx.beginPath();
                    ctx.moveTo(gX + 1, gY + gH / 2);
                    ctx.lineTo(_dividerX - 1, gY + gH / 2);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    // ── Double block bars ──────────────────────────────────────
                    for (let i = 0; i < N_DOUBLE; i++) {
                        const { img, txt } = strengths.db[i];
                        const barX     = gX + i * _dbColW;
                        const barInW   = Math.max(1, _dbColW - 2);
                        const innerX   = barX + 1;
                        const isDragDB = drag?.type === "db" && drag.idx === i;

                        // img bar (top half)
                        const imgNorm = clamp(img / STR_MAX, 0, 1);
                        const imgH    = (gH / 2 - 3) * imgNorm;
                        const imgBarY = gY + (gH / 2 - 3) - imgH;

                        if (img > 0.001) {
                            const ig = ctx.createLinearGradient(0, imgBarY, 0, imgBarY + imgH);
                            ig.addColorStop(0, (isDragDB && drag.comp === "img") ? "#d0b8ff" : "#9b7fff");
                            ig.addColorStop(1, "#2a0a5a");
                            ctx.fillStyle = ig;
                        } else {
                            ctx.fillStyle = "#1a1a2e";
                        }
                        ctx.fillRect(innerX, imgBarY, barInW, imgH);
                        if (img > 0.001) {
                            ctx.fillStyle = (isDragDB && drag.comp === "img") ? "#ffffff" : "#c8b8ff";
                            ctx.fillRect(innerX, imgBarY, barInW, 2);
                        }

                        // txt bar (bottom half)
                        const txtNorm = clamp(txt / STR_MAX, 0, 1);
                        const txtH    = (gH / 2 - 3) * txtNorm;
                        const txtBarY = gY + gH / 2 + 3;

                        if (txt > 0.001) {
                            const tg = ctx.createLinearGradient(0, txtBarY, 0, txtBarY + txtH);
                            tg.addColorStop(0, (isDragDB && drag.comp === "txt") ? "#a0f0ff" : "#50c8f0");
                            tg.addColorStop(1, "#0a2a30");
                            ctx.fillStyle = tg;
                        } else {
                            ctx.fillStyle = "#1a1a2e";
                        }
                        ctx.fillRect(innerX, txtBarY, barInW, txtH);
                        if (txt > 0.001) {
                            ctx.fillStyle = (isDragDB && drag.comp === "txt") ? "#ffffff" : "#a0e8ff";
                            ctx.fillRect(innerX, txtBarY, barInW, 2);
                        }

                        // Hover outline
                        if (isDragDB) {
                            ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = 1;
                            ctx.strokeRect(innerX, gY + 1, barInW, gH - 2);
                        }

                        // Index label
                        ctx.fillStyle = "#3a3a5a"; ctx.font = "7px monospace";
                        ctx.textAlign = "center";
                        ctx.fillText(String(i), barX + _dbColW / 2, gY + gH - 3);
                        ctx.textAlign = "left";
                    }

                    // ── Single block bars ──────────────────────────────────────
                    for (let i = 0; i < N_SINGLE; i++) {
                        const val     = strengths.sb[i];
                        const barX    = _dividerX + i * _sbColW;
                        const barInW  = Math.max(1, _sbColW - 1);
                        const innerX  = barX + 1;
                        const isSB    = drag?.type === "sb" && drag.idx === i;

                        const norm  = clamp(val / STR_MAX, 0, 1);
                        const barH  = (gH - 4) * norm;
                        const barY  = gY + (gH - 4) - barH;

                        if (val > 0.001) {
                            const sg = ctx.createLinearGradient(0, barY, 0, barY + barH);
                            sg.addColorStop(0, isSB ? "#c0ffd8" : "#5ee89a");
                            sg.addColorStop(1, "#0a2a18");
                            ctx.fillStyle = sg;
                        } else {
                            ctx.fillStyle = "#1a1a2e";
                        }
                        ctx.fillRect(innerX, barY, barInW, barH);
                        if (val > 0.001) {
                            ctx.fillStyle = isSB ? "#ffffff" : "#a0ffcc";
                            ctx.fillRect(innerX, barY, barInW, 2);
                        }

                        if (isSB) {
                            ctx.strokeStyle = "rgba(255,255,255,0.2)"; ctx.lineWidth = 1;
                            ctx.strokeRect(innerX, gY + 1, barInW, gH - 2);
                        }

                        // Index label (every 4)
                        if (i % 4 === 0) {
                            ctx.fillStyle = "#3a3a5a"; ctx.font = "7px monospace";
                            ctx.textAlign = "center";
                            ctx.fillText(String(i), barX + _sbColW / 2, gY + gH - 3);
                            ctx.textAlign = "left";
                        }
                    }

                    // Section labels
                    ctx.fillStyle = "#4a3a7a"; ctx.font = "bold 8px monospace";
                    ctx.fillText("DOUBLE", gX + 3, gY + 11);
                    ctx.fillStyle = "#2a6a4a"; ctx.font = "bold 8px monospace";
                    ctx.fillText("SINGLE", _dividerX + 3, gY + 11);

                    ctx.fillStyle = "#5533aa"; ctx.font = "bold 7px monospace";
                    ctx.fillText("IMG", gX + 3, gY + gH / 2 - 4);
                    ctx.fillStyle = "#2a7a9a";
                    ctx.fillText("TXT", gX + 3, gY + gH / 2 + 10);

                    // Drag tooltip
                    if (drag) {
                        const val = getVal(drag.type, drag.idx, drag.comp);
                        const tipX = drag.type === "db"
                            ? gX + (drag.idx + 0.5) * _dbColW
                            : _dividerX + (drag.idx + 0.5) * _sbColW;
                        const tx = clamp(tipX, gX + 22, gX + gW - 22);
                        ctx.fillStyle = "#0d0d1a";
                        roundRect(ctx, tx - 22, gY + gH / 2 - 9, 44, 14, 3);
                        ctx.fill();
                        ctx.fillStyle = drag.type === "sb" ? "#a0ffcc" :
                                        drag.comp === "img" ? "#c8b8ff" : "#a0e8ff";
                        ctx.font = "bold 9px monospace"; ctx.textAlign = "center";
                        ctx.fillText(val.toFixed(2), tx, gY + gH / 2);
                        ctx.textAlign = "left";
                    }

                    // ── Label row ──────────────────────────────────────────────
                    const lY = gY + gH + 6;
                    ctx.fillStyle = "#4a4a6a"; ctx.font = "8px monospace";
                    ctx.fillText(`global: ${gs.toFixed(2)}`, gX + 2, lY + 11);
                    ctx.fillStyle = "#5533aa";
                    ctx.fillText("■ img", gX + 64, lY + 11);
                    ctx.fillStyle = "#2a7a9a";
                    ctx.fillText("■ txt", gX + 96, lY + 11);
                    ctx.fillStyle = "#2a6a4a";
                    ctx.fillText("■ single", gX + 124, lY + 11);
                    ctx.fillStyle = "#3a3a5a";
                    ctx.fillText("drag↕ | click=toggle | shift=all", gX + gW - 200, lY + 11);
                },

                mouse(event, pos, node) {
                    const [mx, my] = pos;
                    const gs = globalStrength();

                    // ── Button clicks ──────────────────────────────────────────
                    if (event.type === "pointerdown") {
                        for (const [key, b] of Object.entries(_btnBounds)) {
                            if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
                                if (key === "reset") {
                                    strengths = defaultStrengths();
                                } else if (key === "mirror") {
                                    for (let i = 0; i < N_DOUBLE; i++) strengths.db[i].txt = strengths.db[i].img;
                                } else if (key === "flatten") {
                                    for (let i = 0; i < N_DOUBLE; i++) {
                                        strengths.db[i].img = gs;
                                        strengths.db[i].txt = gs;
                                    }
                                    for (let i = 0; i < N_SINGLE; i++) strengths.sb[i] = gs;
                                }
                                syncWidget();
                                node.setDirtyCanvas(true, true);
                                return true;
                            }
                        }
                    }

                    if (!_graphBounds) return false;
                    const { x: gX, y: gY, w: gW, h: gH } = _graphBounds;

                    // ── Drag start ─────────────────────────────────────────────
                    if (event.type === "pointerdown") {
                        const hit = hitTest(mx, my, gX, gW, gY, gH);
                        if (!hit) return false;
                        drag = {
                            ...hit,
                            startY:   my,
                            startVal: getVal(hit.type, hit.idx, hit.comp),
                            shift:    event.shiftKey,
                        };
                        node.setDirtyCanvas(true, true);
                        return true;
                    }

                    // ── Drag end / toggle ──────────────────────────────────────
                    if ((event.type === "pointerup" || event.type === "pointercancel") && drag) {
                        if (Math.abs(my - drag.startY) < 4) {
                            // Small movement = toggle
                            const cur = getVal(drag.type, drag.idx, drag.comp);
                            if (cur < 0.01) {
                                // Restore last non-zero
                                if (drag.type === "db") {
                                    setVal("db", drag.idx, drag.comp, lastDb[drag.idx][drag.comp]);
                                } else {
                                    setVal("sb", drag.idx, "single", lastSb[drag.idx]);
                                }
                            } else {
                                // Save and zero
                                if (drag.type === "db") lastDb[drag.idx][drag.comp] = cur;
                                else                    lastSb[drag.idx]            = cur;
                                setVal(drag.type, drag.idx, drag.comp, 0);
                            }
                        }
                        drag = null;
                        syncWidget();
                        node.setDirtyCanvas(true, true);
                        return false;
                    }

                    // ── Drag move ──────────────────────────────────────────────
                    if (event.type === "pointermove" && drag) {
                        const halfH  = drag.type === "db" ? gH / 2 : gH;
                        const dy     = drag.startY - my;
                        const delta  = (dy / (halfH - 4)) * STR_MAX;
                        const newVal = clamp(drag.startVal + delta, STR_MIN, STR_MAX);

                        if (drag.shift) {
                            const diff = newVal - drag.startVal;
                            bumpAll(drag.type, drag.comp, diff);
                            drag.startVal = newVal;
                            drag.startY   = my;
                        } else {
                            setVal(drag.type, drag.idx, drag.comp, newVal);
                        }

                        syncWidget();
                        node.setDirtyCanvas(true, true);
                        return true;
                    }

                    return false;
                },

                serializeValue() { return undefined; },
            };

            if (!node.widgets) node.widgets = [];
            node.widgets.push(gw);
            node.setSize(node.computeSize());

            // ── Restore state after workflow reload ────────────────────────────
            const origConfigure = node.onConfigure?.bind(node);
            node.onConfigure = function (config) {
                origConfigure?.(config);
                setTimeout(() => {
                    const w = W("layer_strengths");
                    if (!w || !w.value || w.value === "{}") return;
                    try {
                        const raw = JSON.parse(w.value);
                        if (typeof raw !== "object") return;

                        const db = raw.db ?? {};
                        for (const [k, v] of Object.entries(db)) {
                            const i = parseInt(k, 10);
                            if (isNaN(i) || i < 0 || i >= N_DOUBLE) continue;
                            if (typeof v.img === "number") {
                                strengths.db[i].img = v.img;
                                if (v.img > 0.001) lastDb[i].img = v.img;
                            }
                            if (typeof v.txt === "number") {
                                strengths.db[i].txt = v.txt;
                                if (v.txt > 0.001) lastDb[i].txt = v.txt;
                            }
                        }

                        const sb = raw.sb ?? {};
                        for (const [k, v] of Object.entries(sb)) {
                            const i = parseInt(k, 10);
                            if (isNaN(i) || i < 0 || i >= N_SINGLE) continue;
                            if (typeof v === "number") {
                                strengths.sb[i] = v;
                                if (v > 0.001) lastSb[i] = v;
                            }
                        }

                        node.setDirtyCanvas(true, true);
                    } catch (e) {
                        // Malformed JSON — stay with defaults
                    }
                }, 0);
            };

            // ── Gray out lora_name dropdown when lora_name_override is linked ──
            const _origDraw = nodeType.prototype.onDrawBackground?.bind(nodeType.prototype);
            const _checkOverride = function() {
                const loraWidget     = W("lora_name");
                const overrideWidget = W("lora_name_override");
                if (!loraWidget) return;

                // Check if lora_name_override input has a link
                const overrideInput = node.inputs?.find(inp => inp.name === "lora_name_override");
                const isLinked = overrideInput?.link != null;

                if (isLinked) {
                    loraWidget.disabled  = true;
                    loraWidget._origType = loraWidget._origType || loraWidget.type;
                    // Force ComfyUI to render it grayed
                    loraWidget.computeSize = () => [0, 22];
                    if (!node._overrideBannerShown) {
                        node._overrideBannerShown = true;
                    }
                } else {
                    loraWidget.disabled = false;
                    delete node._overrideBannerShown;
                }
                node.setDirtyCanvas(true, false);
            };

            // Patch onConnectionsChange to react when override link is added/removed
            const _origConnChange = nodeType.prototype.onConnectionsChange?.bind(nodeType.prototype);
            nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
                _origConnChange?.(type, index, connected, link_info);
                _checkOverride();
            };

            // Also run on draw so it stays correct on workflow load
            const _origDrawFG = nodeType.prototype.onDrawForeground?.bind(nodeType.prototype);
            nodeType.prototype.onDrawForeground = function(ctx) {
                _origDrawFG?.(ctx);

                const overrideInput = this.inputs?.find(inp => inp.name === "lora_name_override");
                const isLinked = overrideInput?.link != null;
                const loraWidget = this.widgets?.find(w => w.name === "lora_name");

                if (loraWidget && isLinked) {
                    // Draw a gray overlay bar over the lora_name widget area
                    const wY = loraWidget.last_y ?? 0;
                    const wW = this.size[0] - 20;
                    ctx.save();
                    ctx.fillStyle   = "rgba(20, 20, 30, 0.75)";
                    ctx.strokeStyle = "#3a3a5a";
                    ctx.lineWidth   = 1;
                    ctx.beginPath();
                    ctx.roundRect(10, wY, wW, 22, 4);
                    ctx.fill();
                    ctx.stroke();
                    ctx.fillStyle = "#555577";
                    ctx.font      = "12px monospace";
                    ctx.textAlign = "center";
                    ctx.fillText("⟵ overridden by link", this.size[0] / 2, wY + 15);
                    ctx.restore();
                }
            };
            // ─────────────────────────────────────────────────────────────────

            return result;
        };
    },
});
