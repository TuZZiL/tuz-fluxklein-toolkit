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
//   Buttons: Reset | Match global | Protect Face | Protect Body | Style Bias
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
const BADGE_H   = 18;
const WIDGET_H  = GRAPH_H + BTN_ROW_H + LABEL_H + BADGE_H + PAD * 4;
const MIN_NODE_W = 430;

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
    widget.computeSize = () => [0, 0];
    widget.draw        = () => {};
}

function defaultStrengths() {
    const db = {};
    for (let i = 0; i < N_DOUBLE; i++) db[i] = { img: 1.0, txt: 1.0 };
    const sb = {};
    for (let i = 0; i < N_SINGLE; i++) sb[i] = 1.0;
    return { db, sb };
}

const GRAPH_PRESETS = {
    face: {
        db: {
            0: { img: 1.0, txt: 0.90 }, 1: { img: 1.0, txt: 0.90 },
            2: { img: 1.0, txt: 0.90 }, 3: { img: 1.0, txt: 0.90 },
            4: { img: 1.0, txt: 0.85 }, 5: { img: 1.0, txt: 0.85 },
            6: { img: 1.0, txt: 0.85 }, 7: { img: 1.0, txt: 0.85 },
        },
        sb: {
            0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.95, 7: 0.95,
            8: 0.90, 9: 0.85, 10: 0.80, 11: 0.75, 12: 0.65, 13: 0.60, 14: 0.55, 15: 0.50,
            16: 0.45, 17: 0.40, 18: 0.38, 19: 0.35, 20: 0.33, 21: 0.32, 22: 0.30, 23: 0.30,
        },
    },
    body: {
        db: {
            0: { img: 1.0, txt: 0.85 }, 1: { img: 1.0, txt: 0.85 },
            2: { img: 1.0, txt: 0.85 }, 3: { img: 1.0, txt: 0.85 },
            4: { img: 1.0, txt: 0.80 }, 5: { img: 1.0, txt: 0.80 },
            6: { img: 1.0, txt: 0.80 }, 7: { img: 1.0, txt: 0.80 },
        },
        sb: {
            0: 1.0, 1: 1.0, 2: 1.0, 3: 0.95, 4: 0.75, 5: 0.72, 6: 0.70, 7: 0.68,
            8: 0.65, 9: 0.62, 10: 0.60, 11: 0.55, 12: 0.50, 13: 0.47, 14: 0.44, 15: 0.40,
            16: 0.38, 17: 0.35, 18: 0.33, 19: 0.32, 20: 0.30, 21: 0.30, 22: 0.30, 23: 0.30,
        },
    },
    style: {
        db: {
            0: { img: 0.40, txt: 1.0 }, 1: { img: 0.40, txt: 1.0 },
            2: { img: 0.45, txt: 1.0 }, 3: { img: 0.45, txt: 1.0 },
            4: { img: 0.50, txt: 1.0 }, 5: { img: 0.50, txt: 1.0 },
            6: { img: 0.55, txt: 1.0 }, 7: { img: 0.55, txt: 1.0 },
        },
        sb: {
            0: 1.0, 1: 1.0, 2: 1.0, 3: 0.95, 4: 0.90, 5: 0.85, 6: 0.80, 7: 0.75,
            8: 0.70, 9: 0.65, 10: 0.60, 11: 0.55, 12: 0.50, 13: 0.45, 14: 0.40, 15: 0.38,
            16: 0.35, 17: 0.33, 18: 0.30, 19: 0.30, 20: 0.30, 21: 0.30, 22: 0.30, 23: 0.30,
        },
    },
};

app.registerExtension({
    name: "Comfy.FluxLoraGraph",

    async setup() {
        function findNodeByExecutionId(executionId) {
            const target = String(executionId ?? "");
            if (!target) return null;
            const localId = Number(target.split(":").at(-1));
            if (!Number.isFinite(localId)) return null;
            return app.graph?._nodes?.find((node) => node?.id === localId && node?.type === "FluxLoraLoader") ?? null;
        }

        function messageHandler(event) {
            const detail = event?.detail ?? {};
            const node = findNodeByExecutionId(detail.node);
            const value = detail.layer_strengths;
            if (!node || typeof value !== "string") return;
            const widget = node.widgets?.find((w) => w.name === "layer_strengths");
            if (!widget) return;
            widget.value = value;
            widget.callback?.(value);
            node.setDirtyCanvas(true, true);
        }

        function compatHandler(event) {
            const detail = event?.detail ?? {};
            const node = findNodeByExecutionId(detail.node);
            if (!node) return;
            node._fluxCompatReport = {
                status: detail.status ?? "failed",
                matched_modules: Number(detail.matched_modules ?? 0),
                total_modules: Number(detail.total_modules ?? 0),
                skipped_modules: Number(detail.skipped_modules ?? 0),
            };
            node.setDirtyCanvas(true, true);
        }

        app.api.addEventListener("flux_lora.auto_strength", messageHandler);
        app.api.addEventListener("flux_lora.compat_report", compatHandler);
    },

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
            node._fluxCompatReport = null;

            // Last non-zero trackers for toggle
            const lastDb = {};
            const lastSb = {};
            for (let i = 0; i < N_DOUBLE; i++) lastDb[i] = { img: 1.0, txt: 1.0 };
            for (let i = 0; i < N_SINGLE; i++) lastSb[i] = 1.0;

            // ── Auto-strength live watcher ─────────────────────────────────────
            // Intercepts the layer_strengths widget value setter so that when
            // auto_strength computes per-layer JSON or the value changes,
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
                return W("strength")?.value ?? 1.0;
            }

            function graphBaseStrength() {
                return Math.max(0.0, Math.abs(globalStrength()));
            }

            function rememberCurrentAsLast() {
                for (let i = 0; i < N_DOUBLE; i++) {
                    if (strengths.db[i].img > 0.001) lastDb[i].img = strengths.db[i].img;
                    if (strengths.db[i].txt > 0.001) lastDb[i].txt = strengths.db[i].txt;
                }
                for (let i = 0; i < N_SINGLE; i++) {
                    if (strengths.sb[i] > 0.001) lastSb[i] = strengths.sb[i];
                }
            }

            function applyGraphPreset(kind) {
                if (kind === "reset") {
                    strengths = defaultStrengths();
                    rememberCurrentAsLast();
                    return;
                }

                const gs = graphBaseStrength();
                if (kind === "global") {
                    for (let i = 0; i < N_DOUBLE; i++) {
                        strengths.db[i].img = gs;
                        strengths.db[i].txt = gs;
                    }
                    for (let i = 0; i < N_SINGLE; i++) strengths.sb[i] = gs;
                    rememberCurrentAsLast();
                    return;
                }

                const mask = GRAPH_PRESETS[kind];
                if (!mask) return;
                for (let i = 0; i < N_DOUBLE; i++) {
                    const cfg = mask.db[i];
                    strengths.db[i].img = clamp(gs * cfg.img, STR_MIN, STR_MAX);
                    strengths.db[i].txt = clamp(gs * cfg.txt, STR_MIN, STR_MAX);
                }
                for (let i = 0; i < N_SINGLE; i++) {
                    strengths.sb[i] = clamp(gs * mask.sb[i], STR_MIN, STR_MAX);
                }
                rememberCurrentAsLast();
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
                    const btnGap = 6;
                    const btnW = (iW - btnGap * 4) / 5;
                    const btns = [
                        { key: "reset",  label: "Reset" },
                        { key: "global", label: "Global" },
                        { key: "face",   label: "Face" },
                        { key: "body",   label: "Body" },
                        { key: "style",  label: "Style" },
                    ];
                    btns.forEach((btn, idx) => {
                        const btnX = PAD + idx * (btnW + btnGap);
                        _btnBounds[btn.key] = { ...btn, w: btnW, h: bH, y: bY };
                        _btnBounds[btn.key].x = btnX;
                        roundRect(ctx, btnX, bY, btnW, bH, 3);
                        ctx.fillStyle   = "#1a1a2e"; ctx.fill();
                        ctx.strokeStyle = "#3a3a5a"; ctx.lineWidth = 1; ctx.stroke();
                        ctx.fillStyle   = "#6655aa"; ctx.font = "bold 9px monospace";
                        ctx.textAlign   = "center";
                        ctx.fillText(btn.label, btnX + btnW / 2, bY + bH * 0.68);
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

                    const legendY = lY + 11;
                    let legendX = gX + 64;

                    ctx.fillStyle = "#5533aa";
                    ctx.fillText("■ image", legendX, legendY);
                    legendX += ctx.measureText("■ image").width + 10;

                    ctx.fillStyle = "#2a7a9a";
                    ctx.fillText("■ text", legendX, legendY);
                    legendX += ctx.measureText("■ text").width + 10;

                    ctx.fillStyle = "#2a6a4a";
                    ctx.fillText("■ single", legendX, legendY);
                    legendX += ctx.measureText("■ single").width + 16;

                    const helpText = gW >= 520
                        ? "drag to adjust • click toggles • shift moves all"
                        : gW >= 450
                            ? "drag • click toggle • shift all"
                            : "drag • click • shift";
                    const helpWidth = ctx.measureText(helpText).width;
                    const helpX = gX + gW - helpWidth - 4;
                    ctx.fillStyle = "#3a3a5a";
                    if (helpX > legendX) {
                        ctx.fillText(helpText, helpX, legendY);
                    }

                    const compat = node._fluxCompatReport;
                    if (compat) {
                        const badgeText = `Compat ${compat.matched_modules}/${compat.total_modules}`;
                        const badgeY = lY + LABEL_H + 3;
                        const badgeW = Math.max(84, ctx.measureText(badgeText).width + 16);
                        const badgeX = gX + 2;
                        let fill = "#2a1a1a";
                        let stroke = "#5a2a2a";
                        let text = "#ff9b9b";
                        if (compat.status === "ok") {
                            fill = "#14261a";
                            stroke = "#2f6a42";
                            text = "#9bf0b6";
                        } else if (compat.status === "partial") {
                            fill = "#2a2412";
                            stroke = "#7a6230";
                            text = "#ffd37a";
                        }
                        roundRect(ctx, badgeX, badgeY, badgeW, BADGE_H - 4, 4);
                        ctx.fillStyle = fill;
                        ctx.fill();
                        ctx.strokeStyle = stroke;
                        ctx.lineWidth = 1;
                        ctx.stroke();
                        ctx.fillStyle = text;
                        ctx.font = "bold 9px monospace";
                        ctx.fillText(badgeText, badgeX + 8, badgeY + 10);
                    }
                },

                mouse(event, pos, node) {
                    const [mx, my] = pos;
                    const gs = globalStrength();

                    // ── Button clicks ──────────────────────────────────────────
                    if (event.type === "pointerdown") {
                        for (const [key, b] of Object.entries(_btnBounds)) {
                            if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
                                applyGraphPreset(key);
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
            if (node.size[0] < MIN_NODE_W) {
                node.size[0] = MIN_NODE_W;
            }
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


            return result;
        };
    },
});
