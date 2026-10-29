// Builds "VMD UVMD Comparison.pptx" (22 slides). Run: node deck/build_pptx.js
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/d4baf5f1-ecad-4957-853d-2cd53cc6a216_4180cbdd-69c9-4c47-a69c-ce98f5f5c39a/pptx/scripts/apply_theme.js");
const path = require("path");
const R = path.join(__dirname, "..", "results");
const IMG = {
  bars: path.join(R, "separation_comparison", "comparison_bars.png"),
  truth: path.join(R, "separation_comparison", "synthetic_proxy_vs_truth.png"),
  omega: path.join(R, "omega_shift_all", "omega_shift_all.png"),
};
const OUT = process.argv[2] || "VMD UVMD Comparison.pptx";

const THEME = {
  name: "VMD UVMD", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "121A2B", lt1: "FFFFFF", dk2: "1C2740", lt2: "F2F4F7", accent1: "E8833A", accent2: "1F5FA8",
    accent3: "7DB8F0", accent4: "4A5568", accent5: "C4621F", accent6: "B8C4D9", hlink: "1F5FA8", folHlink: "4A5568" },
};
const DARK = "121A2B", PANEL = "1C2740", LIGHT = "F2F4F7", WHITE = "FFFFFF", ORANGE = "E8833A", DORANGE = "C4621F",
  BLUE = "1F5FA8", LBLUE = "7DB8F0", MUTED = "4A5568", MUTEDL = "B8C4D9", BORDER = "D5DAE2", TXT = "2D3748";
const MONO = "Courier New";

// ---------- native PowerPoint equations (OMML) ----------
// Equations are written as placeholder runs "@@EQ:id:size:color@@" and replaced by real <a14:m> math after the file is built.
const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const r = (t) => `<m:r>@RPR@<m:t>${esc(t)}</m:t></m:r>`;
const rp = (t) => `<m:r><m:rPr><m:sty m:val="p"/></m:rPr>@RPR@<m:t>${esc(t)}</m:t></m:r>`; // upright text
const sub = (b, s) => `<m:sSub><m:e>${b}</m:e><m:sub>${s}</m:sub></m:sSub>`;
const sup = (b, s) => `<m:sSup><m:e>${b}</m:e><m:sup>${s}</m:sup></m:sSup>`;
const subsup = (b, s, p) => `<m:sSubSup><m:e>${b}</m:e><m:sub>${s}</m:sub><m:sup>${p}</m:sup></m:sSubSup>`;
const frac = (n, d) => `<m:f><m:num>${n}</m:num><m:den>${d}</m:den></m:f>`;
const hat = (b) => `<m:acc><m:accPr><m:chr m:val="&#770;"/></m:accPr><m:e>${b}</m:e></m:acc>`;
const dl = (c, a = "(", b = ")") => `<m:d><m:dPr><m:begChr m:val="${a}"/><m:endChr m:val="${b}"/></m:dPr><m:e>${c}</m:e></m:d>`;
const nary = (chr, lo, hi, body) => `<m:nary><m:naryPr><m:chr m:val="${chr}"/><m:limLoc m:val="${chr === "∫" ? "subSup" : "undOvr"}"/>${lo ? "" : '<m:subHide m:val="1"/>'}${hi ? "" : '<m:supHide m:val="1"/>'}</m:naryPr><m:sub>${lo || ""}</m:sub><m:sup>${hi || ""}</m:sup><m:e>${body}</m:e></m:nary>`;
const limlow = (b, l) => `<m:limLow><m:e>${b}</m:e><m:lim>${l}</m:lim></m:limLow>`;
const EQS = {};
const defEq = (id, omml, fallback) => { EQS[id] = { omml, fallback }; };
const [U, W, F, L] = [r("u"), r("ω"), r("f"), r("λ")];
const uk = sub(U, r("k")), wk = sub(W, r("k"));
const uh = hat(U), fh = hat(F), lh = hat(L);
const wArg = dl(W);
const absq = (x) => dl(x, "|", "|");

defEq("vmd_min", limlow(rp("min"), dl(uk, "{", "}") + r(",") + dl(wk, "{", "}")) + nary("∑", r("k"), "", subsup(dl(sub(r("∂"), r("t")) + dl(dl(r("δ(t)+") + frac(r("j"), r("πt"))) + r("∗") + uk + dl(r("t")), "[", "]") + sup(r("e"), r("−j") + wk + r("t"))), r("2"), r("2"))),
  "min Σ_k ‖∂t[(δ(t)+j/(πt)) ∗ u_k(t)] e^(−jω_k t)‖²");
defEq("vmd_st", rp("s.t.  ") + nary("∑", r("k"), "", uk) + r("=f"), "s.t. Σ_k u_k = f");
defEq("admm_u", subsup(uh, r("k"), r("n+1")) + dl(W) + r("=") + frac(fh + wArg + r("−") + nary("∑", r("i≠k"), "", sub(uh, r("i")) + wArg) + r("+") + frac(lh + wArg, r("2")), r("1+2α") + sup(dl(W + r("−") + wk), r("2"))), "û_k^(n+1)(ω) = (f̂ − Σ û_i + λ̂/2) / (1 + 2α(ω−ω_k)²)");
defEq("admm_w", subsup(W, r("k"), r("n+1")) + r("=") + frac(nary("∫", r("0"), r("∞"), W + sup(absq(sub(uh, r("k")) + wArg), r("2")) + r("dω")), nary("∫", r("0"), r("∞"), sup(absq(sub(uh, r("k")) + wArg), r("2")) + r("dω"))), "ω_k^(n+1) = ∫ ω|û_k|² dω / ∫ |û_k|² dω");
defEq("admm_l", sup(lh, r("n+1")) + r("←") + sup(lh, r("n")) + r("+τ") + dl(fh + r("−") + nary("∑", r("k"), "", sub(uh, r("k")))), "λ̂ ← λ̂ + τ(f̂ − Σ_k û_k)");
defEq("uvmd_u", sub(uh, r("k")) + r("=") + frac(sub(fh, r("+")) + r("−") + nary("∑", r("i≠k"), "", sub(uh, r("i"))) + r("−") + frac(sub(lh, r("n")), r("2")), r("1+softplus") + dl(sub(r("α"), r("k"))) + sup(dl(W + r("−") + wk), r("2"))), "û_k = (f̂₊ − Σ û_i − λ̂_n/2) / (1 + softplus(α_k)(ω−ω_k)²)");
defEq("paper_u", sub(uh, r("k")) + r("=") + frac(sub(fh, r("+")) + r("−") + nary("∑", r("i≠k"), "", sub(uh, r("i"))) + r("+") + frac(sub(lh, r("n")), r("2")), r("1+2softplus") + dl(sub(r("α"), r("k"))) + sup(dl(W + r("−") + wk), r("2"))), "û_k = (f̂₊ − Σ û_i + λ̂_n/2) / (1 + 2softplus(α_k)(ω−ω_k)²)");
defEq("uvmd_w", wk + r("=") + frac(nary("∑", r("ω"), "", W + sup(absq(sub(uh, r("k"))), r("2"))), nary("∑", r("ω"), "", sup(absq(sub(uh, r("k"))), r("2")))), "ω_k = Σ ω|û_k|² / Σ|û_k|²");
defEq("unf_iter", r("x←T") + dl(r("x;θ")), "x ← T(x; θ)");
defEq("unf_1", sub(r("x"), r("1")) + r("=T") + dl(sub(r("x"), r("0")) + r(";") + sub(r("θ"), r("1"))), "x₁ = T(x₀; θ₁)");
defEq("unf_2", sub(r("x"), r("2")) + r("=T") + dl(sub(r("x"), r("1")) + r(";") + sub(r("θ"), r("2"))), "x₂ = T(x₁; θ₂)");
defEq("unf_n", sub(r("x"), r("N")) + r("=T") + dl(sub(r("x"), r("N−1")) + r(";") + sub(r("θ"), r("N"))), "x_N = T(x_{N−1}; θ_N)");
defEq("loss", r("𝓛=") + nary("∑", r("ω"), "", absq(nary("∑", r("k"), "", sub(uh, r("k")) + wArg) + r("−") + sub(fh, r("+")) + wArg)), "L = Σ_ω |Σ_k û_k(ω) − f̂₊(ω)|");
defEq("ortho", rp("orthogonality = ") + frac(nary("∑", r("i<j"), "", absq(dl(sub(r("u"), r("i")) + r(",") + sub(r("u"), r("j")), "⟨", "⟩"))), sup(dl(r("x"), "‖", "‖"), r("2"))), "orthogonality = Σ_{i<j}|⟨u_i,u_j⟩| / ‖x‖²");
defEq("energy", rp("energy ratio = ") + frac(nary("∑", r("k"), "", sup(dl(uk, "‖", "‖"), r("2"))), sup(dl(r("x"), "‖", "‖"), r("2"))), "energy ratio = Σ_k ‖u_k‖² / ‖x‖²");
defEq("neff", sub(r("N"), rp("eff")) + r("=") + frac(r("1"), nary("∑", r("k"), "", sup(sub(r("s"), r("k")), r("2")))), "N_eff = 1 / Σ_k s_k²");
defEq("recon", rp("rel. error = ") + frac(dl(r("x−") + nary("∑", r("k"), "", uk), "‖", "‖"), dl(r("x"), "‖", "‖")), "‖x − Σ_k u_k‖ / ‖x‖");

// a text box holding native equations, one per paragraph; ids = equation ids
function eqText(s, ids, x, y, w, h, o = {}) {
  const size = o.size || 22, color = o.color || "F4F1EA";
  const runs = ids.map((id, i) => ({ text: `@@EQ:${id}:${size * 100}:${color}@@`, options: { fontSize: size, color, breakLine: i < ids.length - 1, paraSpaceAfter: o.gap || 10 } }));
  s.addText(runs, { x, y, w, h, valign: "middle", align: "left", margin: 0, isTextBox: true });
}
function eqCard(s, ids, x, y, w, h, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: o.fill || DARK }, line: { color: o.fill || DARK, width: 1 } });
  eqText(s, ids, x + 0.3, y + 0.1, w - 0.6, h - 0.2, o);
}
async function injectEquations(file) {
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(require("fs").readFileSync(file));
  const re = /<a:r>(?:(?!<\/a:r>)[\s\S])*?@@EQ:(\w+):(\d+):(\w+)@@(?:(?!<\/a:r>)[\s\S])*?<\/a:r>/g;
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n))) {
    let xml = await zip.file(name).async("string");
    if (!xml.includes("@@EQ:")) continue;
    xml = xml.replace(re, (_m, id, sz, col) => {
      const e = EQS[id];
      if (!e) throw new Error("unknown equation " + id);
      const rpr = `<a:rPr lang="en-US" sz="${sz}" i="1"><a:solidFill><a:srgbClr val="${col}"/></a:solidFill><a:latin typeface="Cambria Math"/><a:cs typeface="Cambria Math"/></a:rPr>`;
      const omml = e.omml.split("@RPR@").join(rpr);
      const fb = `<a:r><a:rPr lang="en-US" sz="${sz}"><a:solidFill><a:srgbClr val="${col}"/></a:solidFill></a:rPr><a:t>${esc(e.fallback)}</a:t></a:r>`;
      return `<mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"><mc:Choice xmlns:a14="http://schemas.microsoft.com/office/drawing/2010/main" Requires="a14"><a14:m><m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMathParaPr><m:jc m:val="left"/></m:oMathParaPr><m:oMath>${omml}</m:oMath></m:oMathPara></a14:m></mc:Choice><mc:Fallback>${fb}</mc:Fallback></mc:AlternateContent>`;
    });
    zip.file(name, xml);
  }
  require("fs").writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "VMD UVMD Comparison";
pres.author = "Aitzaz";
pres.theme = { headFontFace: "Cambria", bodyFontFace: "Calibri" };

function slide(title, opts = {}) {
  const s = pres.addSlide();
  const dark = !!opts.dark;
  s.background = { color: dark ? DARK : LIGHT };
  s.dark = dark;
  if (opts.kicker) s.addText(opts.kicker.toUpperCase(), { x: 0.6, y: 0.4, w: 12, h: 0.35, fontSize: 14, bold: true, color: dark ? ORANGE : DORANGE, charSpacing: 2, margin: 0, isTextBox: true });
  if (title) s.addText(title, { x: 0.6, y: opts.kicker ? 0.75 : 0.5, w: 12.1, h: 0.9, fontFace: "Cambria", fontSize: 34, bold: true, color: dark ? "F4F1EA" : DARK, margin: 0, valign: "top", isTextBox: true });
  if (opts.notes) s.addNotes(opts.notes);
  return s;
}
function card(s, x, y, w, h, head, body, o = {}) {
  const dark = o.dark;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: dark ? (o.fill || DARK) : WHITE }, line: { color: o.line || (dark ? (o.fill || DARK) : BORDER), width: o.lw || 1 } });
  const runs = [];
  if (o.tag) runs.push({ text: o.tag, options: { fontSize: 12, bold: true, color: o.tagColor || (dark ? ORANGE : DORANGE), breakLine: true } });
  if (head) runs.push({ text: head, options: { fontSize: o.hs || 20, bold: true, color: dark ? "F4F1EA" : DARK, breakLine: true } });
  if (body) runs.push({ text: body, options: { fontSize: o.bs || 15, color: dark ? MUTEDL : MUTED } });
  s.addText(runs, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: h - 0.24, valign: "top", margin: 0, paraSpaceAfter: 4, isTextBox: true });
}
function bullets(s, items, x, y, w, h, size = 17, color) {
  const dark = s.dark;
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 } })),
    { x, y, w, h, fontSize: size, color: color || (dark ? "D5DAE6" : TXT), valign: "top", margin: 0, isTextBox: true });
}
function eq(s, lines, x, y, w, h, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: o.fill || DARK }, line: { color: o.fill || DARK, width: 1 } });
  const runs = lines.map((l, i) => ({ text: l.t || l, options: { fontFace: MONO, fontSize: o.size || 17, color: l.c || "F4F1EA", breakLine: i < lines.length - 1, paraSpaceAfter: 6 } }));
  s.addText(runs, { x: x + 0.3, y: y + 0.15, w: w - 0.6, h: h - 0.3, valign: "middle", margin: 0, isTextBox: true });
}
function table(s, rows, x, y, w, colW, o = {}) {
  const fs = o.size || 16;
  const data = rows.map((r, ri) => r.map((c, ci) => ({
    text: String(c),
    options: {
      bold: ri === 0 || (o.boldCells && o.boldCells.includes(`${ri},${ci}`)),
      color: ri === 0 ? WHITE : DARK, fill: { color: ri === 0 ? DARK : (ri % 2 ? WHITE : "E9EDF3") },
      align: ci > 0 && o.right ? "right" : "left", valign: "middle", fontSize: fs, fontFace: "Calibri",
    },
  })));
  s.addTable(data, { x, y, w, colW, rowH: o.rowH || 0.5, border: { type: "solid", color: BORDER, pt: 0.75 }, margin: [0.04, 0.1, 0.04, 0.1] });
}
function pill(s, text, x, y, w) {
  s.addText(text, { x, y, w, h: 0.42, shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.2, fill: { color: DARK }, color: "F4F1EA", fontSize: 14, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function arrow(s, x, y) { s.addText("→", { x, y, w: 0.5, h: 0.6, fontSize: 28, color: DORANGE, bold: true, align: "center", valign: "middle", margin: 0, isTextBox: true }); }

// 1 cover
{
  const s = slide(null, { dark: true, notes: "Plan: classic VMD, the idea of unfolding, UVMD in detail, theoretical caveats, then reconstruction and separation results on synthetic, CWRU and LargeST data." });
  s.addText("VARIATIONAL MODE DECOMPOSITION", { x: 0.9, y: 1.7, w: 11.5, h: 0.4, fontSize: 16, bold: true, color: ORANGE, charSpacing: 4, margin: 0, isTextBox: true });
  s.addText("From VMD to unfolded VMD", { x: 0.9, y: 2.2, w: 11.5, h: 1.3, fontFace: "Cambria", fontSize: 54, bold: true, color: "F4F1EA", margin: 0, isTextBox: true });
  s.addText("The ADMM solver, its deep-unfolded version UVMD (Ahmad et al., 2025), where it falls short in theory, and a three-dataset comparison at K = 13", { x: 0.9, y: 3.7, w: 10.5, h: 1.2, fontSize: 22, color: MUTEDL, margin: 0, valign: "top", isTextBox: true });
  s.addText("Aitzaz · SPML group, Aarhus University · October 2026", { x: 0.9, y: 6.5, w: 11, h: 0.4, fontSize: 14, color: "8FA0BC", margin: 0, isTextBox: true });
}
// 2 outline
{
  const s = slide("What this talk covers");
  const rows = [["1", "Classic VMD", "The problem, the ADMM updates, how the loop runs"], ["2", "Deep unfolding", "Iterations become layers, hyperparameters become weights"],
    ["3", "UVMD (Ahmad et al.)", "Layer, trained parameters, training, paper versus code"], ["4", "Where it falls short", "Theoretical caveats and how far ω moves in practice"],
    ["5", "Results", "Reconstruction, then separation: synthetic, CWRU, LargeST"]];
  rows.forEach((r, i) => {
    const y = 1.7 + i * 1.02;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 12.1, h: 0.88, rectRadius: 0.1, fill: { color: WHITE }, line: { color: BORDER, width: 1 } });
    s.addText(r[0], { x: 0.85, y, w: 0.6, h: 0.88, fontSize: 28, bold: true, color: DORANGE, valign: "middle", margin: 0, isTextBox: true });
    s.addText(r[1], { x: 1.6, y, w: 3.8, h: 0.88, fontSize: 22, bold: true, color: DARK, valign: "middle", margin: 0, isTextBox: true });
    s.addText(r[2], { x: 5.5, y, w: 7.0, h: 0.88, fontSize: 18, color: MUTED, valign: "middle", margin: 0, isTextBox: true });
  });
}
// 3 vmd problem
{
  const s = slide("VMD: split a signal into K narrow-band modes", { kicker: "Classic VMD", notes: "VMD minimises the summed bandwidth of K modes, each centred on its own frequency, under the constraint that the modes reconstruct the signal exactly." });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.9, w: 7.0, h: 3.6, rectRadius: 0.1, fill: { color: DARK }, line: { color: DARK, width: 1 } });
  s.addText("CONSTRAINED PROBLEM", { x: 0.9, y: 2.05, w: 6.4, h: 0.35, fontSize: 13, bold: true, color: ORANGE, margin: 0, isTextBox: true });
  eqText(s, ["vmd_min", "vmd_st"], 0.9, 2.5, 6.4, 2.8, { size: 20, gap: 18 });
  s.addText("Reading the terms", { x: 8.0, y: 1.9, w: 4.7, h: 0.4, fontSize: 20, bold: true, color: DARK, margin: 0, isTextBox: true });
  bullets(s, ["u_k: mode k, ω_k: its centre frequency", "Hilbert term: makes u_k analytic, one-sided spectrum", "e^(−jω_k t): shifts mode k to baseband", "‖∂t ·‖²: the bandwidth of that shifted mode", "Constraint: the modes must add up to f"], 8.0, 2.4, 4.7, 3.3, 16);
  s.addText("Hyperparameters: number of modes K, bandwidth penalty α, dual step τ.", { x: 0.6, y: 6.0, w: 12, h: 0.5, fontSize: 18, color: MUTED, margin: 0, isTextBox: true });
}
// 4 admm
{
  const s = slide("ADMM turns it into three closed-form updates", { kicker: "Classic VMD", notes: "Standard VMD (Dragomiretskiy and Zosso, 2014). In our benchmark tau is 0, so the multiplier stays at zero." });
  const items = [["Mode update", "admm_u", 1.45], ["Centre frequency", "admm_w", 1.45], ["Dual ascent", "admm_l", 0.8]];
  let yy = 1.7;
  items.forEach((it) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: yy, w: 12.1, h: it[2], rectRadius: 0.1, fill: { color: DARK }, line: { color: DARK, width: 1 } });
    s.addText(it[0], { x: 0.9, y: yy, w: 2.6, h: it[2], fontSize: 18, bold: true, color: ORANGE, valign: "middle", margin: 0, isTextBox: true });
    eqText(s, [it[1]], 3.6, yy + 0.05, 8.9, it[2] - 0.1, { size: 20 });
    yy += it[2] + 0.15;
  });
  bullets(s, ["The mode update is a Wiener-like filter centred on ω_k; α sets how narrow it is", "ω_k is the power-weighted mean frequency of the mode, positive half spectrum", "All updates run in the FFT domain, so the method is global and non-causal"], 0.6, 5.85, 12.1, 1.4, 15);
}
// 5 loop
{
  const s = slide("How the VMD loop runs", { kicker: "Classic VMD", notes: "Settings in the pills are the ones used in this benchmark. The mode sweep is Gauss-Seidel." });
  const st = [["START", "Initialise", "ω_k on a uniform grid, û = 0, λ̂ = 0"], ["STEP 1", "Modes", "Update û_k for k = 1..K, one after the other"], ["STEP 2", "Centres", "ω_k from the mode’s power spectrum"], ["STEP 3", "Multiplier", "Dual ascent on λ̂ with step τ"]];
  st.forEach((c, i) => { card(s, 0.6 + i * 3.1, 1.8, 2.6, 2.1, c[1], c[2], { tag: c[0] }); if (i < 3) arrow(s, 3.2 + i * 3.1, 2.55); });
  s.addText("Repeat until the modes stop changing (tolerance) or the iteration cap is reached. Typically hundreds of FFT sweeps per signal.", { x: 0.6, y: 4.3, w: 12.1, h: 0.8, fontSize: 18, color: TXT, margin: 0, valign: "top", isTextBox: true });
  ["K = 13", "α = 2000", "τ = 0", "tol = 1e-7", "max 499 iterations", "DC mode, mirrored extension"].forEach((t, i) => pill(s, t, 0.6 + [0, 1.6, 3.4, 4.8, 6.5, 8.9][i], 5.5, [1.4, 1.6, 1.2, 1.5, 2.2, 3.1][i]));
}
// 6 unfold idea
{
  const s = slide("Unfolding: an iterative algorithm becomes a network", { dark: true, kicker: "Deep unfolding", notes: "The structure of the algorithm is kept, so the network stays interpretable: every layer is still one step of the original method." });
  card(s, 0.6, 1.9, 5.9, 2.7, "", "", { dark: true, fill: PANEL });
  s.addText("ITERATIVE SOLVER", { x: 0.85, y: 2.05, w: 5.4, h: 0.3, fontSize: 13, bold: true, color: ORANGE, margin: 0, isTextBox: true });
  s.addText("repeat for n = 1, …, 500:", { x: 0.85, y: 2.5, w: 5.4, h: 0.4, fontSize: 17, color: "F4F1EA", margin: 0, isTextBox: true });
  eqText(s, ["unf_iter"], 0.85, 2.95, 5.4, 0.7, { size: 24 });
  s.addText("One operator T, one fixed θ (α, τ), many repetitions until convergence.", { x: 0.85, y: 3.75, w: 5.4, h: 0.7, fontSize: 15, color: MUTEDL, margin: 0, valign: "top", isTextBox: true });
  card(s, 6.8, 1.9, 5.9, 2.7, "", "", { dark: true, fill: PANEL });
  s.addText("UNFOLDED NETWORK", { x: 7.05, y: 2.05, w: 5.4, h: 0.3, fontSize: 13, bold: true, color: LBLUE, margin: 0, isTextBox: true });
  eqText(s, ["unf_1", "unf_2", "unf_n"], 7.05, 2.4, 5.4, 1.5, { size: 20, gap: 6 });
  s.addText("N copies (N small), each with its own learnable θ_n.", { x: 7.05, y: 3.95, w: 5.4, h: 0.5, fontSize: 15, color: MUTEDL, margin: 0, valign: "top", isTextBox: true });
  bullets(s, ["Each iteration becomes a layer; fixed depth N replaces “until convergence”", "Hyperparameters become weights that gradient descent can tune", "Origin: LISTA, which unfolded ISTA for sparse coding (Gregor and LeCun, 2010)"], 0.6, 5.0, 12.1, 1.8, 18);
}
// 7 unfold train
{
  const s = slide("How an unfolded algorithm is trained", { kicker: "Deep unfolding", notes: "Training is end to end: the loss only sees the final layer, but its gradient reaches every layer’s parameters." });
  const c = [["1", "Truncate", "Keep N iterations, N = 1 or 2 here"], ["2", "Parametrise", "Pick hyperparameters to learn, one set per layer"], ["3", "Define a loss", "Score the output of the last layer"], ["4", "Backpropagate", "Gradients flow through all layers; Adam updates the parameters"], ["5", "Fix and deploy", "Freeze the weights, run N layers on new signals"]];
  c.forEach((x, i) => card(s, 0.6 + (i % 3) * 4.1, 1.8 + Math.floor(i / 3) * 2.45, 3.9, 2.2, x[1], x[2], { tag: x[0], tagColor: DORANGE, hs: 22, bs: 16 }));
  card(s, 0.6 + 2 * 4.1, 1.8 + 2.45, 3.9, 2.2, "Speed for guarantees", "Far fewer steps, but no convergence guarantee and tuned to the training data", { dark: true, tag: "TRADE", hs: 22, bs: 16 });
}
// 8 uvmd layer
{
  const s = slide("UVMD: one layer is one VMD sweep", { kicker: "UVMD (Ahmad et al.)", notes: "Source: Ahmad, Wesemann, Waschkowski, Khalid, arXiv 2509.00703, and the released code (OsamaAhmad369/MAGN, uvmd.py). The equation form is the code’s; the paper differs slightly." });
  const st = [["Input f", "Mirror, FFT, keep the positive half: f̂₊", false], ["Layer 1", "α_k, λ̂₁", true], ["Layer N", "α_k, λ̂_N", true], ["Modes u_k", "Inverse FFT; loss on Σ_k û_k", false]];
  st.forEach((c, i) => { card(s, 0.6 + i * 3.1, 1.8, 2.6, 1.5, c[0], c[1], { dark: c[2], fill: DARK, hs: 18, bs: 14 }); if (i < 3) arrow(s, 3.2 + i * 3.1, 2.25); });
  eqCard(s, ["uvmd_u", "uvmd_w"], 0.6, 3.6, 12.1, 2.3, { size: 20, gap: 8 });
  s.addText("Layer n sweeps through the K modes (Gauss-Seidel), then recomputes every ω_k from the new spectra. Mode 0 stays at DC. Written as in the released code.", { x: 0.6, y: 6.1, w: 12.1, h: 1.0, fontSize: 16, color: MUTED, margin: 0, valign: "top", isTextBox: true });
}
// 9 params
{
  const s = slide("What UVMD trains, and what it does not", { kicker: "UVMD (Ahmad et al.)", notes: "The released code allocates N+1 rows for lambda; the last is never used. Alpha is one vector of K values reused in every layer." });
  card(s, 0.6, 1.9, 3.9, 3.5, "α_k, K values", "• Bandwidth penalty per mode, through softplus\n• Shared by all layers\n• Initialised at 2000", { tag: "TRAINED", tagColor: BLUE, line: BLUE, lw: 2, hs: 24, bs: 16 });
  card(s, 4.7, 1.9, 3.9, 3.5, "λ̂_n, one per layer", "• Complex vector of length 2T (T = signal length)\n• Replaces the dual variable\n• Initialised at 0", { tag: "TRAINED", tagColor: BLUE, line: BLUE, lw: 2, hs: 24, bs: 16 });
  card(s, 8.8, 1.9, 3.9, 3.5, "ω_k", "• Recomputed from û_k in every layer\n• Starts on a fixed uniform grid, 0.5·k/K\n• No weight of its own", { dark: true, tag: "NOT TRAINED", hs: 24, bs: 16 });
  s.addText([{ text: "Both trained parameters enter only the ", options: {} }, { text: "mode update", options: { bold: true } }, { text: ". The frequency update has none, so ω can change only through û_k.", options: {} }], { x: 0.6, y: 5.8, w: 12.1, h: 0.9, fontSize: 19, color: TXT, margin: 0, valign: "top", isTextBox: true });
}
// 10 training
{
  const s = slide("How UVMD is trained", { kicker: "UVMD (Ahmad et al.)", notes: "We use the upstream training loop unchanged. Upstream saves the best weights and then overwrites them with the last-epoch weights; the training loss was still falling at epoch 100." });
  eqCard(s, ["loss"], 0.6, 1.75, 7.2, 1.0, { size: 24 });
  s.addText("Spectral reconstruction loss (L1 in the code, L2 in the paper). No ground-truth modes: the loss sees only the sum.", { x: 8.1, y: 1.75, w: 4.6, h: 1.0, fontSize: 15, color: MUTED, margin: 0, valign: "middle", isTextBox: true });
  table(s, [["Item", "Setting"], ["Optimiser", "Adam, learning rate 0.1, batch size 1"], ["Schedule", "Up to 100 epochs, early stopping with patience 10"], ["Data split", "70 / 15 / 15, random_state 42"], ["Model size", "K = 13 modes, N = 1 or 2 layers"], ["Initial values", "α_k = 2000, λ̂ = 0, ω on the uniform grid"]], 0.6, 3.1, 12.1, [2.6, 9.5], { size: 17, rowH: 0.52 });
  s.addText("The learned α and λ̂ are shared by every signal of the dataset; they are not computed per signal.", { x: 0.6, y: 6.5, w: 12.1, h: 0.5, fontSize: 16, color: MUTED, margin: 0, isTextBox: true });
}
// 11 paper vs code
{
  const s = slide("The paper and the released code differ", { kicker: "UVMD (Ahmad et al.)", notes: "We use the released code unchanged as the reference implementation." });
  s.addText("PAPER", { x: 0.6, y: 1.6, w: 5.9, h: 0.3, fontSize: 13, bold: true, color: DORANGE, margin: 0, isTextBox: true });
  eqCard(s, ["paper_u"], 0.6, 1.95, 5.9, 1.6, { size: 17 });
  s.addText("RELEASED CODE", { x: 6.8, y: 1.6, w: 5.9, h: 0.3, fontSize: 13, bold: true, color: DORANGE, margin: 0, isTextBox: true });
  eqCard(s, ["uvmd_u"], 6.8, 1.95, 5.9, 1.6, { size: 17 });
  table(s, [["Item", "Paper", "Released code"], ["Multiplier term and factor 2", "+ λ̂/2 and 2·softplus(α)", "− λ̂/2 and softplus(α)"], ["Loss", "L2 norm", "L1, sum of absolute values"], ["Problem statement", "No constraint, no λ update given", "λ̂ is a free trained tensor"]], 0.6, 3.8, 12.1, [3.7, 4.2, 4.2], { size: 16, rowH: 0.45 });
  bullets(s, ["The sign of λ̂ is absorbed by training, but the factor 2 changes what α means: the same number gives a different filter width", "The headline “250× faster” is about 162× from the paper’s own timings", "“Robust” is in the title, but no noise experiments are reported", "The upstream repository carries no licence"], 0.6, 5.7, 12.1, 1.6, 14);
}
// 12 caveat omega
{
  const s = slide("The trained parameters never act on ω", { dark: true, kicker: "Caveat 1", notes: "Our measurements on the next slides show this holds on traffic and synthetic data, but not on CWRU." });
  card(s, 0.6, 1.9, 3.5, 1.5, "α_k, λ̂", "The only weights", { dark: true, fill: PANEL, hs: 22, bs: 16 }); arrow(s, 4.2, 2.35);
  card(s, 4.8, 1.9, 3.5, 1.5, "û_k update", "Filter width and offset", { dark: true, fill: PANEL, hs: 22, bs: 16 }); arrow(s, 8.4, 2.35);
  card(s, 9.0, 1.9, 3.7, 1.5, "ω_k update", "Mean frequency of û_k, no weight", { dark: true, fill: PANEL, hs: 22, bs: 16 });
  bullets(s, ["ω can move only indirectly, through the spectrum that α and λ̂ shape", "With N = 1 or 2 it is updated once or twice; classic VMD updates it hundreds of times", "It starts on a uniform grid that need not match the signal’s true frequencies", "Nothing in the loss pushes ω toward the right place"], 0.6, 3.9, 12.1, 3.0, 19);
}
// 13 caveat loss
{
  const s = slide("The loss rewards reconstruction only", { kicker: "Caveat 2", notes: "The CWRU result is the clearest case: near-perfect reconstruction with modes that overlap far more than the untrained ones." });
  bullets(s, ["λ̂ is a free complex vector of length 2T: it can absorb the residual and cut the error without making better modes", "α can collapse: with α → 0 a mode passes everything and the sum is exact", "No term asks for orthogonal or narrow-band modes", "One α and λ̂ for the whole dataset; changing T means retraining"], 0.6, 1.9, 6.2, 4.8, 19);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.2, y: 1.9, w: 5.5, h: 4.4, rectRadius: 0.1, fill: { color: DARK }, line: { color: DARK, width: 1 } });
  s.addText([{ text: "SEEN ON CWRU, 1 LAYER", options: { fontSize: 13, bold: true, color: ORANGE, breakLine: true } }, { text: "Reconstruction error", options: { fontSize: 18, color: "F4F1EA", breakLine: true } }, { text: "0.0662 → 0.0004", options: { fontSize: 28, bold: true, color: LBLUE, breakLine: true } }, { text: "Spectral overlap", options: { fontSize: 18, color: "F4F1EA", breakLine: true } }, { text: "0.066 → 0.371", options: { fontSize: 28, bold: true, color: ORANGE, breakLine: true } }, { text: "Untrained VMD-1 to trained UVMD-1. One α went to 0.", options: { fontSize: 14, color: MUTEDL } }], { x: 7.5, y: 2.1, w: 4.9, h: 4.0, margin: 0, valign: "middle", paraSpaceAfter: 10, isTextBox: true });
}
// 14 caveat other
{
  const s = slide("Other limits to keep in mind", { kicker: "Further caveats" });
  const c = [["Far from the fixed point", "1 or 2 sweeps cannot reach what 500 reach, so the output is not a VMD solution"], ["Non-causal", "A global FFT sees the whole window, so forecasting use risks leakage from the future"], ["Fixed window length", "λ̂ has length 2T, so a different T needs new training"], ["Batch size 1", "One signal per step, so a GPU helps little"], ["Training bookkeeping", "Best weights are saved, then overwritten by the last epoch; loss still falling at epoch 100"], ["Not tuned for forecasting", "The paper reports that more layers overfit"]];
  c.forEach((x, i) => card(s, 0.6 + (i % 3) * 4.1, 1.9 + Math.floor(i / 3) * 2.5, 3.9, 2.3, x[0], x[1], { hs: 20, bs: 16 }));
}
// 15 omega evidence
{
  const s = slide("Does ω really barely move? Partly", { notes: "Shift is in grid spacings of 0.5/K cycles per sample, averaged over modes 1 to K-1 and over test signals. Converged VMD is computed on 10 signals." });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.6, w: 6.6, h: 4.2, fill: { color: WHITE }, line: { color: BORDER, width: 1 } });
  s.addImage({ path: IMG.omega, ...fit(IMG.omega, 6.5, 4.1, 0.65, 1.65) });
  table(s, [["Mean shift", "VMD", "UVMD-1", "UVMD-2"], ["Synthetic", "0.46", "0.69", "0.46"], ["CWRU", "0.62", "1.86", "2.05"], ["Traffic", "1.61", "0.38", "0.28"]], 7.6, 1.6, 5.1, [1.8, 1.0, 1.15, 1.15], { size: 16, rowH: 0.5, right: true });
  s.addText("Grid spacings from the start grid; VMD is the converged run.", { x: 7.6, y: 3.75, w: 5.1, h: 0.5, fontSize: 13, color: MUTED, margin: 0, isTextBox: true });
  bullets(s, ["Traffic and synthetic: UVMD moves ω little, as predicted", "CWRU: UVMD moves it 3× more than converged VMD", "Moving more is not moving right: synthetic centre-frequency error is 0.0025 for UVMD-1 against 0.0011 for VMD"], 0.6, 6.0, 12.1, 1.3, 16);
}
// 16 setup
{
  const s = slide("Experimental setup", { kicker: "Results", notes: "The traffic run uses 100 training, 20 validation and 30 test nodes; the full protocol has not been run." });
  card(s, 0.6, 1.9, 3.9, 1.7, "Synthetic", "4 known modes, ground truth available, 30 test signals", { hs: 22, bs: 16 });
  card(s, 4.7, 1.9, 3.9, 1.7, "CWRU bearing", "Vibration with fault lines, 300 test signals", { hs: 22, bs: 16 });
  card(s, 8.8, 1.9, 3.9, 1.7, "LargeST traffic", "San Diego sensors, bounded run, 30 test windows", { hs: 22, bs: 16 });
  s.addText([{ text: "Methods, all at K = 13: ", options: { bold: true } }, { text: "converged VMD; VMD-N (VMD stopped after N iterations); trained UVMD-N with N = 1, 2.", options: { breakLine: true } },
    { text: "VMD-N equals UVMD-N before training, so the pair isolates the effect of training.", options: { breakLine: true } },
    { text: "Reconstruction: ", options: { bold: true } }, { text: "relative error ‖x − Σu_k‖ / ‖x‖.", options: { breakLine: true } },
    { text: "Separation, without ground truth: ", options: { bold: true } }, { text: "orthogonality index, spectral overlap, relative bandwidth, energy ratio, with the effective number of modes as a guard." }],
    { x: 0.6, y: 4.0, w: 12.1, h: 2.9, fontSize: 18, color: TXT, margin: 0, valign: "top", paraSpaceAfter: 10, isTextBox: true });
}
// 17 recon table
{
  const s = slide("Reconstruction: trained UVMD wins", { kicker: "Results", notes: "Mean over test signals. Bold marks the best entry of each row (tie on CWRU)." });
  table(s, [["Relative error ↓", "VMD converged", "VMD-1", "UVMD-1", "VMD-2", "UVMD-2"], ["Synthetic", "0.0294", "0.0499", "0.0066", "0.0379", "0.0142"], ["CWRU", "0.0760", "0.0662", "0.0004", "0.1070", "0.0004"], ["LargeST traffic", "0.0303", "0.0729", "0.0205", "0.0838", "0.0372"]], 0.6, 1.8, 12.1, [2.7, 2.0, 1.85, 1.85, 1.85, 1.85], { size: 18, rowH: 0.6, right: true, boldCells: ["1,3", "2,3", "2,5", "3,3"] });
  bullets(s, ["Trained UVMD-N beats untrained VMD-N on 100% of signals in every dataset and for both depths", "One trained layer also beats converged VMD on all three datasets", "Exception: on traffic, UVMD-2 (0.0372) is slightly worse than converged VMD (0.0303)"], 0.6, 4.6, 12.1, 2.2, 19);
}
// 18 recon effect
{
  const s = slide("How much training cuts the error", { dark: true, kicker: "Results", notes: "Ratios of means from Table A2. CWRU 2 layers is 0.00 in the table, below rounding precision." });
  s.addText("Trained UVMD-N error divided by untrained VMD-N error, same depth. Lower is better.", { x: 0.6, y: 1.7, w: 12.1, h: 0.5, fontSize: 18, color: MUTEDL, margin: 0, isTextBox: true });
  [["Synthetic", "0.13×", "0.38×"], ["CWRU", "0.01×", "<0.01×"], ["LargeST traffic", "0.28×", "0.44×"]].forEach((d, i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 2.4, 3.9, 3.3, "", "", { dark: true, fill: PANEL });
    s.addText([{ text: d[0], options: { fontSize: 22, bold: true, color: LBLUE, breakLine: true } }, { text: d[1], options: { fontSize: 44, bold: true, color: "F4F1EA", breakLine: true } }, { text: "1 layer", options: { fontSize: 15, color: MUTEDL, breakLine: true } }, { text: d[2], options: { fontSize: 44, bold: true, color: "F4F1EA", breakLine: true } }, { text: "2 layers", options: { fontSize: 15, color: MUTEDL } }], { x: x + 0.3, y: 2.55, w: 3.3, h: 3.0, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("CWRU shows the biggest drop. But there, one α was pushed to 0, which lets the model reach a very low error without really separating the modes. So a low error alone does not prove good separation. Check the separation results next.", { x: 0.6, y: 6.0, w: 12.1, h: 0.9, fontSize: 18, color: "D5DAE6", margin: 0, valign: "top", isTextBox: true });
}
// 19 sep criteria
{
  const s = slide("How separation is measured without ground truth", { kicker: "Results", notes: "Spearman correlation against grouped mode SNR over 150 points (30 signals by 5 methods). The energy ratio showed no relation (-0.14)." });
  table(s, [["Criterion", "What it measures", "Better"], ["Orthogonality index", "Content shared between modes (formula below)", "lower"], ["Spectral overlap", "Share of spectrum two modes occupy together", "lower"], ["Relative bandwidth", "Energy-weighted spectral width of a mode ÷ Nyquist", "lower"], ["Energy ratio", "Above 1 the modes double count, below 1 energy is lost", "closer to 1"], ["Effective modes", "s_k: energy share of mode k; a guard, not a score", "not near 1"]], 0.6, 1.7, 12.1, [2.9, 7.4, 1.8], { size: 16, rowH: 0.5 });
  eqCard(s, ["ortho"], 0.6, 4.85, 4.9, 1.2, { size: 16 });
  eqCard(s, ["energy"], 5.7, 4.85, 3.9, 1.2, { size: 16 });
  eqCard(s, ["neff"], 9.8, 4.85, 2.9, 1.2, { size: 16 });
  s.addText("On the synthetic data, where true modes are known, relative bandwidth tracks true separation best (Spearman −0.85), orthogonality and overlap about −0.5. Reconstruction error points the wrong way (+0.52).", { x: 0.6, y: 6.2, w: 12.1, h: 1.0, fontSize: 15, color: TXT, margin: 0, valign: "top", isTextBox: true });
}
// 20 sep results
{
  const s = slide("Separation: training does not help", { kicker: "Results", notes: "Lower is better for orthogonality, overlap and bandwidth. Ratios are trained UVMD-1 over untrained VMD-1 from Table A2." });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.7, w: 7.6, h: 5.1, fill: { color: WHITE }, line: { color: BORDER, width: 1 } });
  s.addImage({ path: IMG.bars, ...fit(IMG.bars, 7.5, 5.0, 0.65, 1.75) });
  bullets(s, ["Converged VMD is best on orthogonality, overlap and bandwidth in all three datasets", "Trained UVMD-1 never beats VMD-1 on bandwidth (0% of signals): 1.65× synthetic, 2.59× CWRU, 1.08× traffic", "Overlap is worse after training in five of six dataset and depth pairs, level in the sixth", "Energy ratio is the one criterion that improves"], 8.5, 1.7, 4.2, 5.1, 16);
}
// 21 sep truth
{
  const s = slide("Synthetic data: separation against the true modes", { kicker: "Results", notes: "Grouped mode SNR: each estimated mode is assigned to the true mode it projects onto most, and the assigned modes are summed. Frequency error is in cycles per sample." });
  table(s, [["Method", "Mode SNR (dB) ↑", "Freq. error ↓"], ["VMD converged", "20.7", "0.00106"], ["VMD-1", "12.4", "0.00120"], ["UVMD-1", "8.5", "0.00252"], ["VMD-2", "18.8", "0.00090"], ["UVMD-2", "7.3", "0.00207"]], 0.6, 1.7, 5.4, [2.0, 1.8, 1.6], { size: 16, rowH: 0.5, right: true, boldCells: ["1,1", "4,2"] });
  s.addText("Training lowers the true mode SNR by 4 to 12 dB while reconstruction error falls by a factor of 2.7 to 7.6. Centre frequencies get farther from the truth.", { x: 0.6, y: 5.0, w: 5.4, h: 1.8, fontSize: 16, color: TXT, margin: 0, valign: "top", isTextBox: true });
  s.addShape(pres.shapes.RECTANGLE, { x: 6.3, y: 1.7, w: 6.4, h: 4.6, fill: { color: WHITE }, line: { color: BORDER, width: 1 } });
  s.addImage({ path: IMG.truth, ...fit(IMG.truth, 6.3, 4.5, 6.35, 1.75) });
}
// 22 conclusions
{
  const s = slide("Conclusions and next steps", { dark: true, notes: "The next-step ideas are proposals, not results." });
  card(s, 0.6, 1.8, 5.95, 4.4, "", "", { dark: true, fill: PANEL });
  s.addText("WHAT WE FOUND", { x: 0.9, y: 1.95, w: 5.4, h: 0.35, fontSize: 13, bold: true, color: ORANGE, margin: 0, isTextBox: true });
  bullets(s, ["UVMD reconstructs better than VMD at the same depth on every signal, with 1 or 2 layers instead of hundreds of iterations", "It does not separate better: at best on par with untrained VMD-N, below converged VMD", "Reconstruction error is no evidence of separation", "ω barely moves on traffic and synthetic, but moves a lot on CWRU"], 0.9, 2.4, 5.4, 3.7, 16);
  card(s, 6.75, 1.8, 5.95, 4.4, "", "", { dark: true, fill: PANEL });
  s.addText("NEXT STEPS", { x: 7.05, y: 1.95, w: 5.4, h: 0.35, fontSize: 13, bold: true, color: LBLUE, margin: 0, isTextBox: true });
  bullets(s, ["Add a separation term (bandwidth or orthogonality) to the loss", "Give ω a trainable path: learned initial grid or a filter on the spectrum", "Ablate: train α only, then λ̂ only", "Test on forecasting with causal windows", "Run the full traffic protocol"], 7.05, 2.4, 5.4, 3.7, 16);
  s.addText("Traffic results come from a bounded run (100 / 20 / 30 nodes). Code, tables and figures: repository vmd-uvmd-comparison.", { x: 0.6, y: 6.5, w: 12.1, h: 0.5, fontSize: 14, color: "8FA0BC", margin: 0, isTextBox: true });
}

// fit(): contain an image of known size in a box
function fit(file, bw, bh, bx, by) {
  const { imageSize } = (() => { try { return require("image-size"); } catch (e) { return {}; } })();
  let iw = 1600, ih = 1000;
  try {
    const buf = require("fs").readFileSync(file);
    iw = buf.readUInt32BE(16); ih = buf.readUInt32BE(20);
  } catch (e) {}
  const r = Math.min(bw / iw, bh / ih);
  const w = iw * r, h = ih * r;
  return { x: bx + (bw - w) / 2, y: by + (bh - h) / 2, w, h };
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  await injectEquations(OUT);
  console.log("wrote", OUT);
})();
