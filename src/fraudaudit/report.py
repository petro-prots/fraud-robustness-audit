"""Gate report: ``out/audit.json`` (every run, every check, the verdict) and the Ukrainian Markdown summary
for ``$GITHUB_STEP_SUMMARY`` / ``out/summary.md``."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .audit import _num, _pct, strategy_dict, to_json
from .gate import Check, GateConfig, parse_run_key, passed

VERDICT_OK, VERDICT_FAIL = "ПРОЙДЕНО", "НЕ ПРОЙДЕНО"

_TITLE = {
    "ver_uplift": "приріст VER під атакою – верхня межа {conf} % інтервалу VER_adv − VER_clean, "
                  "найгірша атака × seed",
    "recall_clean": "recall на чистих операціях test при τ",
    "feasibility": "частка виконаних змагальних операцій у межах моделі загроз (мінімум за прогонами)",
    "skew": "розбіжності рішень навчального і продуктивного шляхів ознак",
}


def _value(c: Check) -> str:
    if c.value is None:
        return "–"
    if c.name in ("ver_uplift", "feasibility"):
        return _pct(c.value)
    if c.name == "recall_clean":
        return _num(c.value, 4)
    return str(int(c.value))


def _threshold(c: Check) -> str:
    if c.threshold is None:
        return "не задано (δ = null)"
    return {"ver_uplift": f"≤ {_pct(c.threshold)}", "recall_clean": f"≥ {_num(c.threshold, 2)}",
            "feasibility": "= 100,0 %", "skew": "= 0 (і всі ознаки в допуску)"}[c.name]


def _ci(t) -> str:
    return f"{_pct(t[0])} [{_pct(t[1])}; {_pct(t[2])}]"


def summary_markdown(checks: list[Check], results: dict, meta: dict, merchants=None) -> str:
    """Ukrainian Markdown: verdict, checks, recommendations, attacks × seeds, baseline, clean quality,
    the worst run's strategy, metadata. ``results`` is ``{run_key: AuditResult}``; ``meta`` holds the
    gate-level metadata (``gate``, ``gate_file``, ``wall_seconds``, …); per-run metadata comes from the runs."""
    ok = passed(checks)
    by = {c.name: c for c in checks}
    up = by["ver_uplift"]
    worst_key = up.worst
    w = results[worst_key]
    wm = w.meta
    conf = int(round(100 * wm["confidence"]))
    med = float(wm["median_amount"])
    w_attack, w_seed = parse_run_key(worst_key)
    delta = up.threshold
    failed = [c for c in checks if not c.passed]

    lines = [f"# {'✅ ' + VERDICT_OK if ok else '❌ ' + VERDICT_FAIL} – шлюз стійкості до епізодних атак", ""]
    if ok:
        lines.append("Усі перевірки шлюзу пройдено.")
    else:
        lines.append("Не пройдено: " + ", ".join(f"`{c.name}`" for c in failed) + ". Що робити – у розділі "
                     "«Рекомендації».")
    lines += ["",
              f"Найгірший випадок: атака `{w_attack}`, seed атаки {w_seed}; верхня межа {conf} % інтервалу "
              f"приросту VER_adv − VER_clean {_pct(up.value)} при "
              + (f"δ = {_pct(delta)}." if delta is not None else "незаданому δ."),
              ""]
    if delta is not None:
        lines.append(f"У доларах (частка × медіанна сума епізоду A = {_num(med, 2)} $): атака додає шахраю до "
                     f"~{_num(up.value * med, 0)} $ на скомпрометовану картку при допустимих "
                     f"~{_num(delta * med, 0)} $.")
    else:
        lines.append(f"У доларах (частка × медіанна сума епізоду A = {_num(med, 2)} $): атака додає шахраю до "
                     f"~{_num(up.value * med, 0)} $ на скомпрометовану картку; допустиму величину не задано.")
    lines += ["", "## Перевірки", "", "| перевірка | значення | поріг | результат |", "|---|---:|---|---|"]
    for c in checks:
        title = _TITLE[c.name].format(conf=conf)
        show = c.name == "ver_uplift" or (not c.passed and c.name == "feasibility")
        where = f" (прогін `{c.worst}`)" if show and c.worst and len(results) > 1 else ""
        lines.append(f"| `{c.name}` – {title}{where} | {_value(c)} | {_threshold(c)} "
                     f"| {'✅ так' if c.passed else '❌ ні'} |")
    if "skew" not in by:
        lines += ["", "Звіт звірки шляхів ознак (`--skew-report`) не передано: перевірку `skew` не виконано."]

    if failed:
        lines += ["", "## Рекомендації", ""]
        for c in failed:
            lines.append(f"- `{c.name}`: {c.recommendation}")

    lines += ["", "## Атаки × seed", "",
              f"Розподіл карток на цільові й запасні однаковий для всіх прогонів (seed {wm['seed']}); змінюється "
              "лише випадковий потік атаки. Шлюз бере найгірший прогін.", "",
              f"| атака | seed атаки | VER_adv [{conf} %] | приріст [{conf} %] | ASR | операцій розвідки "
              "| оцінок стратегій | |",
              "|---|---:|---|---|---|---:|---:|---|"]
    for key, r in results.items():
        a, s = parse_run_key(key)
        mark = "**⚠ найгірший**" if key == worst_key else ""
        lines.append(f"| {a} | {s} | {_ci(r.ver_adv)} | {_ci(r.uplift)} | {_pct(r.asr)} "
                     f"({r.meta['n_success']} з {r.n_targets}) | {r.recon_ops} | {r.meta['recon']['evals']} | {mark} |")

    lines += ["", "## Базова лінія і припущення", "",
              f"- VER_clean (повтор реальних шахрайських операцій цільових епізодів): {_ci(w.ver_clean)}; "
              f"без атаки поріг ρ·A досягли {wm['n_clean_success']} з {w.n_targets} епізодів.",
              "- Обидва прогони (VER_clean і VER_adv) стартують зі стану картки до початку епізоду, а легітимні "
              "операції картки всередині вікна епізоду не відтворюються, тож VER_clean – це повтор лише "
              "шахрайських операцій, а не історичний результат банку.",
              f"- VER > 1 (через нижню межу суми операції) у найгіршому прогоні: змагальних епізодів "
              f"{wm['n_ver_adv_gt_1']} (макс. {_num(wm['max_ver_adv'])}), реальних {wm['n_ver_clean_gt_1']} "
              f"(макс. {_num(wm['max_ver_clean'])}); значення не обрізаються.",
              f"- Цільових епізодів: {w.n_targets} (карток: {wm['n_target_cards']}); запасних карток для розвідки: "
              f"{len(w.spare_cards)}; бюджет розвідки {wm['recon_budget_ops']} операцій на прогін.",
              f"- Інтервали: percentile bootstrap за епізодами ({wm['n_boot']} повторів, приріст – парний); межа "
              f"ASR – Clopper–Pearson."]

    gate = meta.get("gate") or {}
    lines += ["", "## Якість моделі на чистих даних (при τ)", ""]
    if "target_fpr" in gate:
        lines += [f"Цільовий FPR на операцію (`gate.yaml`): {_pct(gate['target_fpr'])}; τ = {wm['model']['tau']}.", ""]
    lines += ["| період | recall | FPR | precision | операцій | шахрайських |", "|---|---:|---:|---:|---:|---:|"]
    for p in ("val", "test"):
        q = w.clean_quality[p]
        lines.append(f"| {p} | {_num(q['recall'], 4)} | {_num(q['fpr'], 4)} | {_num(q['precision'], 4)} "
                     f"| {q['n']} | {q['n_fraud']} |")

    b = strategy_dict(w.best, merchants)
    lines += ["", f"## Стратегія найгіршого прогону (`{worst_key}`)", "",
              f"Операцій k = {b['k']}, сумарно ≈ {_pct(b['u'])} від A; план у порядку запланованого часу (день від "
              "опівночі дня початку епізоду), до перенесення у вікно і розсування операцій:", "",
              "| № | частка A | день | година | категорія |", "|---:|---:|---:|---:|---|"]
    for i, op in enumerate(b["ops"], 1):
        lines.append(f"| {i} | {_pct(op['share_of_A'])} | {op['day_index']} | {op['time_of_day']} "
                     f"| {op.get('category', '–')} |")

    run_walls = ", ".join(f"{k} {r.meta['wall_seconds']} с" for k, r in results.items())
    lines += ["", "## Метадані", "",
              f"- seed розподілу {wm['seed']}; seed атак: "
              + ", ".join(str(s) for s in sorted({parse_run_key(k)[1] for k in results}))
              + f"; git `{wm['git_sha']}` (незакомічені зміни: {wm['git_dirty']})",
              f"- платформа: {wm['platform']['platform']}; процесів: {wm['n_jobs']}"
              + (f"; час шлюзу {meta['wall_seconds']} с" if "wall_seconds" in meta else "")
              + (f" (з завантаженням даних і моделі {meta['wall_seconds_with_loading']} с)"
                 if "wall_seconds_with_loading" in meta else ""),
              f"- час прогонів: {run_walls}",
              f"- модель: `{wm['model']['dir']}`, τ = {wm['model']['tau']}, SHA-256 `model.json`: "
              f"`{wm['model']['model_json_sha256']}`",
              f"- модель загроз: `{wm['threat_model']['file']}`, SHA-256 `{wm['threat_model']['sha256']}`"]
    if meta.get("gate_file"):
        lines.append(f"- шлюз: `{meta['gate_file']}`, SHA-256 `{meta.get('gate_sha256')}`")
    if gate:
        lines.append("- параметри шлюзу: " + ", ".join(f"{k} = {v}" for k, v in gate.items()))
    lines += ["- версії: " + ", ".join(f"{k} {v}" for k, v in wm["versions"].items()), ""]
    return "\n".join(lines)


def write_gate_outputs(out_dir: Path | str, cfg: GateConfig, checks: list[Check], results: dict, meta: dict,
                       merchants=None, skew: dict | None = None) -> dict:
    """``out_dir/audit.json`` (verdict, checks, gate, every run via ``to_json``, skew) and ``out_dir/summary.md``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ok = passed(checks)
    meta = dict(meta, gate=dataclasses.asdict(cfg))
    worst = next(c.worst for c in checks if c.name == "ver_uplift")
    data = {"verdict": VERDICT_OK if ok else VERDICT_FAIL, "passed": ok,
            "checks": [dataclasses.asdict(c) for c in checks], "gate": dataclasses.asdict(cfg),
            "worst_run": worst, "runs": {k: to_json(r, merchants) for k, r in results.items()},
            "skew": skew, "meta": meta}
    (out / "audit.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(summary_markdown(checks, results, meta, merchants), encoding="utf-8")
    return data
