"""Gate report: ``out/audit.json`` (every run, every check, the verdict) and the Ukrainian Markdown summary
for ``$GITHUB_STEP_SUMMARY`` / ``out/summary.md``."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from .audit import BaselineResult, _num, _pct, baseline_to_json, strategy_dict, to_json
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
    if c.name == "feasibility" and c.value != 1.0:
        return _pct_exact(c.value)
    if c.name in ("ver_uplift", "feasibility"):
        return _pct(c.value)
    if c.name == "recall_clean":
        return _num(c.value, 4)
    return str(int(c.value))


def _pct_exact(x: float) -> str:
    """Percent with as many decimals as needed so that a value below 1 never prints as 100 %."""
    if not math.isfinite(x):
        return f"{x} %"
    for d in range(1, 13):
        s = f"{100 * x:.{d}f}"
        if float(s) != 100.0:
            break
    return f"{s} %".replace(".", ",")


def _threshold(c: Check) -> str:
    if c.threshold is None:
        return "не задано (δ = null)"
    return {"ver_uplift": f"≤ {_pct(c.threshold)}", "recall_clean": f"≥ {_num(c.threshold, 2)}",
            "feasibility": "= 100,0 %", "skew": "= 0 (і всі ознаки в допуску)"}[c.name]


def _ci(t) -> str:
    return f"{_pct(t[0])} [{_pct(t[1])}; {_pct(t[2])}]"


def baseline_interpretation(b: BaselineResult, worst) -> str:
    """One plain Ukrainian sentence: does the conventional audit over- or under-state the risk?

    The conventional audit's estimate is ``b.evaded_amount_share`` (share of the fraud amount whose
    operations evade); the episode audit's is VER_adv of the worst run ``worst`` (an ``AuditResult``), the
    same scale (share of A). Above the upper bound of VER_adv's interval: overstates; below the lower
    bound: understates; inside: a close overall estimate. The replay VER says what the operations the
    conventional audit found actually withdraw as an episode."""
    claim, va, wu = b.evaded_amount_share, worst.ver_adv, worst.uplift
    rep, ru = b.replay_ver[0], b.replay_uplift[0]
    tail = (f"ті самі операції, виконані як епізод з перерахунком агрегатів і реакцією банку, виводять "
            f"{_pct(rep)} суми епізоду (приріст до VER_clean {_pct(ru)})")
    episode = f"епізодна атака виводить {_pct(va[0])} (приріст {_pct(wu[0])})"
    if claim > va[2]:
        return (f"Звичайний аудит завищує ризик: за ним ухиляються операції на {_pct(claim)} суми шахрайства, "
                f"але {tail}, а {episode}.")
    if claim < va[1]:
        return (f"Звичайний аудит занижує ризик: за ним ухиляються операції лише на {_pct(claim)} суми "
                f"шахрайства, тоді як {episode}, а {tail}.")
    return (f"Звичайний аудит дає близьку оцінку загального ризику ({_pct(claim)} суми проти VER_adv "
            f"{_pct(va[0])} епізодної атаки), але {tail}, тож приріст від атаки він не вимірює.")


def _baseline_section(b: BaselineResult, worst_key: str, worst, conf: int) -> list[str]:
    tms = b.meta["single_threat_model"]["params"]
    ep = b.meta["episode_params"]
    return [
        "", "## Звичайний аудит (атака на одну транзакцію) проти епізодного", "",
        "Інформаційне порівняння: не впливає на вердикт шлюзу.", "",
        f"| показник | значення | {conf} % інтервал |", "|---|---:|---|",
        f"| Частка ухилених операцій (звичайний аудит) | {_pct(b.evaded_tx_share)} ({b.n_evaded} з {b.n_ops}) | – |",
        f"| Частка ухиленої суми (суми знайдених операцій / сума A) | {_pct(b.evaded_amount_share)} | – |",
        f"| Те саме за оригінальними сумами ухилених операцій | {_pct(b.evaded_orig_amount_share)} | – |",
        f"| VER повтору знайдених операцій як епізоду | {_pct(b.replay_ver[0])} "
        f"| [{_pct(b.replay_ver[1])}; {_pct(b.replay_ver[2])}] |",
        f"| Приріст повтору до VER_clean (парний bootstrap) | {_pct(b.replay_uplift[0])} "
        f"| [{_pct(b.replay_uplift[1])}; {_pct(b.replay_uplift[2])}] |",
        f"| VER_adv епізодної атаки (найгірший прогін `{worst_key}`) | {_pct(worst.ver_adv[0])} "
        f"| [{_pct(worst.ver_adv[1])}; {_pct(worst.ver_adv[2])}] |",
        f"| Приріст епізодної атаки (найгірший прогін `{worst_key}`) | {_pct(worst.uplift[0])} "
        f"| [{_pct(worst.uplift[1])}; {_pct(worst.uplift[2])}] |",
        "",
        baseline_interpretation(b, worst),
        "",
        f"- Звичайний аудит: кожна реальна шахрайська операція цільових епізодів атакується окремо випадковим "
        f"пошуком ({tms['single_tx_budget']} запитів на операцію, лише рішення моделі): сума "
        f"{_num(tms['amount_min'], 2)}–{_num(tms['amount_max'], 2)} $, зсув часу до {tms['max_shift_days']} днів "
        f"(з перенесенням у {ep['window_days']}-денне вікно епізоду цілими добами), мерчант. Стан картки – "
        "оригінальний потік (історія до епізоду і реальні попередні операції епізоду), агрегати не "
        "перераховуються, реакції банку немає; береться перший кандидат, якого модель не відхиляє, а якщо "
        "такого немає – оригінальна операція (не ухилена).",
        f"- Повтор: знайдені операції кожного епізоду впорядковуються за часом (не ближче ніж "
        f"{ep['min_gap_secs']} с одна до одної) і виконуються як епізод з перерахунком агрегатів, "
        f"`max_declines` = {ep['max_declines']} і ρ = {_num(ep['rho'], 2)}; поріг ρ·A досягли "
        f"{b.meta['n_replay_success']} з {b.n_targets} епізодів, заблоковано карток: {b.meta['n_replay_blocked']}.",
        f"- Ті самі {b.n_targets} цільових епізодів, що й в епізодному аудиті (seed розподілу {b.meta['seed']}); "
        f"запитів до моделі {b.meta['queries']}; час {b.meta['wall_seconds']} с.",
    ]


def summary_markdown(checks: list[Check], results: dict, meta: dict, merchants=None,
                     baseline: BaselineResult | None = None) -> str:
    """Ukrainian Markdown: verdict, checks, recommendations, attacks × seeds, baseline, the conventional
    (single-transaction) audit comparison when ``baseline`` is given (informational), clean quality,
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
    per_card = int(wm["n_target_cards"]) == int(w.n_targets)
    unit = ("на одну скомпрометовану картку (один цільовий епізод на картку)" if per_card else "на епізод")
    head = f"У доларах (орієнтовно: частка × медіанна сума епізоду A = {_num(med, 2)} $): "
    if up.value is None or not math.isfinite(up.value):
        lines.append(head + "не обчислено – приріст VER у найгіршому прогоні не скінченний.")
    elif delta is not None:
        lines.append(head + f"атака додає шахраю до ~{_num(up.value * med, 0)} $ {unit} при допустимих "
                     f"~{_num(delta * med, 0)} $.")
    else:
        lines.append(head + f"атака додає шахраю до ~{_num(up.value * med, 0)} $ {unit}; допустиму величину "
                     "не задано.")
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

    if baseline is not None:
        lines += _baseline_section(baseline, worst_key, w, conf)

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
    if "model_target_fpr" in meta:
        mt = meta["model_target_fpr"]
        lines.append("- target_fpr моделі не вказано — звірку пропущено" if mt is None
                     else f"- target_fpr моделі (`model.json`): {mt}, збігається з `gate.yaml`")
    if gate:
        lines.append("- параметри шлюзу: " + ", ".join(f"{k} = {v}" for k, v in gate.items()))
    lines += ["- версії: " + ", ".join(f"{k} {v}" for k, v in wm["versions"].items()), ""]
    return "\n".join(lines)


def error_summary(message: str) -> str:
    """Minimal summary when the audit could not run (configuration or artifact error): CI must show why."""
    return (f"# ❌ {VERDICT_FAIL} — помилка конфігурації/артефактів\n\n"
            "Аудит не виконано, тож шлюз закритий (exit 2). Причина:\n\n"
            f"```\n{message}\n```\n\n"
            "Виправте конфігурацію (`gate.yaml`, модель загроз, звіт skew) або артефакти моделі (`models/`, "
            "перевірка SHA-256 у `manifest.json`) і повторіть аудит; не обходьте перевірку цілісності.\n")


def write_error_outputs(out_dir: Path | str, message: str) -> str:
    """``out_dir/audit.json`` (verdict, error) and ``out_dir/summary.md`` for a run that could not start."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = error_summary(message)
    (out / "audit.json").write_text(json.dumps({"verdict": VERDICT_FAIL, "passed": False, "error": message},
                                               indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(md, encoding="utf-8")
    return md


def write_gate_outputs(out_dir: Path | str, cfg: GateConfig, checks: list[Check], results: dict, meta: dict,
                       merchants=None, skew: dict | None = None, baseline: BaselineResult | None = None) -> dict:
    """``out_dir/audit.json`` (verdict, checks, gate, every run via ``to_json``, skew, the single-transaction
    ``baseline`` or null) and ``out_dir/summary.md``. The baseline never changes the verdict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ok = passed(checks)
    meta = dict(meta, gate=dataclasses.asdict(cfg))
    worst = next(c.worst for c in checks if c.name == "ver_uplift")
    data = {"verdict": VERDICT_OK if ok else VERDICT_FAIL, "passed": ok,
            "checks": [dataclasses.asdict(c) for c in checks], "gate": dataclasses.asdict(cfg),
            "worst_run": worst, "runs": {k: to_json(r, merchants) for k, r in results.items()},
            "skew": skew, "baseline": None if baseline is None else baseline_to_json(baseline), "meta": meta}
    (out / "audit.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(summary_markdown(checks, results, meta, merchants, baseline), encoding="utf-8")
    return data
