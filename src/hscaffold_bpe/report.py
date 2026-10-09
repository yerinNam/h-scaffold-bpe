from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


VARIANT_LABELS = {
    "bpe": "BPE",
    "scaffold": "Scaffold-BPE",
    "hierarchical": "H-Scaffold-BPE",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return html.escape(str(value))


def _parse_run(name: str) -> tuple[str, str, int] | None:
    for architecture in ("llama-3.2-1b", "llama-3.1-8b"):
        prefix = architecture + "-"
        if not name.startswith(prefix):
            continue
        remainder = name[len(prefix) :]
        variant, vocab = remainder.rsplit("-", 1)
        return architecture, variant, int(vocab)
    return None


def _training_speed(run_dir: Path) -> float | None:
    path = run_dir / "train_log.jsonl"
    if not path.exists():
        return None
    values: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                values.append(float(json.loads(line)["tokens_per_second"]))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
    return statistics.median(values[-100:]) if values else None


def _table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _pct_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return (value / baseline - 1.0) * 100.0


def _safe_div(value: float | None, denominator: float | None) -> float | None:
    if value is None or denominator in (None, 0):
        return None
    return value / denominator


def _delta_cell(value: float | None, lower_is_better: bool) -> str:
    if value is None:
        return "—"
    improved = value < 0 if lower_is_better else value > 0
    css = "good" if improved else "bad" if value else "neutral"
    verdict = "개선" if improved else "저하" if value else "동일"
    return f"<span class='{css}'>{value:+.2f}% ({verdict})</span>"


def generate(
    eval_dir: Path, lm_dir: Path, destination: Path, architectures: tuple[str, ...],
    ablation_path: Path | None = None,
) -> None:
    tokenizer_rows: list[dict] = []
    for path in sorted(eval_dir.glob("*.json")):
        payload = _load_json(path)
        domains = {Path(item["path"]).stem: item for item in payload.get("domains", [])}
        tokenizer_rows.append(
            {
                **payload,
                "wiki": domains.get("wikitext103_test", {}),
                "wmt": domains.get("wmt17_de_en_test_en", {}),
                "blimp": domains.get("blimp_all", {}),
            }
        )

    lm_rows: list[dict] = []
    for run_dir in sorted(path for path in lm_dir.iterdir() if path.is_dir()) if lm_dir.exists() else []:
        parsed = _parse_run(run_dir.name)
        result_path = run_dir / "result.json"
        test_path = run_dir / "test_result.json"
        if parsed is None or not result_path.exists():
            continue
        architecture, variant, vocab = parsed
        if architecture not in architectures:
            continue
        result = _load_json(result_path)
        test = _load_json(test_path).get("test", {}) if test_path.exists() else {}
        lm_rows.append(
            {
                "architecture": architecture,
                "variant": variant,
                "vocab": vocab,
                "result": result,
                "test": test,
                "speed": _training_speed(run_dir),
            }
        )

    expected_tokenizers = 6
    expected_lms = 6 * len(architectures)
    complete = len(tokenizer_rows) == expected_tokenizers and len(lm_rows) == expected_lms and all(
        row["test"] for row in lm_rows
    )
    status = "완료" if complete else "진행 중 / 일부 결과 누락"

    tok_table = _table(
        ["방법", "Vocab", "Expanded", "L0 / L1 / L2", "평균 fertility", "Wiki fertility", "Wiki bytes/token", "WMT fertility", "BLiMP fertility", "도메인 σ", "도메인 CV", "Wiki MB/s", "학습 시간"],
        [
            [
                VARIANT_LABELS.get(row["variant"], row["variant"]),
                _fmt(int(row["target_vocab_size"]), 0),
                _fmt(int(row["expanded_vocab_size"]), 0),
                " / ".join(str(row.get("level_counts", {}).get(str(level), 0)) for level in (0, 1, 2)),
                _fmt(row.get("fertility_mean")),
                _fmt(row["wiki"].get("fertility")),
                _fmt(row["wiki"].get("bytes_per_token")),
                _fmt(row["wmt"].get("fertility")),
                _fmt(row["blimp"].get("fertility")),
                _fmt(row.get("fertility_population_std"), 4),
                _fmt(_safe_div(row.get("fertility_population_std"), row.get("fertility_mean")), 4),
                _fmt(row["wiki"].get("megabytes_per_second"), 2),
                _fmt(row.get("tokenizer_training_seconds"), 1) + " s",
            ]
            for row in tokenizer_rows
        ],
    )

    lm_table = _table(
        ["Architecture", "방법", "Vocab", "Params", "Train tok/s", "Train 시간", "Validation PPL", "Validation BPB", "Test PPL", "Test BPB"],
        [
            [
                html.escape(row["architecture"]),
                VARIANT_LABELS.get(row["variant"], row["variant"]),
                _fmt(row["vocab"], 0),
                _fmt(row["result"].get("parameter_count"), 0),
                _fmt(row["speed"], 0),
                _fmt(row["result"].get("wall_seconds", 0) / 3600 if row["result"].get("wall_seconds") else None, 2) + " h",
                _fmt(row["result"].get("validation", {}).get("perplexity")),
                _fmt(row["result"].get("validation", {}).get("bits_per_byte")),
                _fmt(row["test"].get("perplexity")),
                _fmt(row["test"].get("bits_per_byte")),
            ]
            for row in lm_rows
        ],
    )

    tokenizer_index = {
        (row["variant"], int(row["target_vocab_size"])): row for row in tokenizer_rows
    }
    lm_index = {
        (row["architecture"], row["variant"], row["vocab"]): row for row in lm_rows
    }
    comparison_rows: list[list[object]] = []
    interpretation: list[str] = []
    training_deltas: list[float] = []
    fertility_deltas: list[float] = []
    robustness_deltas: list[float] = []
    encoding_deltas: list[float] = []
    ppl_deltas: list[float] = []
    bpb_deltas: list[float] = []

    for vocab in (32768, 65536):
        ours = tokenizer_index.get(("hierarchical", vocab))
        base = tokenizer_index.get(("scaffold", vocab))
        if ours and base:
            metrics = [
                ("Tokenizer 학습 시간", ours.get("tokenizer_training_seconds"), base.get("tokenizer_training_seconds"), True, training_deltas),
                ("평균 fertility", ours.get("fertility_mean"), base.get("fertility_mean"), True, fertility_deltas),
                ("도메인 fertility σ", ours.get("fertility_population_std"), base.get("fertility_population_std"), True, robustness_deltas),
                ("Wiki 인코딩 MB/s", ours["wiki"].get("megabytes_per_second"), base["wiki"].get("megabytes_per_second"), False, encoding_deltas),
            ]
            for metric, value, baseline, lower_better, bucket in metrics:
                delta = _pct_delta(value, baseline)
                if delta is not None:
                    bucket.append(delta)
                comparison_rows.append(
                    [metric, f"{vocab // 1024}K", _fmt(baseline), _fmt(value), _delta_cell(delta, lower_better)]
                )

    for architecture in architectures:
        for vocab in (32768, 65536):
            ours = lm_index.get((architecture, "hierarchical", vocab))
            base = lm_index.get((architecture, "scaffold", vocab))
            if not ours or not base:
                continue
            for metric, key, bucket in (("Test PPL", "perplexity", ppl_deltas), ("Test BPB", "bits_per_byte", bpb_deltas)):
                value = ours["test"].get(key)
                baseline = base["test"].get(key)
                delta = _pct_delta(value, baseline)
                if delta is not None:
                    bucket.append(delta)
                comparison_rows.append(
                    [metric, f"{architecture} · {vocab // 1024}K", _fmt(baseline), _fmt(value), _delta_cell(delta, True)]
                )

    def mean_delta(values: list[float]) -> float | None:
        return statistics.fmean(values) if values else None

    train_delta = mean_delta(training_deltas)
    fertility_delta = mean_delta(fertility_deltas)
    robust_delta = mean_delta(robustness_deltas)
    encode_delta = mean_delta(encoding_deltas)
    ppl_delta = mean_delta(ppl_deltas)
    bpb_delta = mean_delta(bpb_deltas)
    if train_delta is not None:
        interpretation.append(
            f"H-Scaffold tokenizer 학습 시간은 Scaffold-BPE 대비 평균 <b>{train_delta:+.2f}%</b>다. "
            + ("예상대로 느려 레벨 판정·queue 재삽입 비용이 관측됐다." if train_delta > 0 else "예상과 달리 느려지지 않았다; 구현 및 실행 변동을 추가 확인해야 한다.")
        )
    if encode_delta is not None:
        interpretation.append(
            f"WikiText 인코딩 처리량은 평균 <b>{encode_delta:+.2f}%</b>다. 학습 시 레벨 계산 비용과 달리, 추론 시 차이는 staged demolition 및 expanded merge 수에서 발생한다."
        )
    if fertility_delta is not None and robust_delta is not None:
        interpretation.append(
            f"Compression은 fertility 기준 <b>{fertility_delta:+.2f}%</b>, robustness는 도메인 표준편차 기준 <b>{robust_delta:+.2f}%</b>다. 두 지표는 낮을수록 좋다."
        )
    if ppl_delta is not None and bpb_delta is not None:
        direction = "개선" if bpb_delta < 0 else "저하"
        interpretation.append(
            f"LM 품질은 Scaffold-BPE 대비 평균 Test PPL <b>{ppl_delta:+.2f}%</b>, BPB <b>{bpb_delta:+.2f}%</b>로 {direction}됐다. tokenizer 간에는 토큰 단위가 달라 PPL보다 <b>BPB를 우선</b> 해석한다."
        )
        if bpb_delta < 0:
            interpretation.append(
                "가능한 설명: Level 2를 조기 철거해 희귀 중간 조각이 visible vocabulary를 차지하는 것을 줄이면서 Level 1의 조합 기능은 유지해, 같은 visible vocab에서 모델이 학습할 토큰의 품질이 높아졌을 수 있다. 이는 인과 증명이 아니라 fertility·BPB와 일치하는 해석이다."
            )
        else:
            interpretation.append(
                "가능한 설명: 고정된 0.5 경계가 유용한 중간 형태소까지 너무 일찍 분해했거나 시퀀스를 길게 만들어 최적화를 어렵게 했을 수 있다. fertility가 함께 악화됐는지 확인하면 이 설명을 구분할 수 있으며, 인과 확인에는 threshold ablation이 필요하다."
            )

    detailed_sections: list[tuple[str, list[str]]] = []

    compression_notes: list[str] = []
    for vocab in (32768, 65536):
        ours = tokenizer_index.get(("hierarchical", vocab))
        if not ours:
            continue
        comparisons = []
        for baseline_name in ("bpe", "scaffold"):
            baseline = tokenizer_index.get((baseline_name, vocab))
            if baseline:
                delta = _pct_delta(ours.get("fertility_mean"), baseline.get("fertility_mean"))
                comparisons.append(
                    f"{VARIANT_LABELS[baseline_name]} 대비 {_fmt(delta, 2)}%"
                )
        domain_bits = []
        byte_delta = None
        base = tokenizer_index.get(("scaffold", vocab))
        if base:
            for label, key in (("Wikipedia", "wiki"), ("WMT", "wmt"), ("BLiMP", "blimp")):
                delta = _pct_delta(ours[key].get("fertility"), base[key].get("fertility"))
                domain_bits.append(f"{label} {_fmt(delta, 2)}%")
            byte_delta = _pct_delta(
                ours["wiki"].get("bytes_per_token"), base["wiki"].get("bytes_per_token")
            )
        compression_notes.append(
            f"{vocab // 1024}K 평균 fertility: " + ", ".join(comparisons)
            + ("; Scaffold 대비 도메인별 " + ", ".join(domain_bits) if domain_bits else "")
            + (f"; Wiki bytes/token {_fmt(byte_delta, 2)}%" if byte_delta is not None else "")
            + ". fertility는 음수, bytes/token은 양수일 때 더 적은 토큰으로 표현해 개선이다."
        )
    detailed_sections.append(("1. Compression", compression_notes))

    quality_notes: list[str] = []
    scale_deltas: dict[str, list[float]] = {architecture: [] for architecture in architectures}
    for architecture in architectures:
        for vocab in (32768, 65536):
            ours = lm_index.get((architecture, "hierarchical", vocab))
            if not ours or not ours["test"]:
                continue
            parts = []
            for baseline_name in ("bpe", "scaffold"):
                baseline = lm_index.get((architecture, baseline_name, vocab))
                if baseline and baseline["test"]:
                    delta = _pct_delta(
                        ours["test"].get("bits_per_byte"), baseline["test"].get("bits_per_byte")
                    )
                    parts.append(f"{VARIANT_LABELS[baseline_name]} 대비 BPB {_fmt(delta, 2)}%")
                    if baseline_name == "scaffold" and delta is not None:
                        scale_deltas[architecture].append(delta)
            quality_notes.append(
                f"{architecture} · {vocab // 1024}K: " + ", ".join(parts)
                + ". BPB가 음수면 tokenizer 경계를 보정한 실질적인 LM 개선이다."
            )
    detailed_sections.append(("2. Proxy LM quality", quality_notes))

    robustness_notes: list[str] = []
    for vocab in (32768, 65536):
        ours = tokenizer_index.get(("hierarchical", vocab))
        base = tokenizer_index.get(("scaffold", vocab))
        if ours and base:
            delta = _pct_delta(
                ours.get("fertility_population_std"), base.get("fertility_population_std")
            )
            ours_values = [ours[key].get("fertility") for key in ("wiki", "wmt", "blimp")]
            base_values = [base[key].get("fertility") for key in ("wiki", "wmt", "blimp")]
            ours_range = max(ours_values) - min(ours_values) if all(v is not None for v in ours_values) else None
            base_range = max(base_values) - min(base_values) if all(v is not None for v in base_values) else None
            ours_cv = _safe_div(
                ours.get("fertility_population_std"), ours.get("fertility_mean")
            )
            base_cv = _safe_div(
                base.get("fertility_population_std"), base.get("fertility_mean")
            )
            cv_delta = _pct_delta(ours_cv, base_cv)
            robustness_notes.append(
                f"{vocab // 1024}K: fertility 표준편차는 Scaffold 대비 {_fmt(delta, 2)}%; "
                f"CV는 {_fmt(cv_delta, 2)}%; 도메인 범위는 Scaffold {_fmt(base_range, 4)} → H {_fmt(ours_range, 4)}. 모두 작을수록 domain shift에 안정적이다."
            )
    detailed_sections.append(("3. Robustness", robustness_notes))

    cost_notes: list[str] = []
    if train_delta is not None:
        cost_notes.append(
            f"Tokenizer 학습 시간은 Scaffold 대비 평균 {_fmt(train_delta, 2)}%, Wikipedia 인코딩 MB/s는 {_fmt(encode_delta, 2)}% 변했다. 학습과 인코딩 비용을 분리해 해석한다."
        )
    for architecture in architectures:
        for vocab in (32768, 65536):
            ours = lm_index.get((architecture, "hierarchical", vocab))
            base = lm_index.get((architecture, "scaffold", vocab))
            if ours and base:
                wall = _pct_delta(
                    ours["result"].get("wall_seconds"), base["result"].get("wall_seconds")
                )
                speed = _pct_delta(ours.get("speed"), base.get("speed"))
                cost_notes.append(
                    f"{architecture} · {vocab // 1024}K: LM wall time {_fmt(wall, 2)}%, train tok/s {_fmt(speed, 2)}% (H vs Scaffold)."
                )
    detailed_sections.append(("4. Training / search cost", cost_notes))

    vocab_notes: list[str] = []
    for variant in ("bpe", "scaffold", "hierarchical"):
        small = tokenizer_index.get((variant, 32768))
        large = tokenizer_index.get((variant, 65536))
        if small and large:
            delta = _pct_delta(large.get("fertility_mean"), small.get("fertility_mean"))
            vocab_notes.append(
                f"{VARIANT_LABELS[variant]}: 32K→64K에서 평균 fertility {_fmt(delta, 2)}%. 압축 이득과 embedding/LM-head 파라미터 증가를 함께 본다."
            )
    detailed_sections.append(("5. Vocabulary-size sensitivity", vocab_notes))

    scale_notes: list[str] = []
    for architecture, values in scale_deltas.items():
        if values:
            scale_notes.append(
                f"{architecture}: H의 Scaffold 대비 평균 Test BPB 변화 {_fmt(statistics.fmean(values), 2)}%."
            )
    if len(scale_notes) > 1:
        scale_notes.append(
            "1B와 8B의 개선 폭이 다르면 tokenizer 효과가 모델 용량과 상호작용한다. 두 모델에서 방향이 같을 때 일반화 근거가 더 강하다."
        )
    detailed_sections.append(("6. Model-scale interaction", scale_notes))

    level_notes: list[str] = []
    for vocab in (32768, 65536):
        ours = tokenizer_index.get(("hierarchical", vocab))
        if ours:
            counts = ours.get("level_counts", {})
            expanded = int(ours.get("expanded_vocab_size", 0))
            visible = int(ours.get("target_vocab_size", 0))
            overhead = (expanded / visible - 1) * 100 if visible else None
            level_notes.append(
                f"{vocab // 1024}K: L0/L1/L2 = {counts.get('0', 0):,}/{counts.get('1', 0):,}/{counts.get('2', 0):,}, expanded-vocab overhead {_fmt(overhead, 2)}%. Level 2 비율이 비용 증가와 품질 변화의 구조적 단서다."
            )
    detailed_sections.append(("7. Hierarchy behavior", level_notes))

    limitations = [
        "각 조건이 single seed이므로 작은 차이는 통계적 유의성으로 해석하지 않는다. 후속 실험은 최소 3 seeds가 필요하다.",
        "Scaffold-BPE 공식 코드가 공개되지 않아 논문 알고리즘의 독립 구현을 baseline으로 사용했다.",
        "Threshold ablation은 토크나이저 수준만 평가했으며, threshold별 LM 학습이나 BPB 비교는 수행하지 않았다.",
        "PPL은 tokenizer별 토큰 단위가 달라 직접 비교에 편향이 있다. 결론은 BPB와 fertility를 우선한다.",
    ]
    detailed_sections.append(("8. Limitations & follow-up", limitations))

    detailed_html = "".join(
        f"<h3>{html.escape(title)}</h3>"
        + ("<ul>" + "".join(f"<li>{note}</li>" for note in notes) + "</ul>" if notes else "<p>평가 완료 후 자동 생성.</p>")
        for title, notes in detailed_sections
    )

    comparison_table = _table(
        ["지표", "조건", "Scaffold-BPE", "H-Scaffold", "H vs Scaffold"], comparison_rows
    )
    interpretation_html = (
        "<ul>" + "".join(f"<li>{item}</li>" for item in interpretation) + "</ul>"
        if interpretation
        else "<p class='note'>최종 평가가 끝나면 증감률과 원인 해석이 자동으로 채워진다.</p>"
    )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    model_scope = " · ".join(architectures)
    ablation_html = ablation_path.read_text(encoding="utf-8") if ablation_path and ablation_path.exists() else ""
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hierarchical Scaffold-BPE 실험 보고서</title>
<style>
:root{{--ink:#18212b;--muted:#627080;--line:#dce3e9;--paper:#fff;--accent:#176b87;--soft:#edf7fa}}
*{{box-sizing:border-box}} body{{margin:0;background:#f3f5f7;color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Noto Sans KR",sans-serif}}
main{{max-width:1380px;margin:32px auto;padding:0 24px 64px}} header{{background:linear-gradient(125deg,#123a4a,#176b87);color:white;padding:32px;border-radius:16px}}
h1{{margin:0 0 8px;font-size:32px}} h2{{margin:30px 0 12px;font-size:22px}} .meta{{opacity:.82}} .cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}}
.card{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:16px}} .card b{{display:block;font-size:24px;color:var(--accent)}}
.note{{background:var(--soft);border-left:4px solid var(--accent);padding:12px 16px;border-radius:6px}} .table-wrap{{overflow:auto;background:white;border:1px solid var(--line);border-radius:12px}}
table{{border-collapse:collapse;width:100%;white-space:nowrap}} th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}} th{{background:#f7fafb;color:#344656;font-size:13px}} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
code{{background:#edf1f4;padding:2px 5px;border-radius:4px}} .good{{color:#087443;font-weight:700}} .bad{{color:#b42318;font-weight:700}} .neutral{{color:var(--muted)}} footer{{margin-top:30px;color:var(--muted)}} @media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<header><h1>Hierarchical Scaffold-BPE</h1><div>Tokenizer compression, robustness, and scratch-LM quality</div><div class="meta">생성 {generated} · 상태: {status} · <a href="example.html" style="color:white">실제 토큰화 예시 보기</a></div></header>
<div class="cards"><div class="card"><span>Tokenizer 결과</span><b>{len(tokenizer_rows)}/6</b></div><div class="card"><span>LM 결과</span><b>{len(lm_rows)}/{expected_lms}</b></div><div class="card"><span>Test 평가</span><b>{sum(bool(row['test']) for row in lm_rows)}/{expected_lms}</b></div><div class="card"><span>Vocab</span><b>32K · 64K</b></div></div>
<section><h2>실험 설계</h2><p class="note">Wikipedia EN 1B로 학습한 byte-level BPE, Scaffold-BPE, H-Scaffold-BPE를 비교한다. 각 tokenizer로 WikiText-103을 재토큰화하고 {html.escape(model_scope)} 구조를 5 epochs scratch 학습한다. 모델 간 비교에는 token PPL과 tokenizer에 독립적인 bits-per-byte를 함께 사용한다.</p></section>
<section><h2>Tokenizer: compression & robustness</h2>{tok_table}</section>
<section><h2>Language model quality</h2>{lm_table}</section>
<section><h2>H-Scaffold vs Scaffold-BPE</h2>{comparison_table}<h2>결과 해석</h2>{interpretation_html}</section>
<section><h2>분석 항목</h2>{detailed_html}</section>
{ablation_html}
<section><h2>재현 조건</h2><p>GPU 4–7의 NVIDIA B200, BF16 compute, B200 Flash-Attention, sequence length 1024, effective batch 16. 1B는 gradient checkpointing OFF이며 optimizer는 8-bit AdamW이다. H-Scaffold는 Level 2를 먼저 철거한 뒤 Level 1을 철거한다.</p></section>
<footer>원시 결과: <code>artifacts/eval</code>, <code>artifacts/lm</code>. 본 문서는 후처리 스크립트가 자동 생성했다. <a href="example.html">토큰화 예시와 동작 차이</a></footer>
</main></body></html>"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--lm-dir", type=Path, default=Path("artifacts/lm"))
    parser.add_argument("--output", type=Path, default=Path("report/report.html"))
    parser.add_argument("--ablation", type=Path, default=Path("artifacts/ablation/analysis.html"))
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["llama-3.2-1b", "llama-3.1-8b"],
        choices=["llama-3.2-1b", "llama-3.1-8b"],
    )
    args = parser.parse_args()
    generate(args.eval_dir, args.lm_dir, args.output, tuple(args.architectures), args.ablation)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
