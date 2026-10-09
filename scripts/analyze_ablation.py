"""Summarize L2 tokens and observed second-pass merges on the WikiText test set."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

from hscaffold_bpe.model import TokenizerModel
from hscaffold_bpe.pretokenize import iter_pretokens
from hscaffold_bpe.tokenizer import HierarchicalTokenizer

SUFFIXES = ("tion", "sion", "ment", "ness", "ally", "able", "ible", "ing", "ers", "ed", "ly")


def show(value: bytes) -> str:
    return value.decode("utf-8", "backslashreplace").replace(" ", "␠").replace("\n", "↵")


def traits(value: bytes) -> list[str]:
    decoded = value.decode("utf-8", "replace")
    bare = decoded.strip()
    labels = []
    if decoded.startswith(" "):
        labels.append("leading_space")
    if bare.isalpha():
        labels.append("alphabetic")
    if bare.isdigit():
        labels.append("numeric")
    if any(char.isdigit() for char in bare) and any(char.isalpha() for char in bare):
        labels.append("alphanumeric")
    if bare and bare[0].isupper() and any(char.islower() for char in bare):
        labels.append("capitalized_candidate")
    if (
        bare.isalpha()
        and bare.islower()
        and not decoded.startswith(" ")
        and len(bare) >= 3
        and bare.endswith(SUFFIXES)
    ):
        labels.append("suffix_shape")
    if not bare.isalnum():
        labels.append("contains_symbol")
    return labels


def trace_second_pass(
    tokenizer: HierarchicalTokenizer, pretoken: bytes
) -> tuple[list[int], list[int], list[tuple[int, bool, bool]], list[int]]:
    model = tokenizer.model
    expanded = tokenizer._apply_bpe(list(pretoken), max_output_level=2)
    l2 = [token_id for token_id in expanded if model.levels[token_id] == 2]
    if not l2:
        return expanded, expanded, [], []
    # Each early token remembers the expanded token from which it came.
    sequence: list[tuple[int, frozenset[int]]] = []
    for origin, token_id in enumerate(expanded):
        for child in tokenizer._demolish_token(token_id, max_level=1):
            sequence.append((child, frozenset((origin,))))
    early = [token_id for token_id, _ in sequence]
    events: list[tuple[int, bool, bool]] = []
    while len(sequence) > 1:
        candidates = [
            (tokenizer._rank[(left[0], right[0])][0], left[0], right[0])
            for left, right in zip(sequence, sequence[1:])
            if (left[0], right[0]) in tokenizer._rank
            and model.levels[tokenizer._rank[(left[0], right[0])][1]] <= 1
        ]
        if not candidates:
            break
        _, best_left, best_right = min(candidates)
        output = tokenizer._rank[(best_left, best_right)][1]
        next_sequence = []
        index = 0
        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index][0] == best_left
                and sequence[index + 1][0] == best_right
            ):
                left, right = sequence[index : index + 2]
                origins = left[1] | right[1]
                crosses = left[1] != right[1]
                touches_l2 = any(model.levels[expanded[origin]] == 2 for origin in origins)
                events.append((output, crosses, touches_l2))
                next_sequence.append((output, origins))
                index += 2
            else:
                next_sequence.append(sequence[index])
                index += 1
        sequence = next_sequence
    result = [token_id for token_id, _ in sequence]
    if result != tokenizer._apply_bpe(early, max_output_level=1):
        raise AssertionError("second-pass trace differs from tokenizer implementation")
    return early, result, events, l2


def table(headers: list[str], rows: list[list[object]]) -> str:
    return (
        "<div class='table-wrap'><table><thead><tr>"
        + "".join(f"<th>{html.escape(item)}</th>" for item in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/ablation/tokenizers"))
    parser.add_argument("--eval-dir", type=Path, default=Path("artifacts/ablation/eval"))
    parser.add_argument("--wiki", type=Path, default=Path("data/eval/wikitext103_test.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ablation"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wiki = args.wiki.read_text(encoding="utf-8", errors="ignore")
    pretoken_counts = Counter(iter_pretokens(wiki))
    results = []
    for vocab in (32768, 65536):
        for threshold in (0.25, 0.5, 0.75):
            tag = f"t{round(threshold * 100):03d}"
            model_path = (
                Path("artifacts/tokenizers") / f"hierarchical-{vocab}.json"
                if threshold == 0.5
                else args.model_dir / tag / f"hierarchical-{vocab}.json"
            )
            model = TokenizerModel.load(model_path)
            tokenizer = HierarchicalTokenizer(model, backend="python")
            eval_path = args.eval_dir / f"{tag}-{vocab}.json"
            evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
            l2_ids = [token_id for token_id, level in model.levels.items() if level == 2]
            merge_frequency = {merge.output: merge.frequency for merge in model.merges}
            trait_counts = Counter(
                label for token_id in l2_ids for label in traits(model.tokens[token_id])
            )
            rows = [
                {
                    "id": token_id,
                    "token": show(model.tokens[token_id]),
                    "hex": model.tokens[token_id].hex(),
                    "bytes": len(model.tokens[token_id]),
                    "merge_frequency_at_creation": merge_frequency.get(token_id, ""),
                    "traits": ";".join(traits(model.tokens[token_id])),
                }
                for token_id in l2_ids
            ]
            rows.sort(key=lambda row: (-int(row["merge_frequency_at_creation"] or 0), row["id"]))
            csv_path = args.output_dir / f"l2-{tag}-{vocab}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            l2_occurrences = Counter()
            remerge_outputs = Counter()
            cross_outputs = Counter()
            examples = Counter()
            affected_pretokens = 0
            remerge_events = 0
            crossing_events = 0
            for pretoken, weight in pretoken_counts.items():
                early, merged, events, l2 = trace_second_pass(tokenizer, pretoken)
                if not l2:
                    continue
                affected_pretokens += weight
                l2_occurrences.update(
                    {token_id: amount * weight for token_id, amount in Counter(l2).items()}
                )
                for output, crosses, touches_l2 in events:
                    remerge_events += weight
                    remerge_outputs[output] += weight
                    if crosses and touches_l2:
                        crossing_events += weight
                        cross_outputs[output] += weight
                if events and early != merged:
                    examples[pretoken] += weight
            domains = {Path(item["path"]).stem: item for item in evaluation["domains"]}
            result = {
                "vocab": vocab,
                "threshold": threshold,
                "l2_count": len(l2_ids),
                "l1_count": sum(level == 1 for level in model.levels.values()),
                "expanded_vocab": len(model.tokens),
                "mean_l2_bytes": sum(len(model.tokens[token_id]) for token_id in l2_ids)
                / len(l2_ids),
                "traits": dict(trait_counts),
                "fertility_mean": evaluation["fertility_mean"],
                "domain_fertility": {name: item["fertility"] for name, item in domains.items()},
                "wiki_pretokens": sum(pretoken_counts.values()),
                "affected_pretokens": affected_pretokens,
                "l2_occurrences": sum(l2_occurrences.values()),
                "remerge_events": remerge_events,
                "crossing_events": crossing_events,
                "top_l2": [
                    {"token": show(model.tokens[token_id]), "count": count}
                    for token_id, count in l2_occurrences.most_common(20)
                ],
                "top_suffix_shape": [
                    {"token": show(model.tokens[token_id]), "count": count}
                    for token_id, count in l2_occurrences.most_common()
                    if "suffix_shape" in traits(model.tokens[token_id])
                ][:15],
                "top_capitalized": [
                    {"token": show(model.tokens[token_id]), "count": count}
                    for token_id, count in l2_occurrences.most_common()
                    if "capitalized_candidate" in traits(model.tokens[token_id])
                ][:15],
                "top_remerged": [
                    {"token": show(model.tokens[token_id]), "count": count}
                    for token_id, count in remerge_outputs.most_common(20)
                ],
                "top_crossing": [
                    {"token": show(model.tokens[token_id]), "count": count}
                    for token_id, count in cross_outputs.most_common(20)
                ],
                "examples": [
                    {
                        "pretoken": show(token),
                        "count": count,
                        "expanded": " | ".join(
                            show(model.tokens[item])
                            for item in tokenizer._apply_bpe(list(token), max_output_level=2)
                        ),
                        "early": " | ".join(
                            show(model.tokens[item])
                            for item in trace_second_pass(tokenizer, token)[0]
                        ),
                        "remerged": " | ".join(
                            show(model.tokens[item])
                            for item in trace_second_pass(tokenizer, token)[1]
                        ),
                    }
                    for token, count in examples.most_common(12)
                ],
                "csv": csv_path.name,
            }
            results.append(result)
            print(f"{tag} {vocab}: L2={len(l2_ids)} remerge={remerge_events}", flush=True)

    (args.output_dir / "analysis.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    baseline = {(r["vocab"]): r for r in results if r["threshold"] == 0.5}
    comparison = []
    for result in results:
        base = baseline[result["vocab"]]
        comparison.append(
            [
                f"{result['vocab'] // 1024}K",
                result["threshold"],
                f"{result['l2_count']:,}",
                f"{result['l1_count']:,}",
                f"{result['fertility_mean']:.6f}",
                f"{(result['fertility_mean'] / base['fertility_mean'] - 1) * 100:+.4f}%",
                f"{result['affected_pretokens']:,}",
                f"{result['remerge_events']:,}",
                f"{result['crossing_events']:,}",
            ]
        )
    sections = [
        "<section id='ablation'><h2>L2 threshold ablation · 토크나이저 전용</h2>",
        "<p class='note'>같은 Wikipedia EN 1B pretoken counts로 0.25/0.5/0.75를 학습하고, 같은 WikiText-103 test·WMT17·BLiMP에서 평가했다. 0.5는 기존 모델을 재사용했다. Fertility 평균은 세 도메인의 산술 평균이다. 재병합 횟수는 WikiText-103 test의 pretoken 출현 빈도로 가중한 실제 2차 merge 이벤트 수다. 2차 merge 중 원래 expanded-token 경계를 가로지르며 L2 조각에 닿은 이벤트를 별도로 센다.</p>",
        table(
            [
                "Vocab",
                "L2 경계",
                "L2 수",
                "L1 수",
                "평균 fertility",
                "0.5 대비",
                "L2 포함 pretoken",
                "2차 merge",
                "L2 경계 횡단",
            ],
            comparison,
        ),
        "<p>경계를 0.25에서 0.75로 올리면 L2 수와 실제 재병합은 모두 증가한다. "
        + "평균 fertility 변화는 "
        + ", ".join(
            f"{vocab // 1024}K에서 "
            f"{(next(r for r in results if r['vocab'] == vocab and r['threshold'] == 0.75)['fertility_mean'] / next(r for r in results if r['vocab'] == vocab and r['threshold'] == 0.25)['fertility_mean'] - 1) * 100:+.4f}%"
            for vocab in (32768, 65536)
        )
        + "로 작다. 0.5 모델에서 재병합은 WikiText test 247,230개 pretoken당 "
        + f"32K {baseline[32768]['remerge_events']}회, "
        + f"64K {baseline[65536]['remerge_events']}회 관측됐다. "
        + "이 결과만으로 LM 품질 변화의 원인을 단정할 수 없다.</p>",
        "<h3>도메인별 fertility</h3>",
        table(
            ["Vocab", "L2 경계", "WikiText-103", "WMT17", "BLiMP"],
            [
                [
                    f"{result['vocab'] // 1024}K",
                    result["threshold"],
                    *[
                        f"{result['domain_fertility'][key]:.6f}"
                        for key in (
                            "wikitext103_test",
                            "wmt17_de_en_test_en",
                            "blimp_all",
                        )
                    ],
                ]
                for result in results
            ],
        ),
        "<h3>최종 L2 토큰 분포</h3><p>전체 목록은 각 행의 CSV에서 확인할 수 있다. 접미사형은 앞 공백 없이 소문자로 시작하고 지정한 영어 접미사 문자열로 끝나는 조각이다. 이 표면형 휴리스틱은 실제 형태소나 품사를 판정하지 않는다. 대문자 시작도 고유명사 판정이 아니다. 학습 도중 잠시 L2였다가 재활성화된 토큰은 최종 L2 목록에 포함하지 않는다.</p>",
    ]
    traits_rows = []
    for result in results:
        counts = result["traits"]
        total = result["l2_count"]
        traits_rows.append(
            [
                f"{result['vocab'] // 1024}K",
                result["threshold"],
                f"{result['mean_l2_bytes']:.2f}",
                *[
                    f"{counts.get(key, 0):,} ({counts.get(key, 0) / total:.1%})"
                    for key in (
                        "leading_space",
                        "alphabetic",
                        "suffix_shape",
                        "capitalized_candidate",
                    )
                ],
                result["csv"],
            ]
        )
    sections.append(
        table(
            [
                "Vocab",
                "경계",
                "평균 byte",
                "앞 공백",
                "알파벳",
                "접미사형 조각",
                "대문자 시작",
                "L2 목록",
            ],
            traits_rows,
        )
    )
    sections.append(
        "<p>전체 CSV: "
        + " · ".join(
            f'<a href="../artifacts/ablation/{html.escape(result["csv"])}">{result["vocab"] // 1024}K / {result["threshold"]}</a>'
            for result in results
        )
        + ".</p>"
    )
    for vocab in (32768, 65536):
        result = baseline[vocab]
        sections.append(f"<h3>0.5 기준 {vocab // 1024}K · 실제 L2와 재병합</h3>")
        sections.append(
            table(
                ["Wiki test에서 자주 등장한 L2", "출현 횟수"],
                [[item["token"], f"{item['count']:,}"] for item in result["top_l2"][:15]],
            )
        )
        sections.append(
            table(
                ["L2 분해 후 2차 merge 결과", "횟수"],
                [[item["token"], f"{item['count']:,}"] for item in result["top_remerged"][:15]],
            )
        )
        sections.append(
            table(
                ["원래 경계를 넘는 재병합 결과", "횟수"],
                [[item["token"], f"{item['count']:,}"] for item in result["top_crossing"][:10]],
            )
        )
        sections.append(
            table(
                ["접미사형 L2 조각", "Wiki 출현", "대문자 시작 L2 후보", "Wiki 출현"],
                [
                    [
                        result["top_suffix_shape"][i]["token"]
                        if i < len(result["top_suffix_shape"])
                        else "",
                        result["top_suffix_shape"][i]["count"]
                        if i < len(result["top_suffix_shape"])
                        else "",
                        result["top_capitalized"][i]["token"]
                        if i < len(result["top_capitalized"])
                        else "",
                        result["top_capitalized"][i]["count"]
                        if i < len(result["top_capitalized"])
                        else "",
                    ]
                    for i in range(10)
                ],
            )
        )
        sections.append(
            table(
                ["pretoken", "출현", "최초 expanded", "L2 분해", "2차 재병합"],
                [
                    [
                        item["pretoken"],
                        item["count"],
                        item["expanded"],
                        item["early"],
                        item["remerged"],
                    ]
                    for item in result["examples"][:8]
                ],
            )
        )
    sections.append(
        "<p class='note'>2차 merge 결과는 반드시 최종 visible token이라는 뜻은 아니다. Level 1은 마지막 단계에서 다시 분해된다. 이 집계는 merge 이벤트를 세므로 한 pretoken에서 여러 번 발생할 수 있다. 분해 전후 token 수만으로 재병합 빈도를 대체하지 않았다.</p></section>"
    )
    (args.output_dir / "analysis.html").write_text("".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
