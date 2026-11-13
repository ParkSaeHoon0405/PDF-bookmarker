#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_bookmark_generator_fixed.py
================================

PDF 안에 여러 개의 문서가 연속해서 들어 있어도 문서별 계층형 북마크를 생성합니다.

핵심 규칙
---------
1) '1.0 ...'이 나오면 새 문서의 시작으로 판단합니다.
   따라서 하나의 PDF 안에 1.0이 여러 번 나올 수 있습니다.

2) 각 새 문서의 최상위 북마크 제목은 해당 1.0 이전에 있는 제목 영역에서 찾습니다.
   예: Appendix A
       └─ 1.0 SUMMARY
          └─ 1.1 ...
             └─ 1.1.1 ...

3) 번호의 실제 부모 관계를 검사합니다.
   - N.0      -> 문서 제목 바로 아래
   - N.M      -> 같은 N의 N.0 아래
   - N.M.K    -> 같은 N.M 아래
    - N.M.K.L  -> 같은 N.M.K 아래

   즉 2.1이 1.0 아래로 들어가거나, 새 문서의 1.0이 이전 문서의 3.0 아래로
   들어가는 잘못된 구조를 만들지 않습니다.
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import pdfplumber
from pypdf import PdfReader, PdfWriter
from pypdf.generic import Fit


# ---------------------------------------------------------------------------
# 번호 패턴
# ---------------------------------------------------------------------------
# 1.1.1.1 / 2.10.3.4 / 12.3.4.5
LEVEL5_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+)\.?\s+(\S.*)$")
# 1.1.1 / 2.10.3 / 12.3.4
LEVEL4_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.?\s+(\S.*)$")
# 1.0 / 1.1 / 2.0 / 2.10
LEVEL23_RE = re.compile(r"^(\d+)\.(\d+)\.?\s+(\S.*)$")

# 제목 후보에서 제외할 잡음
PAGE_ONLY_RE = re.compile(r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$", re.I)
REV_RE = re.compile(r"^(?:rev(?:ision)?\.?|date|page|document\s+no\.?|doc\.?\s*no\.?)\b", re.I)


def normalize_space(text):
    return re.sub(r"\s+", " ", text or "").strip()


def classify_heading(text):
    """
    절 번호로 시작하는 줄이면 상세 정보를 반환합니다.

    반환 예:
      {
        'level': 2,
        'number': '1.0',
        'parts': (1, 0),
        'title': '1.0 SUMMARY'
      }
    """
    text = normalize_space(text)

    m5 = LEVEL5_RE.match(text)
    if m5:
        a, b, c, d = (int(m5.group(i)) for i in range(1, 5))
        number = f"{a}.{b}.{c}.{d}"
        return {
            "level": 5,
            "number": number,
            "parts": (a, b, c, d),
            "title": f"{number} {m5.group(5)}".strip(),
        }

    m4 = LEVEL4_RE.match(text)
    if m4:
        a, b, c = int(m4.group(1)), int(m4.group(2)), int(m4.group(3))
        number = f"{a}.{b}.{c}"
        return {
            "level": 4,
            "number": number,
            "parts": (a, b, c),
            "title": f"{number} {m4.group(4)}".strip(),
        }

    m23 = LEVEL23_RE.match(text)
    if m23:
        a, b = int(m23.group(1)), int(m23.group(2))
        number = f"{a}.{b}"
        return {
            "level": 2 if b == 0 else 3,
            "number": number,
            "parts": (a, b),
            "title": f"{number} {m23.group(3)}".strip(),
        }

    return None


# ---------------------------------------------------------------------------
# PDF 텍스트를 줄 단위로 추출
# ---------------------------------------------------------------------------
def extract_lines(pdf_path, y_tolerance=3):
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
            if not words:
                continue

            words.sort(key=lambda w: (round(w["top"] / y_tolerance), w["x0"]))

            buckets = defaultdict(list)
            for word in words:
                key = round(word["top"] / y_tolerance)
                buckets[key].append(word)

            for key in sorted(buckets):
                ws = sorted(buckets[key], key=lambda w: w["x0"])
                text = normalize_space(" ".join(w["text"] for w in ws))
                if not text:
                    continue

                lines.append(
                    {
                        "page": page_index,
                        "text": text,
                        "x0": min(w["x0"] for w in ws),
                        "top": min(w["top"] for w in ws),
                        "bottom": max(w["bottom"] for w in ws),
                        "page_width": page.width,
                        "page_height": page.height,
                    }
                )

    return lines


# ---------------------------------------------------------------------------
# 머릿글 / 바닥글 제외
# ---------------------------------------------------------------------------
def detect_header_footer(lines, num_pages, header_ratio, footer_ratio, repeat_threshold=0.5):
    """반복되는 머릿글/바닥글만 제외합니다.

    페이지의 절대적인 상단/하단 위치만으로 제외하면 실제 본문의 절 제목이
    누락될 수 있으므로, 여러 페이지에 반복되는 동일 문자열을 중심으로 판정합니다.
    """
    excluded = set()

    text_position_pages = defaultdict(set)
    for ln in lines:
        norm_text = normalize_space(ln["text"])
        if not norm_text:
            continue
        y_bucket = round(ln["top"] / 5)
        text_position_pages[(norm_text, y_bucket)].add(ln["page"])

    threshold_pages = max(2, int(num_pages * repeat_threshold + 0.9999))
    repeated_keys = {
        key for key, pages in text_position_pages.items()
        if len(pages) >= threshold_pages
    }

    for i, ln in enumerate(lines):
        norm_text = normalize_space(ln["text"])
        y_bucket = round(ln["top"] / 5)
        if (norm_text, y_bucket) in repeated_keys:
            excluded.add(i)

    return excluded


def detect_toc_pages(lines):
    """목차로 보이는 페이지를 찾습니다.

    목차의 번호 자체를 처음부터 버리지 않습니다. 모든 절 번호를 후보로
    인식한 뒤, 최종 단계에서 목차에만 존재하는 후보를 제거합니다.
    """
    by_page = defaultdict(list)
    for i, ln in enumerate(lines):
        by_page[ln["page"]].append((i, ln))

    toc_pages = set()
    for page, items in by_page.items():
        texts = [normalize_space(ln["text"]) for _, ln in items]
        joined = " ".join(texts).lower()

        score = 0
        if re.search(r"\b(?:contents|table\s+of\s+contents)\b", joined, re.I):
            score += 5
        if "목차" in joined:
            score += 5

        dotted = sum(1 for t in texts if re.search(r"\.{3,}|·{3,}|…{2,}", t))
        trailing_page = sum(
            1 for t in texts
            if re.search(r"(?:\.{2,}|·{2,}|…{2,}|\s)\d{1,4}$", t)
        )
        numbered = sum(1 for t in texts if classify_heading(t) is not None)

        if dotted >= 2:
            score += 3
        if trailing_page >= 3:
            score += 2
        if numbered >= 5 and (dotted >= 1 or trailing_page >= 2):
            score += 2

        if score >= 5:
            toc_pages.add(page)

    # 목차가 여러 페이지로 이어지는 경우, 인접 페이지도 목차로 확장합니다.
    changed = True
    while changed:
        changed = False
        for page in list(toc_pages):
            for neighbor in (page - 1, page + 1):
                if neighbor in by_page and neighbor not in toc_pages:
                    texts = [normalize_space(ln["text"]) for _, ln in by_page[neighbor]]
                    numbered = sum(1 for t in texts if classify_heading(t) is not None)
                    dotted = sum(1 for t in texts if re.search(r"\.{3,}|·{3,}|…{2,}", t))
                    trailing_page = sum(
                        1 for t in texts
                        if re.search(r"(?:\.{2,}|·{2,}|…{2,}|\s)\d{1,4}$", t)
                    )
                    if numbered >= 3 and (dotted >= 1 or trailing_page >= 2):
                        toc_pages.add(neighbor)
                        changed = True

    return toc_pages


def filter_toc_headings(headings, toc_pages):
    """목차 후보와 본문 후보를 분리합니다.

    같은 번호가 목차와 본문에 모두 있으면 본문 후보만 남깁니다.
    목차에만 있는 번호는 북마크로 만들지 않습니다.
    즉 목차 페이지를 '인식하지 않는 것'이 아니라 '후보를 인식한 뒤
    본문 여부를 판별하는 것'이 핵심입니다.
    """
    if not toc_pages:
        return headings

    body_numbers = {h["number"] for h in headings if h["page"] not in toc_pages}
    result = []
    for h in headings:
        if h["page"] in toc_pages:
            if h["number"] in body_numbers:
                continue
            # 목차에만 존재하는 번호도 최종 북마크에서는 제외
            continue
        result.append(h)

    return result


# ---------------------------------------------------------------------------
# 헤딩 추출
# ---------------------------------------------------------------------------
def build_headings(lines, excluded, number_zone_ratio):
    """왼쪽 번호 영역에서 시작하는 줄만 절 제목 후보로 인정합니다."""
    headings = []

    for line_index, ln in enumerate(lines):
        if line_index in excluded:
            continue
        if ln["x0"] > ln["page_width"] * number_zone_ratio:
            continue

        info = classify_heading(ln["text"])
        if info is None:
            continue

        headings.append(
            {
                **info,
                "line_index": line_index,
                "page": ln["page"],
                "top": ln["top"],
                "page_height": ln["page_height"],
            }
        )

    return headings


# ---------------------------------------------------------------------------
# 새 문서(1.0) 직전의 제목 찾기
# ---------------------------------------------------------------------------
def is_title_candidate(text):
    text = normalize_space(text)
    if not text:
        return False
    if classify_heading(text):
        return False
    if PAGE_ONLY_RE.match(text):
        return False
    if REV_RE.match(text):
        return False
    if len(text) < 2:
        return False
    return True


def title_score(ln, first_heading):
    """
    1.0에 가까울수록 우선하되, 너무 짧은 잡음보다 제목다운 문자열을 선호합니다.
    같은 페이지 또는 바로 앞 페이지의 제목을 강하게 우선합니다.
    """
    text = normalize_space(ln["text"])
    page_gap = first_heading["page"] - ln["page"]

    score = 0.0

    # 1.0과 가까운 페이지에 있는 줄 우선
    if page_gap == 0:
        score += 80
    elif page_gap == 1:
        score += 55
    elif page_gap == 2:
        score += 20
    else:
        score -= page_gap * 10

    # Appendix / Annex / Attachment 같은 문서명 표기는 강하게 우선
    if re.search(r"\b(?:appendix|annex|attachment|enclosure)\b", text, re.I):
        score += 100

    # 적당한 길이의 제목 선호
    if 4 <= len(text) <= 120:
        score += 20
    if len(text) > 180:
        score -= 30

    # 문서 상단/중앙 쪽의 제목을 조금 더 선호
    rel_top = ln["top"] / max(ln["page_height"], 1)
    if 0.10 <= rel_top <= 0.65:
        score += 10

    # 일반 본문 문장처럼 너무 긴 문장부호가 많으면 감점
    if text.count(".") + text.count(",") + text.count(";") >= 4:
        score -= 15

    return score


def detect_document_title(lines, excluded, start_index, first_heading, fallback_title):
    """
    현재 문서의 1.0 이전 구간에서 최상위 문서 제목을 찾습니다.

    start_index는 이전 문서의 마지막 헤딩 다음 위치에 해당하며, 그 지점부터
    현재 1.0 직전까지만 검색하므로 이전 문서의 제목을 다시 잡지 않습니다.
    """
    end_index = first_heading["line_index"]
    candidates = []

    for i in range(max(0, start_index), end_index):
        if i in excluded:
            continue
        ln = lines[i]
        if not is_title_candidate(ln["text"]):
            continue

        # 너무 멀리 떨어진 옛 페이지는 제목 후보에서 사실상 제외
        if first_heading["page"] - ln["page"] > 3:
            continue

        candidates.append((title_score(ln, first_heading), i, ln))

    if not candidates:
        return {
            "title": fallback_title,
            "page": first_heading["page"],
            "top": 0,
            "page_height": first_heading["page_height"],
            "line_index": first_heading["line_index"],
        }

    # 점수 우선, 동점이면 1.0에 더 가까운(뒤쪽) 줄 우선
    _, idx, best = max(candidates, key=lambda x: (x[0], x[1]))
    return {
        "title": normalize_space(best["text"]),
        "page": best["page"],
        "top": best["top"],
        "page_height": best["page_height"],
        "line_index": idx,
    }


def get_pdf_fallback_title(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        if reader.metadata and reader.metadata.title:
            title = normalize_space(reader.metadata.title)
            if title:
                return title
    except Exception:
        pass

    return os.path.splitext(os.path.basename(pdf_path))[0]


# ---------------------------------------------------------------------------
# 1.0 기준으로 PDF 내부의 여러 문서를 분리
# ---------------------------------------------------------------------------
def split_into_documents(lines, excluded, headings, pdf_path):
    """
    1.0이 나올 때마다 새 문서를 시작합니다.

    중요:
    - 2.0, 3.0 등은 새 문서 시작이 아닙니다.
    - 오직 정확히 1.0만 새 문서 시작점입니다.
    """
    fallback = get_pdf_fallback_title(pdf_path)
    starts = [i for i, h in enumerate(headings) if h["parts"] == (1, 0)]

    if not starts:
        # 1.0이 없는 PDF는 기존처럼 하나의 문서로 처리하되 파일명/메타 제목을 사용
        return [
            {
                "title_info": {
                    "title": fallback,
                    "page": 0,
                    "top": 0,
                    "page_height": headings[0]["page_height"] if headings else 792,
                    "line_index": 0,
                },
                "headings": headings,
            }
        ]

    documents = []

    for doc_no, heading_start_pos in enumerate(starts):
        heading_end_pos = starts[doc_no + 1] if doc_no + 1 < len(starts) else len(headings)
        doc_headings = headings[heading_start_pos:heading_end_pos]
        first_heading = doc_headings[0]

        # 제목 검색 범위 시작점:
        # 첫 문서는 파일 처음부터, 다음 문서는 이전 문서의 마지막 인식 헤딩 뒤부터.
        if doc_no == 0:
            title_search_start = 0
        else:
            prev_heading_start = starts[doc_no - 1]
            prev_heading_end = heading_start_pos
            prev_headings = headings[prev_heading_start:prev_heading_end]
            if prev_headings:
                title_search_start = prev_headings[-1]["line_index"] + 1
            else:
                title_search_start = 0

        title_info = detect_document_title(
            lines,
            excluded,
            title_search_start,
            first_heading,
            fallback_title=f"{fallback} - Part {doc_no + 1}",
        )

        documents.append(
            {
                "title_info": title_info,
                "headings": doc_headings,
            }
        )

    return documents


# ---------------------------------------------------------------------------
# 계층 검증 + PDF 북마크 쓰기
# ---------------------------------------------------------------------------
def write_bookmarks(pdf_path, output_path, documents):
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    writer.append(reader)

    top_margin_pt = 12

    for doc in documents:
        title_info = doc["title_info"]
        headings = doc["headings"]

        title_y = max(title_info["page_height"] - title_info["top"] + top_margin_pt, 0)
        root = writer.add_outline_item(
            title=title_info["title"],
            page_number=title_info["page"],
            fit=Fit.xyz(left=0, top=title_y, zoom=None),
        )

        # 번호를 키로 부모를 저장합니다.
        # 예: section_roots[1] = 1.0 북마크
        #     subsections[(1, 1)] = 1.1 북마크
        section_roots = {}
        subsections = {}

        for h in headings:
            y = max(h["page_height"] - h["top"] + top_margin_pt, 0)
            fit = Fit.xyz(left=0, top=y, zoom=None)
            parts = h["parts"]

            if h["level"] == 2:
                # N.0은 항상 문서 제목 바로 아래
                major = parts[0]
                parent = root
                item = writer.add_outline_item(
                    title=h["title"],
                    page_number=h["page"],
                    parent=parent,
                    fit=fit,
                )
                section_roots[major] = item

                # 같은 major 번호의 예전 하위 항목은 초기화
                for key in list(subsections):
                    if key[0] == major:
                        del subsections[key]

            elif h["level"] == 3:
                # N.M은 반드시 같은 N의 N.0 밑에만 둡니다.
                major, minor = parts
                parent = section_roots.get(major)

                # 부모 N.0이 실제로 인식되지 않았다면 잘못된 다른 번호 밑에 붙이지 않고
                # 문서 제목 바로 아래에 둡니다.
                if parent is None:
                    parent = root

                item = writer.add_outline_item(
                    title=h["title"],
                    page_number=h["page"],
                    parent=parent,
                    fit=fit,
                )
                subsections[(major, minor)] = item

            elif h["level"] == 4:  # level 4: N.M.K
                major, minor, _ = parts

                parent = subsections.get((major, minor))
                if parent is None:
                    parent = section_roots.get(major)
                if parent is None:
                    parent = root

                item = writer.add_outline_item(
                    title=h["title"],
                    page_number=h["page"],
                    parent=parent,
                    fit=fit,
                )
                # 4단계(N.M.K)의 북마크를 5단계(N.M.K.L)의 부모로 사용
                subsections[(major, minor, parts[2])] = item

            else:  # level 5: N.M.K.L
                major, minor, third, _ = parts

                # N.M.K.L은 같은 N.M.K 아래.
                parent = subsections.get((major, minor, third))
                if parent is None:
                    # 1.1.1이 인식되지 않은 경우 1.1 -> 1.0 -> 문서 제목 순으로 안전하게 승격
                    parent = subsections.get((major, minor))
                if parent is None:
                    parent = section_roots.get(major)
                if parent is None:
                    parent = root

                writer.add_outline_item(
                    title=h["title"],
                    page_number=h["page"],
                    parent=parent,
                    fit=fit,
                )

    with open(output_path, "wb") as f:
        writer.write(f)


# ---------------------------------------------------------------------------
# 파일 한 개 처리
# ---------------------------------------------------------------------------
def process_pdf(pdf_path, output_dir, number_zone_ratio, header_ratio, footer_ratio, dry_run=False):
    lines = extract_lines(pdf_path)
    if not lines:
        print(f"  [건너뜀] 텍스트를 추출할 수 없습니다: {pdf_path}")
        return

    num_pages = max(ln["page"] for ln in lines) + 1
    excluded = detect_header_footer(lines, num_pages, header_ratio, footer_ratio)

    # 1차: 목차까지 포함해 모든 번호를 후보로 인식합니다.
    all_headings = build_headings(lines, excluded, number_zone_ratio)
    toc_pages = detect_toc_pages(lines)

    # 2차: 목차 후보와 본문 후보를 비교하여 본문 번호를 최종 채택합니다.
    headings = filter_toc_headings(all_headings, toc_pages)

    documents = split_into_documents(lines, excluded, headings, pdf_path)

    print(f"\n[PDF] {os.path.basename(pdf_path)}")
    print(f"  인식된 내부 문서 수: {len(documents)}")

    for doc_index, doc in enumerate(documents, 1):
        title = doc["title_info"]["title"]
        print(f"\n  [문서 {doc_index}] {title}")

        for h in doc["headings"]:
            if h["level"] == 2:
                indent = "    "
            elif h["level"] == 3:
                indent = "      "
            elif h["level"] == 4:
                indent = "        "
            else:
                indent = "          "
            print(f"{indent}p.{h['page'] + 1}  {h['title']}")

    if dry_run:
        return

    if not headings:
        print("  [알림] 인식된 절 번호가 없습니다. --number-zone-ratio 값을 조정해 보세요.")

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{base}_bookmarked.pdf")

    # 결과 파일이 이미 열려 있거나 존재하더라도 원본/기존 결과를 덮어쓰지 않습니다.
    # 같은 이름의 결과가 있으면 _2, _3 ... 로 새 파일을 만듭니다.
    if os.path.exists(output_path):
        n = 2
        while True:
            candidate = os.path.join(output_dir, f"{base}_bookmarked_{n}.pdf")
            if not os.path.exists(candidate):
                output_path = candidate
                break
            n += 1

    write_bookmarks(pdf_path, output_path, documents)
    print(f"\n  -> 저장됨: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def collect_pdfs(input_path):
    # 단일 PDF만 처리합니다. CMD에서 PDF를 드래그하여 경로를 붙여넣는 방식입니다.
    return [input_path] if os.path.isfile(input_path) and input_path.lower().endswith(".pdf") else []


def main():
    parser = argparse.ArgumentParser(
        description="PDF 내부의 반복되는 1.0을 문서 시작점으로 인식해 계층형 북마크를 생성합니다."
    )
    parser.add_argument("input", nargs="?", default=None, help="PDF 파일 경로")
    parser.add_argument("--output-dir", default=None, help="결과 저장 폴더 (기본: 입력 파일과 같은 위치)")
    parser.add_argument(
        "--number-zone-ratio",
        type=float,
        default=0.15,
        help="절 번호가 시작될 수 있는 왼쪽 영역 폭 비율 (기본 0.15)",
    )
    parser.add_argument(
        "--header-ratio",
        type=float,
        default=0.10,
        help="상단 머릿글 제외 비율 (기본 0.10)",
    )
    parser.add_argument(
        "--footer-ratio",
        type=float,
        default=0.08,
        help="하단 바닥글 제외 비율 (기본 0.08)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 PDF를 저장하지 않고 인식 구조만 출력",
    )
    args = parser.parse_args()

    interactive = args.input is None
    if interactive:
        raw = input(
            "PDF 파일 경로를 입력하세요\n"
            "(따옴표째로 붙여넣어도 됩니다): "
        ).strip()
        args.input = raw.strip('"').strip("'")

    if not args.input:
        print("입력 경로가 없습니다. 프로그램을 종료합니다.")
        if interactive:
            input("Enter 키를 누르면 종료합니다...")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"경로를 찾을 수 없습니다: {args.input}")
        if interactive:
            input("Enter 키를 누르면 종료합니다...")
        sys.exit(1)

    pdfs = collect_pdfs(args.input)
    if not pdfs:
        print("처리할 PDF 파일을 찾지 못했습니다.")
        if interactive:
            input("Enter 키를 누르면 종료합니다...")
        sys.exit(1)

    output_dir = args.output_dir or (
        args.input if os.path.isdir(args.input)
        else os.path.dirname(os.path.abspath(args.input))
    )

    try:
        for pdf_path in pdfs:
            process_pdf(
                pdf_path,
                output_dir,
                args.number_zone_ratio,
                args.header_ratio,
                args.footer_ratio,
                dry_run=args.dry_run,
            )
    except Exception as e:
        print(f"\n[오류] 처리 중 문제가 발생했습니다: {e}")
        if interactive:
            input("Enter 키를 누르면 종료합니다...")
        sys.exit(1)

    if interactive:
        try:
            input("\n완료되었습니다. Enter 키를 누르면 창을 닫습니다...")
        except EOFError:
            pass


if __name__ == "__main__":
    main()
