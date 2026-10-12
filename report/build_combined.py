#!/usr/bin/env python3
"""Build the combined report (Part I full + Part II full + Methods) from the three standalone sources.

The parts are not edited by hand in the combined file: this script copies each body, prefixes its labels and references
(p1:, p2:, m:) so they stay unique, restarts section numbering per part (so "Part I, Sec. III" in Part II stays valid),
and merges the three bibliographies (a key cited by several parts keeps Part I's wording). Run from this directory, then build with tectonic:
    python3 build_combined.py && cd archive && tectonic combined_en.tex && tectonic combined_ko.tex
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

PREAMBLE = {
    "en": r"""\documentclass[conference,10pt]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage[T1]{fontenc}
""",
    "ko": r"""\documentclass[conference,10pt]{IEEEtran}
\IEEEoverridecommandlockouts
""",
}
COMMON = r"""\usepackage{graphicx,booktabs,amsmath,amssymb,multirow,xcolor,url,xspace}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={@PDFTITLE@}, pdfauthor={Jeong Hwan Lee}}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,fit,backgrounds}
@FONTS@\newcommand{\mm}{\,mm\xspace}
\newcommand{\cmt}[1]{}
\def\UrlBreaks{\do\/\do\_\do\.\do\-}\def\UrlFont{\small\ttfamily}\newcommand{\code}[1]{\path{#1}}
% Each part restarts its section numbering; hyperref anchors carry the part number so they stay unique.
\newcounter{rpart}
\renewcommand{\theHsection}{\arabic{rpart}.\arabic{section}}
\renewcommand{\theHsubsection}{\arabic{rpart}.\arabic{section}.\arabic{subsection}}
\newcommand{\reportpart}[3]{\clearpage\stepcounter{rpart}\setcounter{section}{0}%
  \twocolumn[{\centering{\LARGE\bfseries #1\par}\vspace{1.5mm}{\large #2\par}\vspace{3mm}\hrule\vspace{5mm}}]\label{#3}}
\newcommand{\partabstract}[1]{\noindent\textbf{\textit{@ABSLABEL@}}---\textit{#1}\par\medskip}
"""
FONTS_KO = (r"\usepackage{fontspec}\setmainfont{Times New Roman}\usepackage{xeCJK}"
            r"\setCJKmainfont[AutoFakeBold=2.5]{AppleMyungjo}\setCJKsansfont[AutoFakeBold=2.5]{Apple SD Gothic Neo}"
            r"\setCJKmonofont{Apple SD Gothic Neo}\xeCJKsetup{CJKecglue={},CJKspace=true}" + "\n")

PARTS = [  # (prefix, source, part heading, subtitle) per language
    ("p1", "archive/part1_stacking_%s_full.tex"),
    ("p2", "archive/part2_approach_%s_full.tex"),
    ("m", "methods_%s.tex"),
]
TEXT = {
    "en": dict(
        pdftitle="Egocentric Human Demonstrations for a Bimanual Robot: Combined Report",
        abslabel="Abstract",
        title=r"Egocentric Human Demonstrations for a Bimanual Robot\\[1mm]\Large Combined Report: Part I, Part II and Methods",
        author=r"""\author{\IEEEauthorblockN{Jeong Hwan Lee}
\IEEEauthorblockA{Technical report, version 4.1, 12 October 2026 \quad alnrg2arg@gmail.com\\
Code, results and sample data are available on request.}}""",
        heads=[("Part I", "Three-Cube Stacking (bimanual)"),
               ("Part II", "Single-Arm Approach to a Cube"),
               ("Methods and Pipeline", "How each stage works (companion to Parts I and II)")],
        front=r"""\begin{abstract}
The capture device, HandUMI, is an open-source design by BrikHMP18 (murobotics-ai/handumi-hw), not our design.
This document combines the three reports of version 4.1 into one file, unchanged in content: Part I (full length) on 349
bimanual egocentric HandUMI demonstrations of three-cube stacking, the capture hardware, the processing funnel, X-VLA
training, offline evaluation and transfer to the reBot B601; Part II (full length) on single-arm approach data, object-based
scale, robot-side calibration, a shared task frame and deployment settings; and the Methods document, which describes how
every stage of the pipeline works rather than its results. Each part keeps its own abstract, section numbering and
figures; references such as ``Part I, Sec.~III'' refer to the section numbers printed in that part. The bibliography of the
three parts is merged at the end.
\end{abstract}

\noindent\textbf{Contents.}
\begin{itemize}
\item Part I: Three-Cube Stacking \dotfill p.~\pageref{part:p1}
\item Part II: Single-Arm Approach to a Cube \dotfill p.~\pageref{part:p2}
\item Methods and Pipeline \dotfill p.~\pageref{part:m}
\item References \dotfill p.~\pageref{part:refs}
\end{itemize}
""",
        refs="References",
    ),
    "ko": dict(
        pdftitle="로봇 양팔을 위한 에고센트릭 인간 시연: 통합본",
        abslabel="초록",
        title=r"로봇 양팔을 위한 에고센트릭 인간 시연\\[1mm]\Large 통합본: 제1부, 제2부, 방법과 파이프라인",
        author=r"""\author{\IEEEauthorblockN{Jeong Hwan Lee}
\IEEEauthorblockA{기술 보고서, 버전 4.1, 2026년 10월 12일 \quad alnrg2arg@gmail.com\\
코드, 결과, 샘플 데이터는 요청 시 제공합니다.}}""",
        heads=[("제1부", "큐브 삼단 쌓기(양팔)"),
               ("제2부", "단일 팔 큐브 접근(Approach)"),
               ("방법과 파이프라인", "각 단계의 작동 방식(제1부·제2부의 동반 문서)")],
        front=r"""\begin{abstract}
수집 장치 HandUMI는 BrikHMP18의 오픈소스 설계(murobotics-ai/handumi-hw)이며, 우리가 설계한 장치가 아니다.
이 문서는 버전 4.1의 보고서 세 편을 내용 변경 없이 하나로 묶은 것이다. 제1부(전체 버전)는 큐브 삼단 쌓기의 양팔 에고센트릭
HandUMI 시연 349개, 수집 하드웨어, 처리 퍼널, X-VLA 학습, 오프라인 평가, reBot B601로의 전이를 다룬다. 제2부(전체 버전)는 단일 팔
접근 데이터, 물체 기반 스케일, 로봇 측 캘리브레이션, 공유 작업 좌표계, 배포 설정을 다룬다. 방법과 파이프라인 문서는 각 단계의 결과가
아니라 작동 방식을 기술한다. 각 부는 자체 초록, 절 번호, 그림을 유지하며, ``제1부 III절''과 같은 참조는 해당 부에 인쇄된 절 번호를
가리킨다. 세 부의 참고문헌은 끝에 하나로 합쳤다.
\end{abstract}

\noindent\textbf{목차.}
\begin{itemize}
\item 제1부: 큐브 삼단 쌓기 \dotfill p.~\pageref{part:p1}
\item 제2부: 단일 팔 큐브 접근 \dotfill p.~\pageref{part:p2}
\item 방법과 파이프라인 \dotfill p.~\pageref{part:m}
\item 참고문헌 \dotfill p.~\pageref{part:refs}
\end{itemize}
""",
        refs="참고문헌",
    ),
}


def split(src):
    """Return (abstract, body, {key: bibitem text}) of one standalone report."""
    s = src.split(r"\maketitle", 1)[1].split(r"\end{document}", 1)[0]
    abstract, rest = re.search(r"\\begin\{abstract\}\n?(.*?)\\end\{abstract\}(.*)", s, re.S).groups()
    body, bib = rest.split(r"\begin{thebibliography}", 1)
    bib = bib.split(r"\end{thebibliography}", 1)[0]
    items = {}
    for m in re.finditer(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem\{|\Z)", bib, re.S):
        items[m.group(1)] = m.group(2).strip()
    return abstract.strip(), body.strip(), items


def prefix(body, p):
    body = re.sub(r"\\label\{([^}]+)\}", lambda m: r"\label{%s:%s}" % (p, m.group(1)), body)
    return re.sub(r"\\(ref|pageref|eqref)\{([^}]+)\}", lambda m: r"\%s{%s:%s}" % (m.group(1), p, m.group(2)), body)


def build(lang):
    t = TEXT[lang]
    out = [PREAMBLE[lang] + COMMON.replace("@PDFTITLE@", t["pdftitle"]).replace("@FONTS@", FONTS_KO if lang == "ko" else "").replace("@ABSLABEL@", t["abslabel"]),
           r"\begin{document}", r"\title{%s}" % t["title"], t["author"], r"\maketitle", t["front"]]
    bib, order = {}, []
    for (p, path), (head, sub) in zip(PARTS, t["heads"]):
        abstract, body, items = split((HERE / (path % lang)).read_text())
        out += [r"\reportpart{%s}{%s}{part:%s}" % (head, sub, p), r"\partabstract{%s}" % abstract.replace("\n\n", " "), prefix(body, p)]
        for k, v in items.items():
            if k not in bib:  # same key = same work; the first (Part I, most complete) wording is kept
                bib[k] = v
                order.append(k)
    out += [r"\clearpage\renewcommand{\refname}{%s}" % t["refs"], r"\phantomsection\label{part:refs}",
            r"\begin{thebibliography}{99}\footnotesize"]
    out += [r"\bibitem{%s} %s" % (k, bib[k]) for k in order]
    out += [r"\end{thebibliography}", r"\end{document}", ""]
    dst = HERE / "archive" / f"combined_{lang}.tex"
    dst.write_text("\n".join(out))
    print(dst, len(order), "references")


for lang in ("en", "ko"):
    build(lang)
