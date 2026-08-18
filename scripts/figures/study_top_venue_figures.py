#!/usr/bin/env python3
"""Build a reproducible visual-style corpus from top ML systems papers.

The script downloads open paper PDFs, locates the first Figure 1 caption,
renders that page, extracts figure-caption lines, and builds contact sheets for
human visual review. It does not copy any source figure into the manuscript.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont, ImageStat


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "paper" / "mlsys2026" / "build" / "style-corpus"


@dataclass(frozen=True)
class Paper:
    key: str
    title: str
    venue: str
    year: int
    pdf_url: str
    landing_url: str


def mlsys(
    key: str,
    title: str,
    year: int,
    digest: str,
) -> Paper:
    base = f"https://proceedings.mlsys.org/paper_files/paper/{year}"
    return Paper(
        key=key,
        title=title,
        venue="MLSys",
        year=year,
        pdf_url=f"{base}/file/{digest}-Paper-Conference.pdf",
        landing_url=f"{base}/hash/{digest}-Abstract-Conference.html",
    )


PAPERS = [
    mlsys("mlsys25_leanattention", "LeanAttention: Hardware-Aware Scalable Attention Mechanism for the Decode-Phase of Transformers", 2025, "16ec6494e9b5a4138de7238761d715b4"),
    mlsys("mlsys25_kv_compression", "Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving", 2025, "26289c647c6828e862e271ca3c490486"),
    mlsys("mlsys25_sampleattention", "SampleAttention: Near-Lossless Acceleration of Long Context LLM Inference with Adaptive Structured Sparse Attention", 2025, "2d04d97593c8c33d415337f408ed0e1b"),
    mlsys("mlsys25_neo", "NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference", 2025, "66a026c0d17040889b50f0dfa650e5e0"),
    mlsys("mlsys25_flexinfer", "FlexInfer: Flexible LLM Inference with CPU Computations", 2025, "698cfaf72a208aef2e78bcac55b74328"),
    mlsys("mlsys25_context_parallelism", "Context Parallelism for Scalable Million-Token Inference", 2025, "78834433edc3291f4c6cbbd2759324db"),
    mlsys("mlsys25_marconi", "Marconi: Prefix Caching for the Era of Hybrid LLMs", 2025, "7c180af017258d239bac6248d1eb26ac"),
    mlsys("mlsys25_sola", "SOLA: Optimizing SLO Attainment for Large Language Model Serving with State-Aware Scheduling", 2025, "bc82dbfbfa43232be85b8d9838f49c3e"),
    mlsys("mlsys25_thunderserve", "ThunderServe: High-performance and Cost-efficient LLM Serving in Cloud Environments", 2025, "c2a0e26dd9ee7d57e92bb1c24b39659a"),
    mlsys("mlsys25_seesaw", "Seesaw: High-throughput LLM Inference via Model Re-sharding", 2025, "cbc4ab80cd77aa0eb87da062fbcddb46"),
    mlsys("mlsys25_lserve", "LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention", 2025, "cc8c6b9d89f7a898a29f58869b238e46"),
    mlsys("mlsys25_flashinfer", "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving", 2025, "dbf02b21d77409a2db30e56866a8ab3a"),
    mlsys("mlsys25_qserve", "QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving", 2025, "fbe2b2f74a2ece8070d8fb073717bda6"),
    mlsys("mlsys25_fasttree", "FastTree: Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference", 2025, "96894468eb44631a32d7ebd56f9892c7"),
    mlsys("mlsys25_lumos", "Lumos: Efficient Performance Modeling and Estimation for Large-scale LLM Training", 2025, "a66caa1703fe34705a4368c3014c1966"),
    mlsys("mlsys25_turboattention", "TurboAttention: Efficient Attention Approximation for High-throughput LLM Inference", 2025, "f4f55846501f3336f293fd8b6de10770"),
    mlsys("mlsys24_punica", "Punica: Multi-Tenant LoRA Serving", 2024, "054de805fcceb78a201f5e9d53c85908"),
    mlsys("mlsys24_awq", "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration", 2024, "42a452cbafa9dd64e9ba4aa95cc1ef21"),
    mlsys("mlsys24_keyformer", "Keyformer: KV Cache Reduction through Key Token Selection for Efficient Generative Inference", 2024, "48fecef47b19fe501d27d338b6d52582"),
    mlsys("mlsys24_flashdecoding", "FlashDecoding++: Faster Large Language Model Inference with Asynchronization, Flat GEMM Optimization, and Heuristics", 2024, "5321b1dabcd2be188d796c21b733e8c7"),
    mlsys("mlsys24_hetegen", "HeteGen: Efficient Heterogeneous Parallel Inference for Large Language Models on Resource-Constrained Devices", 2024, "5431dca75a8d2abc1fb51e89e8324f10"),
    mlsys("mlsys24_atom", "Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving", 2024, "5edb57c05c81d04beb716ef1d542fe9e"),
    mlsys("mlsys24_sida", "SiDA: Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models", 2024, "698cfaf72a208aef2e78bcac55b74328"),
    mlsys("mlsys24_slora", "SLoRA: Scalable Serving of Thousands of LoRA Adapters", 2024, "906419cd502575b617cc489a1a696a67"),
    mlsys("mlsys24_prompt_cache", "Prompt Cache: Modular Attention Reuse for Low-Latency Inference", 2024, "a66caa1703fe34705a4368c3014c1966"),
    mlsys("mlsys24_vidur", "Vidur: A Large-Scale Simulation Framework for LLM Inference", 2024, "b74a8de47d2b3c928360e0a011f48351"),
    mlsys("mlsys24_q_hitter", "Q-Hitter: A Better Token Oracle for Efficient LLM Inference via Sparse-Quantized KV Cache", 2024, "bbb7506579431a85861a05fff048d3e1"),
    mlsys("mlsys24_acrobat", "ACROBAT: Optimizing Auto-batching of Dynamic Deep Learning at Compile Time", 2024, "096b1019463f34eb241e87cfce8dfe16"),
    Paper("sosp23_vllm", "Efficient Memory Management for Large Language Model Serving with PagedAttention", "SOSP", 2023, "https://arxiv.org/pdf/2309.06180", "https://doi.org/10.1145/3600006.3613165"),
    Paper("osdi22_orca", "Orca: A Distributed Serving System for Transformer-Based Generative Models", "OSDI", 2022, "https://www.usenix.org/system/files/osdi22-yu.pdf", "https://www.usenix.org/conference/osdi22/presentation/yu"),
    Paper("osdi22_alpa", "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning", "OSDI", 2022, "https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf", "https://www.usenix.org/conference/osdi22/presentation/zheng-lianmin"),
    Paper("osdi24_distserve", "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving", "OSDI", 2024, "https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf", "https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin"),
    Paper("osdi24_llumnix", "Llumnix: Dynamic Scheduling for Large Language Model Serving", "OSDI", 2024, "https://www.usenix.org/system/files/osdi24-sun-biao.pdf", "https://www.usenix.org/conference/osdi24/presentation/sun-biao"),
    Paper("osdi24_sarathi", "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", "OSDI", 2024, "https://www.usenix.org/system/files/osdi24-agrawal.pdf", "https://www.usenix.org/conference/osdi24/presentation/agrawal"),
    Paper("osdi24_infinigen", "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management", "OSDI", 2024, "https://www.usenix.org/system/files/osdi24-lee.pdf", "https://www.usenix.org/conference/osdi24/presentation/lee"),
    Paper("icml23_flexgen", "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU", "ICML", 2023, "https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf", "https://proceedings.mlr.press/v202/sheng23a.html"),
    Paper("isca24_splitwise", "Splitwise: Efficient Generative LLM Inference Using Phase Splitting", "ISCA", 2024, "https://arxiv.org/pdf/2311.18677", "https://doi.org/10.1109/ISCA59077.2024.00019"),
    Paper("asplos24_exegpt", "ExeGPT: Constraint-Aware Resource Scheduling for LLM Inference", "ASPLOS", 2024, "https://arxiv.org/pdf/2404.07947", "https://doi.org/10.1145/3620665.3640383"),
    Paper("atc23_fastserve", "Fast Distributed Inference Serving for Large Language Models", "USENIX ATC", 2023, "https://arxiv.org/pdf/2305.05920", "https://doi.org/10.48550/arXiv.2305.05920"),
    Paper("asplos24_gmlake", "GMLake: Efficient and Transparent GPU Memory Defragmentation for Large-scale DNN Training with Virtual Memory Stitching", "ASPLOS", 2024, "https://arxiv.org/pdf/2401.08156", "https://doi.org/10.1145/3620665.3640423"),
]


FIGURE_ONE_RE = re.compile(r"(?im)^\s*(?:figure|fig\.)\s*1\s*[\.:]")
CAPTION_RE = re.compile(r"(?im)^\s*(?:figure|fig\.)\s*(\d+)\s*[\.:]\s*(.+)$")
SYSTEM_WORDS = re.compile(
    r"\b(?:overview|architecture|workflow|design|pipeline|system|framework|execution|serving)\b",
    re.IGNORECASE,
)
QUANT_WORDS = re.compile(
    r"\b(?:throughput|latency|goodput|memory|speedup|accuracy|perplexity|ablation|breakdown|cdf|utilization|scalability)\b",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(paper: Paper, pdf_dir: Path, refresh: bool) -> Path:
    target = pdf_dir / f"{paper.key}.pdf"
    if target.exists() and target.stat().st_size > 1024 and not refresh:
        return target

    request = urllib.request.Request(
        paper.pdf_url,
        headers={"User-Agent": "Mozilla/5.0 (figure-style-audit; academic use)"},
    )
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                payload = response.read()
            if not payload.startswith(b"%PDF"):
                raise ValueError(f"non-PDF response ({len(payload)} bytes)")
            target.write_bytes(payload)
            return target
        except (OSError, urllib.error.URLError, ValueError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"download failed for {paper.key}: {last_error}")


def first_figure_page(document: fitz.Document) -> int:
    for index, page in enumerate(document):
        if FIGURE_ONE_RE.search(page.get_text("text")):
            return index
    return 0


def caption_lines(document: fitz.Document) -> list[dict[str, object]]:
    captions: list[dict[str, object]] = []
    for page_index, page in enumerate(document):
        text = page.get_text("text")
        for match in CAPTION_RE.finditer(text):
            line = re.sub(r"\s+", " ", match.group(2)).strip()
            captions.append(
                {
                    "figure": int(match.group(1)),
                    "page": page_index + 1,
                    "caption_start": line[:320],
                }
            )
    return captions


def render_page(document: fitz.Document, page_index: int, target: Path) -> None:
    page = document[page_index]
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
    pixmap.save(target)


def image_metrics(path: Path) -> dict[str, float]:
    with Image.open(path).convert("RGB") as image:
        thumbnail = image.copy()
        thumbnail.thumbnail((420, 560))
        pixels = list(thumbnail.getdata())
        total = max(1, len(pixels))
        white = sum(1 for r, g, b in pixels if r > 245 and g > 245 and b > 245)
        colored = sum(
            1
            for r, g, b in pixels
            if max(r, g, b) - min(r, g, b) > 24 and min(r, g, b) < 235
        )
        gray = ImageStat.Stat(thumbnail.convert("L")).mean[0]
        return {
            "white_fraction": round(white / total, 4),
            "color_fraction": round(colored / total, 4),
            "mean_luminance": round(gray / 255.0, 4),
        }


def inspect_paper(paper: Paper, pdf_path: Path, page_dir: Path) -> dict[str, object]:
    document = fitz.open(pdf_path)
    page_index = first_figure_page(document)
    preview_path = page_dir / f"{paper.key}_figure1_page.png"
    render_page(document, page_index, preview_path)
    captions = caption_lines(document)
    caption_text = " ".join(str(row["caption_start"]) for row in captions)
    result = {
        "key": paper.key,
        "title": paper.title,
        "venue": paper.venue,
        "year": paper.year,
        "pdf_url": paper.pdf_url,
        "landing_url": paper.landing_url,
        "pdf_sha256": sha256(pdf_path),
        "pdf_bytes": pdf_path.stat().st_size,
        "pages": document.page_count,
        "figure_1_page": page_index + 1,
        "captions_found": len(captions),
        "captions": captions,
        "caption_signals": {
            "system_or_architecture": len(SYSTEM_WORDS.findall(caption_text)),
            "quantitative": len(QUANT_WORDS.findall(caption_text)),
        },
        "preview": str(preview_path.relative_to(DEFAULT_OUTPUT)),
        "preview_metrics": image_metrics(preview_path),
    }
    document.close()
    return result


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def make_contact_sheets(records: list[dict[str, object]], output_dir: Path) -> list[str]:
    columns = 4
    rows = 2
    cell_width = 560
    cell_height = 760
    label_height = 52
    margin = 18
    font = load_font(17)
    sheets: list[str] = []

    for sheet_index in range(0, len(records), columns * rows):
        batch = records[sheet_index : sheet_index + columns * rows]
        sheet = Image.new(
            "RGB",
            (columns * cell_width + 2 * margin, rows * cell_height + 2 * margin),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for local_index, record in enumerate(batch):
            row, column = divmod(local_index, columns)
            x = margin + column * cell_width
            y = margin + row * cell_height
            preview_path = output_dir / str(record["preview"])
            with Image.open(preview_path).convert("RGB") as preview:
                preview.thumbnail((cell_width - 20, cell_height - label_height - 20))
                image_x = x + (cell_width - preview.width) // 2
                image_y = y + label_height
                sheet.paste(preview, (image_x, image_y))
            corpus_number = sheet_index + local_index + 1
            label = f"[{corpus_number:02d}] {record['venue']} {record['year']} | {record['key']} | p.{record['figure_1_page']}"
            draw.text((x + 8, y + 8), label, fill="#111827", font=font)
            draw.rectangle(
                (x, y, x + cell_width - 8, y + cell_height - 8),
                outline="#CBD5E1",
                width=1,
            )
        filename = f"contact_{sheet_index // (columns * rows) + 1:02d}.png"
        sheet.save(output_dir / filename, optimize=True)
        sheets.append(filename)
    return sheets


def write_markdown(records: list[dict[str, object]], sheets: list[str], target: Path) -> None:
    lines = [
        "# Top-venue scientific-figure style corpus",
        "",
        f"Papers successfully inspected: **{len(records)}**.",
        "",
        "The rendered pages are used only for private style review; no source figure is copied into the manuscript.",
        "",
        "## Contact sheets",
        "",
    ]
    lines.extend(f"- `{sheet}`" for sheet in sheets)
    lines.extend(
        [
            "",
            "## Corpus",
            "",
            "| ID | Venue | Year | Paper | Fig. 1 page | Captions | Color fraction | Source |",
            "|---:|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for index, record in enumerate(records, start=1):
        title = str(record["title"]).replace("|", "\\|")
        lines.append(
            f"| {index} | {record['venue']} | {record['year']} | {title} | "
            f"{record['figure_1_page']} | {record['captions_found']} | "
            f"{record['preview_metrics']['color_fraction']:.4f} | "
            f"[official/open PDF]({record['pdf_url']}) |"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=len(PAPERS))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-key", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    pdf_dir = output_dir / "pdfs"
    page_dir = output_dir / "figure-pages"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)

    papers = [
        paper
        for paper in PAPERS[: max(0, min(args.limit, len(PAPERS)))]
        if paper.key not in set(args.skip_key)
    ]
    downloaded: dict[str, Path] = {}
    failures: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download, paper, pdf_dir, args.refresh): paper for paper in papers
        }
        for future in concurrent.futures.as_completed(futures):
            paper = futures[future]
            try:
                downloaded[paper.key] = future.result()
                print(f"downloaded {paper.key}", flush=True)
            except Exception as exc:  # Continue with the rest of the corpus.
                failures.append({"key": paper.key, "error": str(exc)})
                print(f"FAILED {paper.key}: {exc}", flush=True)

    records: list[dict[str, object]] = []
    for paper in papers:
        pdf_path = downloaded.get(paper.key)
        if pdf_path is None:
            continue
        try:
            records.append(inspect_paper(paper, pdf_path, page_dir))
            print(f"inspected {paper.key}", flush=True)
        except Exception as exc:  # Keep a visible failure record.
            failures.append({"key": paper.key, "error": f"inspection: {exc}"})
            print(f"FAILED inspection {paper.key}: {exc}", flush=True)

    sheets = make_contact_sheets(records, output_dir)
    payload = {
        "requested": len(papers),
        "inspected": len(records),
        "minimum_required": 30,
        "minimum_met": len(records) >= 30,
        "failures": failures,
        "papers": records,
        "contact_sheets": sheets,
    }
    (output_dir / "corpus_index.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(records, sheets, output_dir / "corpus_index.md")
    print(json.dumps({
        "status": "PASS" if payload["minimum_met"] else "FAIL",
        "requested": len(papers),
        "inspected": len(records),
        "failures": len(failures),
        "contact_sheets": sheets,
        "output_dir": str(output_dir),
    }, indent=2))
    return 0 if payload["minimum_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
