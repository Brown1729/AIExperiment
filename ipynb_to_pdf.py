#!/usr/bin/env python3
"""Convert a local Jupyter notebook to PDF with pandoc (TOC on, headings preserved)."""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


class ConversionError(RuntimeError):
    """Raised when notebook to PDF conversion fails."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a local Jupyter notebook to PDF with pandoc.",
        epilog=(
            "Examples:\n"
            "  python ipynb_to_pdf.py exp10/rnn.ipynb\n"
            "  python ipynb_to_pdf.py exp10/rnn.ipynb -o exp10/rnn_report.pdf\n"
            "  python ipynb_to_pdf.py exp10/rnn.ipynb --no-toc --number-sections\n"
            "  python ipynb_to_pdf.py exp10/rnn.ipynb --dry-run"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_notebook", help="Path to the source .ipynb file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output PDF file. Defaults to the same name as the input.",
    )
    parser.add_argument(
        "--pdf-engine",
        default="xelatex",
        help="Pandoc PDF engine. Defaults to xelatex.",
    )
    parser.add_argument(
        "--font-size",
        default="10pt",
        help="Base document font size. Defaults to 10pt.",
    )
    parser.add_argument(
        "--margin",
        default="0.72in",
        help="Page margin passed to LaTeX geometry. Defaults to 0.72in.",
    )
    parser.add_argument(
        "--documentclass",
        default="ctexart",
        help="LaTeX document class passed to pandoc. Defaults to ctexart.",
    )
    parser.add_argument(
        "--lua-filter",
        action="append",
        default=[],
        help="Extra pandoc Lua filter. Can be used multiple times.",
    )
    parser.add_argument(
        "--include-in-header",
        help=(
            "Optional LaTeX header include file. Defaults to pandoc_notebook_header.tex "
            "next to this script if that file exists."
        ),
    )
    parser.add_argument(
        "--no-header-style",
        action="store_true",
        help="Disable the default LaTeX header style include.",
    )
    parser.add_argument(
        "--resource-path",
        action="append",
        default=[],
        help="Extra resource directories for images and attachments. Can be used multiple times.",
    )
    parser.add_argument(
        "--mainfont",
        default="Times New Roman" if os.name == "nt" else None,
        help="Main text font. Defaults to Times New Roman on Windows.",
    )
    parser.add_argument(
        "--cjkmainfont",
        default="SimSun" if os.name == "nt" else None,
        help="CJK main font. Defaults to SimSun on Windows.",
    )
    parser.add_argument(
        "--monofont",
        default="Consolas" if os.name == "nt" else None,
        help="Monospace font. Defaults to Consolas on Windows.",
    )
    parser.add_argument(
        "-V",
        "--variable",
        action="append",
        default=[],
        help="Extra pandoc template variable in KEY=VALUE form. Can be used multiple times.",
    )
    parser.add_argument(
        "--toc",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Generate a table of contents. On by default; use --no-toc to disable.",
    )
    parser.add_argument(
        "--toc-depth",
        type=int,
        default=1,
        help="Number of heading levels to include in the TOC. Defaults to 1 (only top-level).",
    )
    parser.add_argument(
        "--number-sections",
        action="store_true",
        help="Add section numbers to headings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pandoc command without running it.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress pandoc/xelatex live output and only report the final result.",
    )
    parser.add_argument(
        "--cover-pdf",
        help="Path to a PDF to use as the cover page (prepended before the TOC).",
    )
    return parser


def resolve_input(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise ConversionError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ConversionError(f"Input path is not a file: {path}")
    if path.suffix.lower() != ".ipynb":
        raise ConversionError(f"Input file is not a .ipynb notebook: {path}")
    return path


def resolve_output(input_notebook: Path, output_text: str | None) -> Path:
    if output_text:
        output_pdf = Path(output_text).expanduser().resolve()
    else:
        output_pdf = input_notebook.with_suffix(".pdf")
    if output_pdf == input_notebook:
        raise ConversionError("Output path must be different from the input notebook file.")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    return output_pdf


def resolve_filter_paths(
    filter_texts: list[str],
) -> list[Path]:
    """Resolve user-supplied Lua filter paths."""
    paths: list[Path] = []
    for text in filter_texts:
        path = Path(text).expanduser().resolve()
        if not path.exists():
            raise ConversionError(f"Lua filter does not exist: {path}")
        paths.append(path)
    return paths


def resolve_header_include_path(
    script_path: Path, include_text: str | None, no_header_style: bool
) -> Path | None:
    if no_header_style:
        return None

    if include_text:
        include_path = Path(include_text).expanduser().resolve()
        if not include_path.exists():
            raise ConversionError(f"Header include file does not exist: {include_path}")
        return include_path

    default_include = script_path.with_name("pandoc_notebook_header.tex")
    if default_include.exists():
        return default_include
    return None


def build_resource_path(input_notebook: Path, extra_paths: list[str]) -> str:
    paths: list[Path] = [input_notebook.parent]
    seen: set[Path] = {input_notebook.parent}

    for path_text in extra_paths:
        path = Path(path_text).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)

    return os.pathsep.join(str(path) for path in paths)


def temp_output_path(output_pdf: Path) -> Path:
    return output_pdf.with_name(
        f"{output_pdf.stem}.tmp.{os.getpid()}{output_pdf.suffix}"
    )


def creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def write_console(text: str) -> None:
    """Write subprocess logs without crashing on terminal encoding issues."""
    if not text:
        return

    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(
            encoding,
            errors="replace",
        )
        sys.stdout.write(safe_text)

    sys.stdout.flush()


def resolve_builtin_filters(script_path: Path) -> list[Path]:
    """Return auto-loaded Lua filters that live next to this script."""
    builtins: list[Path] = []
    for name in ("newpage_before_h1.lua", "math_to_rawtex.lua"):
        candidate = script_path.with_name(name)
        if candidate.exists():
            builtins.append(candidate)
    return builtins


def resolve_cover_path(cover_text: str | None) -> Path | None:
    """Validate and resolve the cover PDF path."""
    if not cover_text:
        return None
    path = Path(cover_text).expanduser().resolve()
    if not path.exists():
        raise ConversionError(f"Cover PDF does not exist: {path}")
    if not path.is_file():
        raise ConversionError(f"Cover PDF path is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ConversionError(f"Cover file is not a PDF: {path}")
    return path


def prepare_cover_tempfiles(
    cover_pdf: Path,
) -> tuple[Path, Path]:
    """Create temporary .tex files for injecting the cover into the LaTeX build.

    Returns ``(header_tex, before_body_tex)``:
    * *header_tex* — contains ``\\usepackage{pdfpages}`` for the preamble.
    * *before_body_tex* — contains ``\\includepdf{...}``, placed right after
      ``\\begin{document}`` (before the TOC so page numbers stay correct).
    """
    # --include-in-header  ------------------------------------------------
    header_fd, header_path = tempfile.mkstemp(suffix=".tex", prefix="cover_hdr_")
    os.write(header_fd, b"\\usepackage{pdfpages}\n")
    os.close(header_fd)

    # --include-before-body -----------------------------------------------
    # XeLaTeX tolerates forward slashes on Windows.
    cover_abs = str(cover_pdf).replace("\\", "/")
    before_fd, before_path = tempfile.mkstemp(suffix=".tex", prefix="cover_body_")
    os.write(before_fd, f"\\includepdf[pages=-]{{{cover_abs}}}\n".encode())
    os.close(before_fd)

    return Path(header_path), Path(before_path)


def build_command(
    args: argparse.Namespace,
    input_notebook: Path,
    output_pdf: Path,
    filter_paths: list[Path],
    header_include_path: Path | None,
    cover_header: Path | None,
    cover_before_body: Path | None,
) -> list[str]:
    pandoc_path = shutil.which("pandoc")
    if not pandoc_path:
        raise ConversionError("pandoc was not found in PATH.")

    if args.pdf_engine and not shutil.which(args.pdf_engine):
        raise ConversionError(f"PDF engine was not found in PATH: {args.pdf_engine}")

    temp_output = temp_output_path(output_pdf)
    command = [
        pandoc_path,
        "--from",
        "ipynb",
        str(input_notebook),
        "-o",
        str(temp_output),
        "--resource-path",
        build_resource_path(input_notebook, args.resource_path),
    ]

    for fp in filter_paths:
        command.extend(["--lua-filter", str(fp)])

    if header_include_path:
        command.extend(["--include-in-header", str(header_include_path)])

    if cover_header:
        command.extend(["--include-in-header", str(cover_header)])

    if cover_before_body:
        command.extend(["--include-before-body", str(cover_before_body)])

    if args.pdf_engine:
        command.extend(["--pdf-engine", args.pdf_engine])

    if args.documentclass:
        command.extend(["-V", f"documentclass={args.documentclass}"])

    # ctex's CJKmath option patches \mathrm/\mathbf/\mathsf/\mathtt to use
    # the CJK main font, so Chinese inside $$...$$ doesn't disappear.
    command.extend(["-V", "classoption=CJKmath=true"])

    if args.font_size:
        command.extend(["-V", f"fontsize={args.font_size}"])

    if args.margin:
        command.extend(["-V", f"geometry:margin={args.margin}"])

    if args.mainfont:
        command.extend(["-V", f"mainfont={args.mainfont}"])

    if args.cjkmainfont:
        command.extend(["-V", f"CJKmainfont={args.cjkmainfont}"])

    if args.monofont:
        command.extend(["-V", f"monofont={args.monofont}"])

    for variable in args.variable:
        command.extend(["-V", variable])

    if args.toc:
        command.append("--toc")
        command.extend(["--toc-depth", str(args.toc_depth)])

    if args.number_sections:
        command.append("--number-sections")

    return command


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def next_available_output_path(output_pdf: Path) -> Path:
    stem = output_pdf.stem
    suffix = output_pdf.suffix
    parent = output_pdf.parent

    for candidate_name in (
        f"{stem}.generated{suffix}",
        f"{stem}.new{suffix}",
    ):
        candidate = parent / candidate_name
        if not candidate.exists():
            return candidate

    counter = 2
    while True:
        candidate = parent / f"{stem}.generated_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def run_command(command: list[str], output_pdf: Path, quiet: bool) -> Path:
    temp_output = temp_output_path(output_pdf)
    if temp_output.exists():
        temp_output.unlink()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creation_flags(),
    )

    stdout_text = ""

    try:
        if quiet:
            stdout_text, _ = process.communicate()
        else:
            assert process.stdout is not None
            output_chunks: list[str] = []

            for line in process.stdout:
                output_chunks.append(line)
                write_console(line)

            process.wait()
            stdout_text = "".join(output_chunks)
    except KeyboardInterrupt as exc:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise ConversionError("Conversion cancelled by user.") from exc

    if process.returncode != 0:
        details = stdout_text.strip() if quiet else "See pandoc/xelatex output above."
        raise ConversionError(
            f"pandoc failed with exit code {process.returncode}.\n{details}"
        )

    if not temp_output.exists() or temp_output.stat().st_size == 0:
        details = stdout_text.strip() if quiet else "See pandoc/xelatex output above."
        raise ConversionError(
            "pandoc did not create a PDF file.\n"
            f"{details or 'No logs were produced.'}"
        )

    try:
        temp_output.replace(output_pdf)
        return output_pdf
    except PermissionError:
        destinations = [output_pdf, next_available_output_path(output_pdf)]

        for destination in destinations:
            try:
                shutil.copyfile(temp_output, destination)
                temp_output.unlink(missing_ok=True)
                return destination
            except PermissionError:
                continue
            except OSError:
                continue

        if output_pdf.exists() and output_pdf.stat().st_size > 0:
            return output_pdf

        if temp_output.exists() and temp_output.stat().st_size > 0:
            return temp_output

        raise ConversionError(
            "The PDF was generated, but the output file could not be written. "
            "Close the target PDF if it is open, then try again."
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Track temp files so we can clean them up even on error.
    _tempfiles: list[Path] = []

    def _cleanup() -> None:
        for p in _tempfiles:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    atexit.register(_cleanup)

    try:
        script_path = Path(__file__).resolve()
        input_notebook = resolve_input(args.input_notebook)
        output_pdf = resolve_output(input_notebook, args.output)
        filter_paths = resolve_builtin_filters(script_path) + resolve_filter_paths(args.lua_filter)
        header_include_path = resolve_header_include_path(
            script_path,
            args.include_in_header,
            args.no_header_style,
        )

        # Cover PDF — generates two temp .tex files for pandoc.
        cover_pdf_path = resolve_cover_path(args.cover_pdf)
        cover_header: Path | None = None
        cover_before_body: Path | None = None
        if cover_pdf_path:
            cover_header, cover_before_body = prepare_cover_tempfiles(cover_pdf_path)
            _tempfiles.extend([cover_header, cover_before_body])

        command = build_command(
            args,
            input_notebook,
            output_pdf,
            filter_paths,
            header_include_path,
            cover_header,
            cover_before_body,
        )

        if args.dry_run:
            print(format_command(command))
            return 0

        print(f"Running pandoc for {input_notebook.name} ...")
        actual_output = run_command(command, output_pdf, quiet=args.quiet)
    except ConversionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if actual_output != output_pdf:
        print(
            f"Created {actual_output} using pandoc "
            f"(the requested file was likely in use: {output_pdf.name})."
        )
    else:
        print(f"Created {output_pdf} using pandoc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
