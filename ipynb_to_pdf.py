#!/usr/bin/env python3
"""Convert a local Jupyter notebook to PDF with pandoc."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
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
            "  python ipynb_to_pdf.py exp10/rnn.ipynb --toc --number-sections\n"
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
        help=(
            "Optional pandoc Lua filter. Defaults to first_h1_as_title.lua next to this script "
            "if that file exists."
        ),
    )
    parser.add_argument(
        "--include-in-header",
        help=(
            "Optional LaTeX header include file. Defaults to pandoc_notebook_header.tex "
            "next to this script if that file exists."
        ),
    )
    parser.add_argument(
        "--no-lua-filter",
        action="store_true",
        help="Disable the default Lua filter.",
    )
    parser.add_argument(
        "--no-header-style",
        action="store_true",
        help="Disable the default LaTeX header style include.",
    )
    parser.add_argument(
        "--keep-notebook-title",
        action="store_true",
        help="Keep notebook metadata title instead of promoting the first H1 to the document title.",
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
        action="store_true",
        help="Generate a table of contents.",
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


def resolve_filter_path(
    script_path: Path, filter_text: str | None, no_lua_filter: bool
) -> Path | None:
    if no_lua_filter:
        return None

    if filter_text:
        filter_path = Path(filter_text).expanduser().resolve()
        if not filter_path.exists():
            raise ConversionError(f"Lua filter does not exist: {filter_path}")
        return filter_path

    default_filter = script_path.with_name("first_h1_as_title.lua")
    if default_filter.exists():
        return default_filter
    return None


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
    return output_pdf.with_name(f"{output_pdf.stem}.tmp{output_pdf.suffix}")


def creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def build_command(
    args: argparse.Namespace,
    input_notebook: Path,
    output_pdf: Path,
    filter_path: Path | None,
    header_include_path: Path | None,
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

    if filter_path:
        command.extend(["--lua-filter", str(filter_path)])

    if header_include_path:
        command.extend(["--include-in-header", str(header_include_path)])

    if not args.keep_notebook_title:
        command.extend(["-M", "title-from-first-h1=true"])

    if args.pdf_engine:
        command.extend(["--pdf-engine", args.pdf_engine])

    if args.documentclass:
        command.extend(["-V", f"documentclass={args.documentclass}"])

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

    if args.number_sections:
        command.append("--number-sections")

    return command


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run_command(command: list[str], output_pdf: Path, quiet: bool) -> None:
    temp_output = temp_output_path(output_pdf)
    if temp_output.exists():
        temp_output.unlink()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
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
            process.wait()
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
    except PermissionError:
        try:
            shutil.copyfile(temp_output, output_pdf)
            temp_output.unlink(missing_ok=True)
        except Exception as copy_exc:
            raise ConversionError(
                "The PDF was generated, but the output file could not be replaced. "
                "Close the target PDF if it is open, then try again."
            ) from copy_exc


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        script_path = Path(__file__).resolve()
        input_notebook = resolve_input(args.input_notebook)
        output_pdf = resolve_output(input_notebook, args.output)
        filter_path = resolve_filter_path(script_path, args.lua_filter, args.no_lua_filter)
        header_include_path = resolve_header_include_path(
            script_path,
            args.include_in_header,
            args.no_header_style,
        )
        command = build_command(
            args,
            input_notebook,
            output_pdf,
            filter_path,
            header_include_path,
        )

        if args.dry_run:
            print(format_command(command))
            return 0

        print(f"Running pandoc for {input_notebook.name} ...")
        run_command(command, output_pdf, quiet=args.quiet)
    except ConversionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Created {output_pdf} using pandoc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
