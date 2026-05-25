"""
Pull historical FX/metals data from Dukascopy via dukascopy-node.

Requires Node.js (already installed). Uses npx so no global install needed.
Output: one CSV per instrument, GMT timestamps,
schema compatible with engine.data_loader.

Default output: C:\\TradingData\\dukascopy\\ (outside OneDrive to avoid sync).
Override via DUKA_DATA_DIR env var or --output-dir flag.

CHUNKED PULLS: full date range is split into N-year chunks (default 2y).
Each chunk is downloaded to a temp file. If a chunk fails, only that chunk
must be retried. Already-completed chunks are skipped on resume. Final
step concatenates all chunks into a single CSV per instrument.

Usage:
    python scripts/pull_dukascopy.py                       # all defaults
    python scripts/pull_dukascopy.py --instruments xauusd  # subset
    python scripts/pull_dukascopy.py --chunk-years 1       # smaller chunks
    python scripts/pull_dukascopy.py --no-concat           # leave as chunks
"""

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(os.environ.get("DUKA_DATA_DIR", r"C:\TradingData\dukascopy"))
DEFAULT_INSTRUMENTS = ["xauusd", "xagusd", "audusd", "gbpusd", "usdjpy"]
DEFAULT_TIMEFRAME = "m1"
DEFAULT_FROM = "2007-01-01"
DEFAULT_TO = date.today().isoformat()
DEFAULT_CHUNK_YEARS = 2


def chunk_ranges(date_from: str, date_to: str, chunk_years: int):
    """Yield (start, end) date strings covering [date_from, date_to] in chunk_years slices."""
    start_year = int(date_from[:4])
    end_year = int(date_to[:4])
    end_month_day = date_to[4:]
    cur = start_year
    while cur <= end_year:
        chunk_end_year = min(cur + chunk_years - 1, end_year)
        chunk_start = f"{cur}-01-01" if cur != start_year else date_from
        if chunk_end_year == end_year:
            chunk_end = date_to
        else:
            chunk_end = f"{chunk_end_year + 1}-01-01"
        yield chunk_start, chunk_end
        cur = chunk_end_year + 1


def pull_chunk(instrument: str, timeframe: str, date_from: str, date_to: str,
               output_dir: Path, chunks_dir: Path) -> tuple[int, Path]:
    """Pull one chunk. Returns (exit code, output file path)."""
    out_name = f"{instrument}_{timeframe}_{date_from}_{date_to}"
    out_path = chunks_dir / f"{out_name}.csv"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [skip] chunk already exists: {out_path.name} ({out_path.stat().st_size // 1024} KB)")
        return 0, out_path
    cmd = [
        "npx", "--yes", "dukascopy-node",
        "-i", instrument,
        "-from", date_from,
        "-to", date_to,
        "-t", timeframe,
        "-f", "csv",
        "-tz", "0",
        "-fl", "true",
        "-dir", str(chunks_dir),
        "-fn", out_name,
        "-v", "true",
        "--retries", "5",
        "--batch-size", "10",
        "--batch-pause", "1000",
    ]
    result = subprocess.run(cmd, shell=True)
    return result.returncode, out_path


def concat_chunks(instrument: str, timeframe: str, date_from: str, date_to: str,
                  chunk_paths: list[Path], output_dir: Path) -> Path:
    """Concatenate chunk CSVs into one final file. Strips repeated headers."""
    final_name = f"{instrument}_{timeframe}_{date_from}_{date_to}.csv"
    final_path = output_dir / final_name
    with final_path.open("w", encoding="utf-8") as out:
        header_written = False
        for chunk_path in chunk_paths:
            with chunk_path.open("r", encoding="utf-8") as f:
                header = f.readline()
                if not header_written:
                    out.write(header)
                    header_written = True
                for line in f:
                    out.write(line)
    return final_path


def pull_instrument(instrument: str, timeframe: str, date_from: str, date_to: str,
                    output_dir: Path, chunks_dir: Path, chunk_years: int,
                    do_concat: bool) -> bool:
    """Pull all chunks for one instrument, then concatenate. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"{instrument.upper()} ({timeframe}) {date_from} -> {date_to} in {chunk_years}y chunks")
    print(f"{'='*60}")
    chunk_paths = []
    chunks = list(chunk_ranges(date_from, date_to, chunk_years))
    for i, (cs, ce) in enumerate(chunks, 1):
        print(f"\n  [{i}/{len(chunks)}] {cs} -> {ce}")
        code, path = pull_chunk(instrument, timeframe, cs, ce, output_dir, chunks_dir)
        if code != 0:
            print(f"  ! chunk {cs}->{ce} FAILED (exit {code})")
            print(f"  ! skipping rest of {instrument}; re-run script to resume")
            return False
        chunk_paths.append(path)
    if do_concat:
        print(f"\n  Concatenating {len(chunk_paths)} chunks...")
        final = concat_chunks(instrument, timeframe, date_from, date_to,
                              chunk_paths, output_dir)
        size_mb = final.stat().st_size / (1024 * 1024)
        print(f"  Wrote: {final} ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruments", nargs="+", default=DEFAULT_INSTRUMENTS)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME,
                        choices=["tick", "m1", "m5", "m15", "m30", "h1", "h4", "d1"])
    parser.add_argument("--from", dest="date_from", default=DEFAULT_FROM)
    parser.add_argument("--to", dest="date_to", default=DEFAULT_TO)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--chunk-years", type=int, default=DEFAULT_CHUNK_YEARS,
                        help=f"Years per chunk. Default {DEFAULT_CHUNK_YEARS}.")
    parser.add_argument("--no-concat", action="store_true",
                        help="Skip final concatenation; leave chunks as separate files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    chunks_dir = output_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    print(f"Chunks: {chunks_dir}")

    failures = []
    for inst in args.instruments:
        ok = pull_instrument(inst, args.timeframe, args.date_from, args.date_to,
                             output_dir, chunks_dir, args.chunk_years,
                             do_concat=not args.no_concat)
        if not ok:
            failures.append(inst)

    print(f"\n{'='*60}")
    print(f"Summary: requested {len(args.instruments)}, failed {len(failures)}")
    if failures:
        print(f"Failed: {failures}", file=sys.stderr)
        print(f"Re-run the script to resume — completed chunks will be skipped.",
              file=sys.stderr)
        sys.exit(1)
    print(f"All done. CSVs in: {output_dir}")


if __name__ == "__main__":
    main()
