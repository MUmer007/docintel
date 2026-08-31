"""
Download a handful of real SEC 10-K filings from EDGAR for the DocIntel corpus.

SEC EDGAR requires a descriptive User-Agent identifying the requester (name +
email) -- anonymous/generic User-Agents get blocked. We also rate-limit
ourselves well under SEC's 10 req/s cap to be a good citizen.

Usage:
    uv run python scripts/download_filings.py
"""

import time
from pathlib import Path

import httpx

# --- CONFIG: edit this before running ---
USER_AGENT = "Umer umershaikh19870@gmail.com"  # SEC REQUIRES a real identifying string

# A small, well-known set of companies (CIK = SEC's company identifier).
# Mix of sectors so retrieval/eval later isn't trivially single-domain.
COMPANIES = {
    "AAPL": "0000320193",   # Apple
    "MSFT": "0000789019",   # Microsoft
    "JPM": "0000019617",    # JPMorgan Chase
}

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
HEADERS = {"User-Agent": USER_AGENT}


def get_latest_10k_url(client: httpx.Client, cik: str) -> tuple[str, str] | None:
    """Look up a company's most recent 10-K filing document URL."""
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = client.get(submissions_url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    for form, accession, doc, date in zip(
        recent["form"],
        recent["accessionNumber"],
        recent["primaryDocument"],
        recent["filingDate"],
        strict=True,
    ):
        if form == "10-K":
            accession_nodash = accession.replace("-", "")
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{accession_nodash}/{doc}"
            )
            return doc_url, date
    return None


def download_filing(client: httpx.Client, ticker: str, cik: str) -> None:
    result = get_latest_10k_url(client, cik)
    if result is None:
        print(f"[skip] No 10-K found for {ticker}")
        return

    doc_url, filing_date = result
    print(f"[fetch] {ticker} 10-K ({filing_date}) -> {doc_url}")

    resp = client.get(doc_url, headers=HEADERS)
    resp.raise_for_status()

    ext = ".htm" if doc_url.endswith((".htm", ".html")) else ".pdf"
    out_path = OUTPUT_DIR / f"{ticker}_10K_{filing_date}{ext}"
    out_path.write_bytes(resp.content)
    print(f"[saved] {out_path} ({len(resp.content) / 1024:.0f} KB)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30.0) as client:
        for ticker, cik in COMPANIES.items():
            download_filing(client, ticker, cik)
            time.sleep(0.5)  # stay well under SEC's rate limit


if __name__ == "__main__":
    main()