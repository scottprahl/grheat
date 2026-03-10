"""Update CITATION.cff and README.rst citation data from the latest GitHub release."""

from __future__ import annotations

import re
from pathlib import Path

import requests
import yaml

USERNAME = "scottprahl"
REPO = "grheat"
GITHUB_API_URL = f"https://api.github.com/repos/{USERNAME}/{REPO}/releases/latest"

HEADERS = {
    "User-Agent": f"{REPO}-citation-updater",
    "Accept": "application/vnd.github+json",
}


def get_release_date() -> tuple[str, str]:
    """Return the latest GitHub release date and year.

    Returns:
        tuple[str, str]: The ISO release date and four-digit release year.
    """
    response = requests.get(GITHUB_API_URL, timeout=10, headers=HEADERS)
    response.raise_for_status()
    release_info = response.json()

    release_date = release_info["published_at"].split("T")[0]
    release_year = release_date.split("-")[0]
    tag_version = release_info.get("tag_name", "").lstrip("v")
    print(
        f"GitHub latest release -> tag: {release_info.get('tag_name')}, "
        f"version (from tag): {tag_version}, date: {release_date}"
    )
    return release_date, release_year


def get_code_version() -> str:
    """Extract the package version from ``__init__.py``.

    Returns:
        str: The package version string.
    """
    init_path = Path(REPO) / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"{init_path} not found; cannot read __version__")

    content = init_path.read_text(encoding="utf-8")
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)
    if not match:
        raise RuntimeError(f"Could not find __version__ in {init_path}")

    version = match.group(1).strip()
    print(f"Version from {init_path} -> {version}")
    return version


def update_citation(version: str, release_date: str, year: str) -> None:
    """Update citation metadata in ``CITATION.cff``.

    Args:
        version (str): Package version to write into the citation file.
        release_date (str): Release date in ISO ``YYYY-MM-DD`` format.
        year (str): Four-digit year associated with the release.

    Returns:
        None: This function updates the citation file in place when needed.
    """
    citation_path = Path("CITATION.cff")
    if not citation_path.exists():
        print("CITATION.cff not found; skipping citation update.")
        return

    with citation_path.open("r", encoding="utf-8") as file:
        cff_data = yaml.safe_load(file)

    if not isinstance(cff_data, dict):
        raise RuntimeError("CITATION.cff does not contain a top-level mapping.")

    changed = False

    if cff_data.get("date-released") != release_date:
        cff_data["date-released"] = release_date
        changed = True

    if cff_data.get("version") != version:
        cff_data["version"] = version
        changed = True

    preferred = cff_data.get("preferred-citation")
    if isinstance(preferred, dict):
        if str(preferred.get("year")) != year:
            preferred["year"] = int(year) if year.isdigit() else year
            changed = True

        if preferred.get("version") != version:
            preferred["version"] = version
            changed = True

        cff_data["preferred-citation"] = preferred

    if changed:
        with citation_path.open("w", encoding="utf-8") as file:
            yaml.dump(cff_data, file, sort_keys=False)
        print(f"CITATION.cff updated -> version: {version}, date: {release_date}")
    else:
        print("CITATION.cff already up to date.")


def update_readme(version: str, year: str) -> None:
    """Update citation references in ``README.rst`` when expected patterns exist.

    Args:
        version (str): Package version to inject into citation text.
        year (str): Four-digit release year to inject into citation text.

    Returns:
        None: This function updates the README in place when needed.
    """
    readme_path = Path("README.rst")
    if not readme_path.exists():
        print("README.rst not found; skipping README update.")
        return

    text = readme_path.read_text(encoding="utf-8")
    original_text = text

    text = re.sub(
        r"(Prahl,\s*S\.\s*\()(\d{4})(\)\.)",
        lambda match: f"{match.group(1)}{year}{match.group(3)}",
        text,
    )
    text = re.sub(r"\(Version [^)]+\)", f"(Version {version})", text)
    text = re.sub(
        r"(@software\{[A-Za-z0-9_]+_prahl_)(\d{4})(\s*,)",
        lambda match: f"{match.group(1)}{year}{match.group(3)}",
        text,
    )
    text = re.sub(
        r"(year\s*=\s*\{)(\d{4})(\s*\},)",
        lambda match: f"{match.group(1)}{year}{match.group(3)}",
        text,
    )
    text = re.sub(
        r"(version\s*=\s*\{)([^}]+)(\s*\},)",
        lambda match: f"{match.group(1)}{version}{match.group(3)}",
        text,
    )

    if text != original_text:
        readme_path.write_text(text, encoding="utf-8")
        print(f"README.rst updated -> version: {version}, year: {year}")
    else:
        print("README.rst citation block already up to date.")


def main() -> None:
    """Update citation metadata files from the latest GitHub release.

    Returns:
        None: This function orchestrates the citation metadata update workflow.
    """
    release_date, year = get_release_date()
    version = get_code_version()
    update_citation(version, release_date, year)
    update_readme(version, year)


if __name__ == "__main__":
    main()
