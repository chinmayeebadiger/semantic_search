"""Data loading and text preprocessing for the semantic search pipeline."""

from __future__ import annotations

import re
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """A source document with normalized text for embedding."""

    id: int
    original_text: str
    text: str       #normalized  
    category: str

#regex. 
_WHITESPACE_RE = re.compile(r"\s+")     #whitespaces
_QUOTE_RE = re.compile(r"^\s*(>|->|\||[a-zA-Z]{1,5}>)")     #email and quoted


DEFAULT_LOCAL_ARCHIVE = Path(
    "/Users/mallappabadiger/Downloads/Twenty Newsgroups/20_newsgroups.tar.gz"
)


def normalize_text(text: str) -> str:
    """Lowercase text and collapse repeated whitespace."""

    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def remove_headers_footers_quotes(text: str) -> str:
    """Approximate sklearn's header, footer, and quote removal for local files."""

    lines = text.splitlines()

    # Headers end at the first blank line in these raw newsgroup files.
    for index, line in enumerate(lines):
        if not line.strip():
            lines = lines[index + 1 :]
            break

    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if _QUOTE_RE.match(stripped):
            continue
        if " writes:" in stripped.lower() or stripped.lower().startswith("in article "):
            continue
        if stripped.startswith(("---", "--", "___")): #end at signature  
            break
        if stripped.lower().startswith(("* origin:", "begin forwarded message")):
            break
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _load_documents_from_local_archive(
    archive_path: str | Path,
    max_documents: int | None = None,
) -> list[Document]:
    """Load documents from a local 20_newsgroups tar.gz archive."""

    path = Path(archive_path).expanduser()
    documents: list[Document] = []
    members_by_category: dict[str, list[tarfile.TarInfo]] = defaultdict(list)
    next_id = 0

    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue

            parts = Path(member.name).parts
            if len(parts) != 3:
                continue

            _, category, _ = parts
            members_by_category[category].append(member) #group files by category

        categories = sorted(members_by_category)
        category_index = 0
        while categories and (max_documents is None or len(documents) < max_documents):
            category = categories[category_index % len(categories)] #round robin - distributed dataset. 
            members = members_by_category[category]
            member = members.pop(0)

            extracted = archive.extractfile(member)
            if extracted is None:
                if not members:
                    categories.remove(category)
                    category_index = 0
                else:
                    category_index += 1
                continue

            raw_text = extracted.read().decode("latin-1", errors="replace")
            cleaned_text = remove_headers_footers_quotes(raw_text)
            normalized_text = normalize_text(cleaned_text)
            if not normalized_text:
                if not members:
                    categories.remove(category)
                    category_index = 0
                else:
                    category_index += 1
                continue

            documents.append( #final clean doc object form- used
                Document(
                    id=next_id,
                    original_text=cleaned_text.strip(),
                    text=normalized_text,
                    category=category,
                )
            )
            next_id += 1

            if not members:
                categories.remove(category)
                category_index = 0
            else:
                category_index += 1

    return documents


def _load_documents_from_sklearn(               #alternative
    subset: str,
    max_documents: int | None,
) -> list[Document]:
    """Load documents through sklearn, which downloads the dataset if needed."""

    from sklearn.datasets import fetch_20newsgroups

    dataset = fetch_20newsgroups(
        subset=subset,
        remove=("headers", "footers", "quotes"),
    )

    documents: list[Document] = []
    for doc_id, raw_text in enumerate(dataset.data):
        normalized_text = normalize_text(raw_text)
        if not normalized_text:
            continue

        target = dataset.target[doc_id]
        documents.append(
            Document(
                id=doc_id,
                original_text=raw_text.strip(),
                text=normalized_text,
                category=dataset.target_names[target],
            )
        )

        if max_documents is not None and len(documents) >= max_documents:
            break

    return documents


def load_20newsgroups_documents(
    subset: str = "train",
    max_documents: int | None = None,
    archive_path: str | Path | None = None,
) -> list[Document]:
    """Load 20 Newsgroups with headers, footers, and quotes removed."""

    if archive_path is not None:
        return _load_documents_from_local_archive(
            archive_path=archive_path,
            max_documents=max_documents,
        )

    return _load_documents_from_sklearn(
        subset=subset,
        max_documents=max_documents,
    )
