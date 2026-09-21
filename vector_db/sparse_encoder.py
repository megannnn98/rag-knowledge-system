"""
Sparse (BM25-style) vector encoding for Qdrant hybrid search.

Term frequency (BM25-saturated) only — Qdrant applies IDF weighting
server-side at query time via the collection's sparse vector Modifier.IDF,
using corpus-wide stats it maintains incrementally. No client-side global
index or rebuild needed.
"""
import hashlib
import re
from collections import Counter

from qdrant_client.models import SparseVector

# Words that carry no search signal
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "as", "if", "not", "no", "nor",
    "so", "yet", "both", "either", "each", "any", "all", "some",
}

# Feature-hashing bucket count for sparse vector indices. Large enough that
# collisions are rare for a corpus vocabulary of a few hundred thousand terms.
_HASH_BUCKETS = 2 ** 31 - 1

# Standard BM25 term-frequency saturation constant. Without this, raw term
# counts are unbounded — a query that accidentally (or deliberately) repeats
# common words many times inflates their weight linearly (e.g. 20 repeats of
# "discount" -> value 20), which can outweigh a genuinely relevant document's
# rarer, more specific terms in RRF fusion and pull in unrelated documents
# that merely share the same common words. Saturation caps each term's
# contribution at (k1+1) regardless of how many times it repeats.
_BM25_K1 = 1.2


# Unicode-aware token pattern: runs of letters/digits in ANY script (Cyrillic
# included), keeping internal hyphens so a technical identifier like
# "A016ISMT-901" survives as one token instead of being cut apart.
# The previous `re.sub(r"[^a-z0-9\s]", " ", lowered)` deleted every non-ASCII
# character BEFORE splitting, so a Russian query tokenized to nothing at all
# ("пороговое значение уровня жидкости" -> []). build_sparse_vector() then
# returned an empty vector, and hybrid_search() only appends the sparse
# prefetch `if sparse_vector.indices` — so every Russian query silently
# degraded to dense-only search with no error anywhere.
# No Russian stemming here deliberately: Qdrant applies IDF server-side
# (Modifier.IDF), which already down-weights the ubiquitous forms a stop-word
# list would remove, and real morphology is a separate, heavier decision.
_TOKEN_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*")


def _stem(token: str) -> str:
    for suffix in ("ment", "tion", "ing", "ness", "ies", "ied", "ed", "er", "ly", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics (any script), remove stop words, stem."""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group()
        tokens.append(token)
        if "-" in token:
            # Index the parts alongside the whole, so "A016ISMT-901" is
            # findable by a query writing only "A016ISMT" (and the reverse) —
            # a hyphenated identifier is routinely cited both ways.
            tokens.extend(part for part in token.split("-") if part)
    return [_stem(t) for t in tokens if len(t) > 1 and t not in STOP_WORDS]


def _stable_token_index(token: str) -> int:
    # hashlib, not Python's builtin hash() — builtin hash() is randomized per
    # process via PYTHONHASHSEED and would silently desync the sparse index
    # across restarts/replicas.
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % _HASH_BUCKETS


def build_sparse_vector(text: str) -> SparseVector:
    """BM25-saturated term-frequency sparse vector. IDF weighting applied by Qdrant at query time."""
    tokens = tokenize(text)
    if not tokens:
        return SparseVector(indices=[], values=[])
    counts = Counter(_stable_token_index(t) for t in tokens)
    values = [(_BM25_K1 + 1) * tf / (_BM25_K1 + tf) for tf in counts.values()]
    return SparseVector(indices=list(counts.keys()), values=values)
