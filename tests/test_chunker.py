from ragmcp.ingest.chunker import approx_tokens, chunk_text

DOC = """Intro text before any heading.

# Bab I

Umum.

## Pasal 1

Isi pasal satu.

## Pasal 2

Isi pasal dua.

# Bab II

```
# not a heading
```

Penutup.
"""


def test_heading_paths():
    got = [(c.heading_path, c.text) for c in chunk_text(DOC)]
    paths = [p for p, _ in got]
    assert paths == ["", "Bab I", "Bab I > Pasal 1", "Bab I > Pasal 2", "Bab II"]
    assert "Isi pasal satu." in got[2][1]
    assert "Isi pasal dua." not in got[2][1]  # boundary at the next heading


def test_fenced_code_hash_is_not_a_heading():
    last = chunk_text(DOC)[-1]
    assert "# not a heading" in last.text and last.heading_path == "Bab II"


def test_size_and_overlap():
    words = [f"w{i}" for i in range(1200)]
    text = "# T\n\n" + "\n\n".join(" ".join(words[i:i + 20]) for i in range(0, 1200, 20))
    chunks = chunk_text(text, max_tokens=100, overlap=15)
    assert len(chunks) > 5
    assert all(c.heading_path == "T" for c in chunks)
    assert all(approx_tokens(c.text) <= 100 + 15 + 5 for c in chunks)
    # consecutive chunks share text (overlap) and nothing is lost
    for a, b in zip(chunks, chunks[1:]):
        assert a.text.split()[-1] in b.text.split()[:20]
    assert {w for c in chunks for w in c.text.split()} == set(words)


def test_oversized_paragraph_is_split():
    text = "# T\n\n" + " ".join(f"k{i}" for i in range(1000))
    chunks = chunk_text(text, max_tokens=100, overlap=10)
    assert len(chunks) >= 10
    assert all(approx_tokens(c.text) <= 130 for c in chunks)


def test_empty_and_invalid():
    assert chunk_text("") == []
    assert chunk_text("# Only heading") == []
    try:
        chunk_text("x", max_tokens=10, overlap=10)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
