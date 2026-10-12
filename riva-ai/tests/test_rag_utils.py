import pytest
from utils import clean_text, split_chunks

def test_clean_text():
    assert clean_text("Hello    world\n\n\nNext") == "Hello world\n\nNext"

def test_empty_chunks():
    assert split_chunks("") == []

def test_chunk_overlap_and_bounds():
    text = "x" * 2500
    chunks = split_chunks(text, size=1000, overlap=150)
    assert len(chunks) == 3
    assert all(len(c) <= 1000 for c in chunks)
    assert chunks[0][-150:] == chunks[1][:150]

def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_chunks("hello", size=10, overlap=10)
