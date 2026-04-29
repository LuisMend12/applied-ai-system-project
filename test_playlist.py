"""Automated tests for playlist_logic.py — verifies core logic and bug fixes."""

import pytest

from playlist_logic import (
    DEFAULT_PROFILE,
    build_playlists,
    classify_song,
    compute_playlist_stats,
    lucky_pick,
    normalize_song,
    search_songs,
)


# ---------------------------------------------------------------------------
# normalize_song
# ---------------------------------------------------------------------------

def test_normalize_preserves_artist_case():
    """Bug fix #1: normalize_artist used to lowercase; now preserves display case."""
    raw = {"title": "Hello", "artist": "The Beatles", "genre": "rock", "energy": 7, "tags": []}
    result = normalize_song(raw)
    assert result["artist"] == "The Beatles"


def test_normalize_strips_whitespace():
    raw = {"title": "  Hello  ", "artist": "  Band  ", "genre": "ROCK", "energy": 8, "tags": []}
    result = normalize_song(raw)
    assert result["title"] == "Hello"
    assert result["genre"] == "rock"


def test_normalize_energy_string():
    raw = {"title": "T", "artist": "A", "genre": "pop", "energy": "7", "tags": []}
    assert normalize_song(raw)["energy"] == 7


def test_normalize_missing_keys():
    result = normalize_song({})
    assert result["title"] == ""
    assert result["energy"] == 0


# ---------------------------------------------------------------------------
# classify_song
# ---------------------------------------------------------------------------

def test_classify_hype_by_energy():
    song = {"title": "Banger", "artist": "X", "genre": "electronic", "energy": 9, "tags": []}
    assert classify_song(song, DEFAULT_PROFILE) == "Hype"


def test_classify_chill_by_energy():
    song = {"title": "Calm", "artist": "X", "genre": "electronic", "energy": 2, "tags": []}
    assert classify_song(song, DEFAULT_PROFILE) == "Chill"


def test_classify_mixed():
    song = {"title": "Neutral", "artist": "X", "genre": "electronic", "energy": 5, "tags": []}
    assert classify_song(song, DEFAULT_PROFILE) == "Mixed"


# ---------------------------------------------------------------------------
# search_songs — Bug fix #4: was `value in q` (backwards), now `q in value`
# ---------------------------------------------------------------------------

def test_search_partial_match():
    """Partial query should match songs where artist contains the query string."""
    songs = [
        {"title": "A", "artist": "Nina Simone", "genre": "jazz", "energy": 5, "tags": []},
        {"title": "B", "artist": "Nirvana", "genre": "rock", "energy": 8, "tags": []},
    ]
    results = search_songs(songs, "nina", field="artist")
    assert len(results) == 1
    assert results[0]["artist"] == "Nina Simone"


def test_search_empty_query_returns_all():
    songs = [{"title": "A", "artist": "X", "genre": "rock", "energy": 5, "tags": []}]
    assert search_songs(songs, "") == songs


def test_search_no_match():
    songs = [{"title": "A", "artist": "Jazz Band", "genre": "jazz", "energy": 3, "tags": []}]
    assert search_songs(songs, "metallica", field="artist") == []


# ---------------------------------------------------------------------------
# lucky_pick — Bug fix #5: random.choice crashed on empty list
# ---------------------------------------------------------------------------

def test_lucky_pick_empty_hype_returns_none():
    result = lucky_pick({"Hype": [], "Chill": []}, mode="hype")
    assert result is None


def test_lucky_pick_empty_any_returns_none():
    result = lucky_pick({"Hype": [], "Chill": []}, mode="any")
    assert result is None


def test_lucky_pick_returns_song():
    song = {"title": "T", "artist": "A", "genre": "rock", "energy": 8, "tags": [], "mood": "Hype"}
    result = lucky_pick({"Hype": [song], "Chill": []}, mode="hype")
    assert result == song


# ---------------------------------------------------------------------------
# compute_playlist_stats — Bug fixes #2 and #3
# ---------------------------------------------------------------------------

def test_stats_hype_ratio_uses_all_songs():
    """Bug fix #2: denominator was len(hype), not len(all_songs)."""
    playlists = {
        "Hype": [{"energy": 8}],
        "Chill": [{"energy": 2}, {"energy": 1}],
        "Mixed": [],
    }
    stats = compute_playlist_stats(playlists)
    assert stats["total_songs"] == 3
    assert abs(stats["hype_ratio"] - 1 / 3) < 0.001


def test_stats_avg_energy_uses_all_songs():
    """Bug fix #3: total_energy only summed hype songs, not all songs."""
    playlists = {
        "Hype": [{"energy": 8}],
        "Chill": [{"energy": 2}],
        "Mixed": [],
    }
    stats = compute_playlist_stats(playlists)
    assert stats["avg_energy"] == pytest.approx(5.0)


def test_stats_empty_playlists():
    stats = compute_playlist_stats({"Hype": [], "Chill": [], "Mixed": []})
    assert stats["total_songs"] == 0
    assert stats["hype_ratio"] == 0.0
    assert stats["avg_energy"] == 0.0


# ---------------------------------------------------------------------------
# build_playlists (integration)
# ---------------------------------------------------------------------------

def test_build_playlists_groups_hype_and_chill():
    songs = [
        {"title": "Loud", "artist": "X", "genre": "rock", "energy": 9, "tags": []},
        {"title": "Quiet", "artist": "Y", "genre": "ambient", "energy": 1, "tags": []},
    ]
    playlists = build_playlists(songs, DEFAULT_PROFILE)
    assert any(s["title"] == "Loud" for s in playlists["Hype"])
    assert any(s["title"] == "Quiet" for s in playlists["Chill"])
