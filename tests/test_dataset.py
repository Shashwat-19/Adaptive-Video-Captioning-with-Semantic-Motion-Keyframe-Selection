import json

import pytest

pd = pytest.importorskip("pandas")

from video_captioning.data.dataset import (
    index_video_paths,
    normalize_captions,
    normalize_segment,
    read_caption_table,
    split_by_video_id,
)


def test_normalize_captions_accepts_common_msvd_column_names():
    raw = pd.DataFrame(
        {
            "Video Name": ["clip_a.avi", "clip_b.mp4", ""],
            "Description": ["A person walks", "A dog runs", "Ignored"],
        }
    )

    normalized = normalize_captions(raw)

    assert normalized[["video_id", "caption"]].to_dict("records") == [
        {"video_id": "clip_a", "caption": "A person walks"},
        {"video_id": "clip_b", "caption": "A dog runs"},
    ]


def test_normalize_captions_cleans_the_official_msvd_corpus(tmp_path):
    # A missing value makes pandas read Start and End as floats, as in the full official CSV.
    path = tmp_path / "MSR Video Description Corpus.csv"
    path.write_text(
        "VideoID,Start,End,WorkerID,Source,AnnotationTime,Language,Description\n"
        "abc,1,5,1,clean,10,English,A  man   is talking\n"
        "abc,1,5,2,clean,10,French,Un homme parle\n"
        "abc,1,5,3,clean,10,English,\n"
        "xyz,,9,4,clean,10,English,A dog runs\n"
        "def,2.5,7,5,clean,10, ENGLISH ,A cat sits\n",
        encoding="utf-8",
    )

    normalized = normalize_captions(read_caption_table(path))

    assert normalized[["video_id", "caption"]].to_dict("records") == [
        {"video_id": "abc_1_5", "caption": "A man is talking"},
        {"video_id": "def_2.5_7", "caption": "A cat sits"},
    ]


def test_normalize_captions_language_filter_can_be_disabled():
    raw = pd.DataFrame(
        {"video_id": ["a", "b"], "caption": ["A dog runs", "Un chien court"], "language": ["English", "French"]}
    )

    assert normalize_captions(raw)["caption"].tolist() == ["A dog runs"]
    assert normalize_captions(raw, language=None)["caption"].tolist() == ["A dog runs", "Un chien court"]


def test_normalize_captions_returns_empty_when_no_row_matches_the_language():
    raw = pd.DataFrame({"VideoID": ["a"], "Start": [1], "End": [2], "Language": ["French"], "Description": ["x"]})

    assert normalize_captions(raw).empty


def test_normalize_captions_drops_blank_segment_bounds():
    # Kept as a separate frame: in the CSV test a blank string would turn the float Start column into strings.
    raw = pd.DataFrame({"video_id": ["a", "b"], "start": [" ", "3"], "end": ["4", "5"], "caption": ["x", "y"]})

    assert normalize_captions(raw)["video_id"].tolist() == ["b_3_5"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(33, "33"), (33.0, "33"), ("33.0", "33"), (" 7 ", "7"), (2.5, "2.5"), ("clip", "clip"), ("", ""), ("inf", "inf")],
)
def test_normalize_segment_matches_video_file_stems(value, expected):
    assert normalize_segment(value) == expected


def test_video_level_splits_are_disjoint():
    captions = pd.DataFrame(
        {"video_id": [f"video_{index}" for index in range(20)], "caption": ["caption"] * 20}
    )

    splits = split_by_video_id(captions, seed=7)

    assert not (splits["train"] & splits["validation"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["validation"] & splits["test"])
    assert set().union(*splits.values()) == set(captions["video_id"])


def test_read_caption_table_parses_csv_and_json(tmp_path):
    rows = [{"video_id": "clip_a", "caption": "a dog runs"}, {"video_id": "clip_b", "caption": "a cat sits"}]
    (tmp_path / "captions.csv").write_text("video_id,caption\nclip_a,a dog runs\nclip_b,a cat sits\n", encoding="utf-8")
    (tmp_path / "list.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "wrapped.json").write_text(json.dumps({"data": rows}), encoding="utf-8")

    for name in ["captions.csv", "list.json", "wrapped.json"]:
        assert read_caption_table(tmp_path / name)[["video_id", "caption"]].to_dict("records") == rows


def test_read_caption_table_strips_a_utf8_byte_order_mark(tmp_path):
    (tmp_path / "captions.csv").write_text(
        "VideoID,Start,End,Language,Description\nabc,1,5,English,A dog runs\n", encoding="utf-8-sig"
    )
    (tmp_path / "captions.json").write_text(
        json.dumps([{"video_id": "abc", "caption": "A dog runs"}]), encoding="utf-8-sig"
    )

    assert list(read_caption_table(tmp_path / "captions.csv").columns)[0] == "VideoID"
    assert normalize_captions(read_caption_table(tmp_path / "captions.csv"))["video_id"].tolist() == ["abc_1_5"]
    assert read_caption_table(tmp_path / "captions.json")["video_id"].tolist() == ["abc"]


def test_read_caption_table_splits_tsv_on_tabs_even_when_every_line_has_a_comma(tmp_path):
    # A comma on every line makes delimiter sniffing pick ",", so only the explicit .tsv rule parses this.
    path = tmp_path / "captions.tsv"
    path.write_text(
        "video_id\tcaption\tsource, round\nclip_a\ta dog, runs\tmturk, 1\nclip_b\ta cat, sits\tmturk, 2\n",
        encoding="utf-8",
    )

    assert read_caption_table(path)[["video_id", "caption"]].to_dict("records") == [
        {"video_id": "clip_a", "caption": "a dog, runs"},
        {"video_id": "clip_b", "caption": "a cat, sits"},
    ]


def test_index_video_paths_keys_supported_files_by_stem(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "clip_a.avi").touch()
    (tmp_path / "nested" / "clip_b.MP4").touch()
    (tmp_path / "notes.txt").touch()

    index = index_video_paths(tmp_path)

    assert index == {"clip_a": tmp_path / "clip_a.avi", "clip_b": tmp_path / "nested" / "clip_b.MP4"}
    assert index_video_paths(tmp_path / "missing") == {}
