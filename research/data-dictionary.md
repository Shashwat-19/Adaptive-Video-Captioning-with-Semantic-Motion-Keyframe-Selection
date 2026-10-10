# MSVD Data Dictionary

How caption tables and video files are read, normalized, matched, and split. The library path (`src/video_captioning/data/dataset.py`, used by `scripts/`) and the committed notebook normalize the official corpus in nearly the same way and read it with the same encoding; the remaining differences are described below.

## Caption table formats

`read_caption_table(path)` chooses a parser from the file extension:

| Extension | Parser |
|---|---|
| `.json` | A JSON list of records, or an object whose `data` key holds the list (`pd.json_normalize`) |
| `.tsv` | Tab-separated (`sep="\t"`) |
| anything else (e.g. `.csv`) | Delimiter sniffed by pandas (`sep=None`, Python engine) |

Every format is decoded as `utf-8-sig`, so a file with or without a UTF-8 byte-order mark reads the same.

## Column mapping (`normalize_captions`)

Column names are matched after stripping, lowercasing, and replacing spaces and hyphens with underscores, so `Video Name`, `video-name`, and `video_name` are equivalent. For each role, the first candidate present wins:

| Role | Candidates, in priority order | Required |
|---|---|---|
| Video ID | `video_id`, `video`, `video_name`, `videoid`, `clip_id`, `filename`, `file_name` | Yes |
| Caption | `caption`, `description`, `sentence`, `text`, `caption_text` | Yes |
| Segment start | `start`, `start_time`, `start_frame` | No |
| Segment end | `end`, `end_time`, `end_frame` | No |
| Language | `language` | No |

A table without a video-ID or caption column raises `ValueError`. All other columns are kept unchanged.

When a language column is present, only rows whose value matches the `language` argument (default `"english"`, compared stripped and case-insensitively) are kept. Pass `language=None` to keep every row. Rows with a missing video ID or caption are dropped, and so are rows with a missing segment bound when both segment columns are present.

## Output fields

| Field | Rule |
|---|---|
| `video_id` | With both segment columns: `<id>_<start>_<end>`, each part stripped. Each segment bound goes through `normalize_segment`, so `33`, `33.0`, and `"33.0"` all become `33` and `2.5` stays `2.5`. Otherwise: the ID with a trailing `.avi`, `.mp4`, `.webm`, `.mov`, or `.mkv` removed (case-insensitive). |
| `caption` | The caption text, with repeated whitespace collapsed to one space and the ends stripped. |

Rows whose `video_id`, `caption`, or a segment bound is an empty string are dropped.

## Video files (`index_video_paths`)

Files under the video directory (searched recursively) with extension `.avi`, `.mp4`, `.webm`, `.mov`, or `.mkv` (case-insensitive) are indexed by filename stem. A caption row matches a video only when its `video_id` equals that stem exactly. A missing video directory gives an empty index.

## Official MSVD corpus

The official caption file is `MSR Video Description Corpus.csv`. The committed notebook (Section 2b) requires the columns `VideoID`, `Start`, `End`, `Language`, and `Description`, and builds `video_id` as `<VideoID>_<Start>_<End>` to match the video file stems. Both the notebook and the library's `normalize_captions`:

- keep English captions only (`Language` compared case-insensitively to `english`);
- normalize segment values so `33`, `33.0`, and `"33.0"` all become `33` (the library's `normalize_segment` is a port of the notebook's);
- drop rows with a missing `Description` and collapse repeated whitespace.

They differ on a missing `Start` or `End`: the notebook raises (and stops the cell), while the library drops the row. The notebook also requires all five columns by name; the library only needs a video-ID and a caption column.

They also read the file slightly differently. Both decode it as `utf-8-sig`, so a UTF-8 byte-order mark is stripped from the first column name. Neither reliably detects malformed rows. Checked on pandas 3.0.6 and 2.2.3:

- A row with too few fields is padded with empty values, and both paths then drop it silently. The library drops it in `normalize_captions`, and the notebook through its empty-`Description` filter, since `Description` is the last column.
- A row with too many fields usually raises `ParserError` in the library (for example 3 or 5 fields in a 2-column file). The notebook skips such a row with a "Skipping line" warning. A row with exactly twice the header's field count is the exception: it silently corrupts the parsed table in both, with no warning. Examples are 4 fields in a 2-column file, or 16 in the 8-column official schema.

This is tracked in `TODO.md` ("`read_caption_table` does not handle a UTF-8 BOM or malformed lines"). Whether the official file has malformed rows is unverified.

## Splits (`split_by_video_id`)

- Splits are made over unique `video_id`s, sorted first, never over caption rows, so every caption of a video lands in the same split.
- Two `sklearn.model_selection.train_test_split` calls with `random_state=seed` give train, validation, and test sets. Defaults: 0.8 / 0.1 / 0.1, seed 42, the same values as `configs/baseline.yaml`.
- The ratios must sum to 1, and at least three unique videos are required.
- The library and `scripts/train.py` do not save split IDs. The notebook writes `train_video_ids.csv`, `val_video_ids.csv`, `test_video_ids.csv`, and `configs/split_metadata.json` under its output directory.

## Example

| Input row (official corpus) | Library `video_id` | Notebook `video_id` |
|---|---|---|
| `VideoID=abc, Start=1, End=5` (integer columns) | `abc_1_5` | `abc_1_5` |
| `VideoID=abc, Start=1.0, End=5` (float column) | `abc_1_5` | `abc_1_5` |
| `VideoID=xyz, Start=<missing>, End=9` | row dropped | cell raises `ValueError` |
| `Language=French` | row dropped | row dropped |
| `Video Name=clip_a.avi` (no segment columns) | `clip_a` | n/a (notebook reads the official corpus) |
