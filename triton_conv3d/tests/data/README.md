# Recorded convolution corpora

Fixtures for `triton_conv3d/tests` and `triton_conv3d/bench`; nothing on the
runtime path reads them.  Loaded through `triton_conv3d.shapes`.

| file | what it is | regenerate with |
|---|---|---|
| `scaffold_corpus.json` | the distinct convolutions of three profiled configurations, ordered by measured MIOpen cost | `make_corpus.py` |
| `profile_points.json` | the MIOpen profile record `make_corpus.py` joins in: per-step ms, FLOPs, bytes and solver per convolution kernel | a profiled run; a measurement, kept as data |
| `scaffold_census.json` | every convolution an instrumented step issued at the four benchmark configurations, in the form the kernel was handed | `make_census.py` |
| `census/cens_{A,B,C,D}.json` | the instrumented runs' captures (three steps, `n_categories: 2`), rank 0 of each | `conv_census.py` |

`make_corpus.py` traces the shapes with `model-analysis/unet_shapes.py`; both
generators take `--check`, which says whether the committed file would change.

`conv_census.py` wraps the shipped `scaffold benchmark` command:

    python triton_conv3d/tests/data/conv_census.py --out cens_A.json -- \
        benchmark -c <config.yml>

and writes `cens_A[.rN].census.json` at exit; copy rank 0's capture into
`census/` and run `make_census.py`.
