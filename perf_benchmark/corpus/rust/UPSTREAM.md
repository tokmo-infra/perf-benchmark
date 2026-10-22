# Trace hash inflator provenance

The block sampler and hash reuse behavior in `src/lib.rs` are adapted from
[`blitz-serving/trace-replayer`](https://github.com/blitz-serving/trace-replayer/tree/84c9193ac67ebbf0e08db161bc9b048d597c2f80),
commit `84c9193ac67ebbf0e08db161bc9b048d597c2f80`, specifically
`src/token_sampler.rs` and `src/dataset.rs`. The upstream project is licensed
under [Apache License 2.0](https://github.com/blitz-serving/trace-replayer/blob/84c9193ac67ebbf0e08db161bc9b048d597c2f80/LICENSE).

This adaptation retains batch token sampling, decode/re-encode block length
validation, bounded producer queues, complete-block hash reuse, and partial
tail replacement. Producer shutdown, failure propagation, the 4 GiB cache
budget, and lazy partial-block generation are local changes for offline corpus
preparation. `tokenizers` is pinned to `0.22.2` in Rust and Python.
