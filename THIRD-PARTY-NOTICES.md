# Third-party notices

This project is MIT licensed (see [`LICENSE`](LICENSE)). It contains **one**
third-party artifact and bundles **no** third-party runtime code.

## Test fixture: `test/fixtures/ascend-trace_view.sample.json`

* **Source**: [`Ascend/mstt`](https://github.com/Ascend/mstt) —
  `profiler/msprof_analyze/test/ut/advisor/advisor_backend/timeline_advice/trace_view.json`
* **License**: Apache License 2.0
* **Modification**: truncated to roughly the first 100 KB of the ~635 KB original
  and cut off mid-event (deliberately, so the parser's tolerance for an export
  interrupted before its closing bracket stays covered by tests). No other change
  was made.

This fixture is the only third-party content in the repository, and it is used
solely as test input. See [`test/fixtures/README.md`](test/fixtures/README.md)
for what it pins down.

## Runtime dependencies

None. The plugin and its browser page use only Node.js built-in modules
(`node:fs`, `node:zlib`, `node:crypto`, `node:path`, …) and browser-native APIs
(Canvas 2D, SVG, `fetch`, `XMLHttpRequest`). Nothing is installed at runtime, and
no front-end library is vendored — which is why the package has an empty
dependency list and can be installed from a git URL without a build step.

## Documentation references

`docs/research/` summarises publicly available Ascend/CANN behaviour with links to
the corresponding official documentation and source files. Those documents quote
file names, column headers, constant names, and field values — factual interface
details needed to parse the format correctly — and attribute every claim to its
source URL. No third-party source code is copied into this repository.
