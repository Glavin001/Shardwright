# Third-party notices

## CoACD

`crates/frac-collision/src/coacd.rs` is a Rust port of algorithms from
CoACD (Xinyue Wei, Minghua Liu, Zhan Ling, Hao Su, "Approximate Convex
Decomposition for 3D Meshes with Collision-Aware Concavity and Tree
Search", ACM SIGGRAPH 2022), reimplemented from the reference
implementation at <https://github.com/SarahWeiii/CoACD>, which is
distributed under the following license:

```
MIT License

Copyright (c) 2022 Xinyue Wei, Minghua Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The differential harness `tools/harness/coacd_diff.py` calls the upstream
`coacd` Python package (same license) for comparison only; it is not a
dependency of the library.
