<!-- spell-checker:ignore println,inproceedings,booktitle -->

# OxiDD

[![Crates.io Version](https://img.shields.io/crates/v/oxidd?style=for-the-badge&logo=rust)](https://crates.io/crates/oxidd)
[![PyPI Version](https://img.shields.io/pypi/v/oxidd?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/oxidd/)
[![Matrix](https://img.shields.io/badge/matrix-join_chat-brightgreen?style=for-the-badge&logo=matrix)](https://matrix.to/#/#oxidd:matrix.org)


These are the Python bindings for OxiDD, a highly modular decision diagram framework written in Rust. The most prominent instance of decision diagrams is provided by [(reduced ordered) binary decision diagrams (BDDs)](https://en.wikipedia.org/wiki/Binary_decision_diagram), which are succinct representations of Boolean functions 𝔹<sup>n</sup> → 𝔹. Such BDD representations are canonical and thus, deciding equality of Boolean functions—in general a co-NP-complete problem—can be done in constant time. Further, many Boolean operations on two BDDs *f,g* are possible in 𝒪(|*f*| · |*g*|) (where |*f*| denotes the node count in *f*). There are various other kinds of decision diagrams for which OxiDD aims to be a framework enabling high-performance implementations with low effort.


## Features

- **Several kinds of (reduced ordered) decision diagrams** are already implemented:
    - Binary decision diagrams (BDDs)
    - BDDs with complement edges (BCDDs)
    - Zero-suppressed BDDs (ZBDDs, aka ZDDs/ZSDDs)
    - Not yet exposed via the Python API: Multi-terminal BDDs (MTBDDs, aka ADDs) and Ternary decision diagrams (TDDs)
- **Extensibility**: Due to OxiDD’s modular design, one can implement new kinds of decision diagrams without having to reimplement core data structures.
- **Concurrency**: Functions represented by DDs can safely be used in multi-threaded contexts. Furthermore, apply algorithms can be executed on multiple CPU cores in parallel.
- **Performance**: Compared to other popular BDD libraries (e.g., BuDDy, CUDD, and Sylvan), OxiDD is already competitive or even outperforms them.
- **Visualization**: Display your DDs with ease through [OxiDD-vis](https://oxidd.net/vis).
- **Support for Reordering**: OxiDD can reorder a decision diagram to a given variable order. Support for dynamic reordering, e.g., via sifting, is about to come.


## Licensing

OxiDD is licensed under either [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) at your opinion.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache 2.0 license, shall be dual licensed as above, without any additional terms or conditions.


## Publications

The [seminal paper](https://doi.org/10.1007/978-3-031-57256-2_13) presenting OxiDD was published at TACAS'24. If you use OxiDD, please cite us as:

Nils Husung, Clemens Dubslaff, Holger Hermanns, and Maximilian A. Köhl: *OxiDD: A safe, concurrent, modular, and performant decision diagram framework in Rust.* In: Proceedings of the 30th International Conference on Tools and Algorithms for the Construction and Analysis of Systems (TACAS’24)

    @inproceedings{oxidd24,
      author        = {Husung, Nils and Dubslaff, Clemens and Hermanns, Holger and K{\"o}hl, Maximilian A.},
      booktitle     = {Proceedings of the 30th International Conference on Tools and Algorithms for the Construction and Analysis of Systems (TACAS'24)},
      title         = {{OxiDD}: A Safe, Concurrent, Modular, and Performant Decision Diagram Framework in {Rust}},
      year          = {2024},
      doi           = {10.1007/978-3-031-57256-2_13}
    }


## Acknowledgements

This work is partially supported by the German Research Foundation (DFG) under the projects TRR 248 (see https://perspicuous-computing.science, project ID 389792660) and EXC 2050/1 (CeTI, project ID 390696704, as part of Germany’s Excellence Strategy).
