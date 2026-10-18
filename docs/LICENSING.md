# Licensing, attribution and citation

## Author-owned materials

The public-release licences are:

- **MIT** to author-owned code, subject to the notices accompanying the original files.
- **CC BY 4.0** to author-generated data, trained actors and figures.
- The existing upstream terms to third-party materials.

The complete texts are in [`../LICENSE`](../LICENSE) and [`../LICENSES/`](../LICENSES/). Attribution and a link to the licence are required when reusing CC BY materials; indicate modifications where applicable.

## Historical copyright is not the current author list

The original scientific archive's MIT notice reads:

> Copyright (c) 2026 Zhuang Shao, Lijun Lei, Peng Wang, Liang Zheng, Jie Zhou.

This repository preserves that notice byte for byte. It records the existing code-licensing notice; it is not rewritten to match the manuscript byline. Preservation does not assert a new copyright transfer or a finding about individual ownership.

The associated current manuscript lists Zhuang Shao, Jiamei Yang, Lijun Lei, Bingnan Li, Ruifeng Ding and Zhenlong Wu. [`../CITATION.cff`](../CITATION.cff) records this six-person research-artifact citation. Citation authorship is distinct from the copyright notices on historical code and from the authorship of upstream dependencies.

## Third-party contributions

The unchanged [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) records the release's source provenance, including:

| Source | Preserved version | Terms |
|---|---|---|
| Wind-RL | `98d056e64e655a09d8d125cca47058ac3011d8e7` | MIT; original Meta Platforms notice retained. |
| Incompact3d | `abb010e615cff520f949a210278945346995966c` | BSD 3-Clause; Xcompact3d source notices retained. |
| SciPilot plotting and QA utilities | Preserved source identities in the scientific archives | MIT; Copyright (c) 2026 Haojae retained. |

The code derives from and compares with named upstream implementations; their notices do not make the research authors owners of those upstream contributions. The upstream study’s original dataset is not redistributed. Python, CUDA and other external runtime dependencies retain their own licences and are not redistributed as installed binaries here.

The root third-party notice was copied verbatim from the scientific archive. Phrases such as “enclosed archives” describe that artifact's packaging; the browsable `science/` directory retains its own archive-member mapping.

## Distribution scope

This public repository and release exclude the preserved Docker image, complete training-resume states, full inlet histories, manuscript sources/full-paper PDFs and private submission records. The licences here do not authorize redistribution of those excluded items.

No DOI or journal-publication metadata has been assigned in this repository. Cite the specific versioned research artifact until a publication record exists.
