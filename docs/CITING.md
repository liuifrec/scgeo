# Cite ScGeo and report your analysis

[Paper](https://doi.org/10.1093/bioadv/vbag266) · [BibTeX](../CITATION.bib) · [RIS](../CITATION.ris) · [CITATION.cff](../CITATION.cff)

## Paper citation

If you use ScGeo or adapt the accompanying analysis workflows, please cite:

Liu Y-C, Yoshida K. **ScGeo reveals non-canonical trajectories beyond RNA
velocity in radiation-induced hematopoietic recovery.** *Bioinformatics Advances*.
2026;6(1):vbag266. [doi:10.1093/bioadv/vbag266](https://doi.org/10.1093/bioadv/vbag266).

The article was published online on 9 September 2026. Use
[`CITATION.bib`](../CITATION.bib) for BibTeX or import
[`CITATION.ris`](../CITATION.ris) into a compatible reference manager, including
Zotero, EndNote, or Mendeley. GitHub's **Cite this repository** entry uses
[`CITATION.cff`](../CITATION.cff), which identifies the paper as the preferred citation.

```bibtex
@article{Liu2026ScGeo,
  author = {Liu, Yu-Chen and Yoshida, Kengo},
  title = {{ScGeo} reveals non-canonical trajectories beyond {RNA} velocity in radiation-induced hematopoietic recovery},
  journal = {Bioinformatics Advances},
  year = {2026},
  volume = {6},
  number = {1},
  pages = {vbag266},
  doi = {10.1093/bioadv/vbag266},
  url = {https://doi.org/10.1093/bioadv/vbag266}
}
```

The article DOI identifies the scientific paper. Record the software version or
commit separately; a distinct software-archive DOI is not provided by these
repositories. Citation does not replace the conditions of the
[MIT license](../LICENSE).

## Details to include in Methods

| Detail | What to report |
| --- | --- |
| Software provenance | ScGeo version and, for source installs, the exact commit; Python and relevant dependency versions |
| Experimental design | Conditions, contrast direction, state labels, biological-sample unit and counts |
| Representations | Construction/preprocessing, dimensions, primary ensemble, and related sensitivity views |
| Displacement and uncertainty | Center estimator, normalization, bootstrap unit, repetitions, seed, and coverage rules |
| Additional evidence | The local-geometry, dynamics, abundance, or distributional analyses actually run, including their settings |
| Inference scope | Descriptive or biological-sample-aware analysis; cross-sectional or longitudinal design; unresolved coverage/sensitivity |
| Reproducibility | Input accessions/checksums, workflow/configuration files, and figure-source tables |

Use the following as a starting point for a **sample-aware displacement
analysis**, replacing every bracketed field and retaining only statements that
describe your actual analysis:

```text
Cell-state displacement was quantified using ScGeo [version/commit]
(Liu and Yoshida, 2026) in [representation and dimensions]. We compared
[group 1] with [group 0] within [state labels], using [center estimator]
and [normalization]. Uncertainty was summarized by resampling
[independent biological-sample unit], with [counts] samples per condition,
[number] bootstrap repetitions, and random seed [seed].
```

If representation stability was assessed, add the prespecified representation
set, sensitivity analyses, and the decision rules. If biological-sample identity
was unavailable, describe the analysis as cell-level/descriptive instead of
using the sample-aware template. Bootstrap magnitude intervals alone are not
null-hypothesis tests. See [Result interpretation](RESULT_INTERPRETATION.md).

## Attribute the data and upstream methods

Cite the original datasets and any preprocessing, integration, representation,
RNA-velocity, fate-inference, or comparator methods used in your analysis.
The companion's [dataset map](https://github.com/liuifrec/scgeo-notebooks/blob/main/docs/DATASET_MAP.md)
links the public accessions and workflow entry points. Cite official R Augur
when that comparator is used; the retained Python approximation has a different
role and should be identified accordingly.

For adaptation or reproduction of a companion workflow, also record its exact
repository commit and configuration. See the companion
[publication guide](https://github.com/liuifrec/scgeo-notebooks/blob/main/docs/PUBLICATION_GUIDE.md)
and [Reproducibility](REPRODUCIBILITY.md).
