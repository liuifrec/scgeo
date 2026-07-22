# Analysis layers

## Robust shift

`scgeo.tl.robust_shift` estimates a state-level displacement in a chosen
representation. With `sample_key`, biological-sample centers are the relevant
units for sample-aware comparison. Report both the raw and normalized
displacement, the direction, the estimator, coverage, and the normalization
scale.

## Biological-sample uncertainty

When independent samples exist, resampling should occur at the biological
sample level. Cell resampling can describe computational stability but cannot
replace biological replication. Small sample counts limit interval precision
and exact-permutation resolution; they do not justify relabeling cells as
replicates.

## Representation stability

`scgeo.tl.representation_stability` compares prespecified representations. Mark
which representations form the primary consensus and which are sensitivity
views. Nested PCA dimensions share a basis and are not independent evidence.
Do not interpret high biological-condition mixing as automatically desirable.

## Local geometry

`scgeo.tl.local_geometry_stability` reports neighborhood overlap, neighbor
Jaccard, and local-shape distortion across representations and neighborhood
sizes. These diagnostics reveal representation sensitivity; they do not prove
universal biological preservation.

## Dynamics agreement

`scgeo.tl.velocity_shift_alignment` compares displacement with supplied
dynamics coordinates when those coordinates are available and scientifically
justified. Alignment and discordance are conditional on both representations.
Unavailable or inconsistent dynamics should remain visible rather than being
forced into a directional conclusion.

## Abundance and distribution

`scgeo.tl.mixscore` and `scgeo.tl.distribution_test` quantify related but
distinct structure. State proportions describe abundance. Energy distance,
MMD, sliced Wasserstein, or other distributional statistics describe shape or
separability at their stated comparison unit. Pooled-cell distances are
descriptive unless the design supplies a valid biological-sample analysis.

See [Result interpretation](RESULT_INTERPRETATION.md) for recommended reporting
language.
