#!/usr/bin/env python3
"""Render frozen ScGeo evidence and projected velocity from explicit local inputs.

No downloads, fits, source rewrites, or implicit substitutions are performed.
Never commit in-house datasets or derived per-mouse tables to a public repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import pandas as pd

import scgeo as sg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('adata', 'shifts', 'alignments'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--condition-key', required=True)
    parser.add_argument('--node-key', required=True)
    parser.add_argument('--group0', required=True)
    parser.add_argument('--group1', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--basis', default='umap')
    parser.add_argument('--state-key', default='state')
    parser.add_argument('--alignment-state-key', default='leiden')
    parser.add_argument('--analysis-rep', default='X_pca')
    parser.add_argument('--paga-key', default='paga')
    parser.add_argument('--mode', choices=('grid', 'stream'), default='grid')
    args = parser.parse_args()
    for filename in (args.adata, args.shifts, args.alignments):
        if not filename.is_file():
            raise FileNotFoundError(f'Required saved input does not exist: {filename}')
    inputs = {name: {'file': str(getattr(args, name)),
                     'sha256': hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()}
              for name in ('adata', 'shifts', 'alignments')}
    mouse = ad.read_h5ad(args.adata)
    shift = pd.read_csv(args.shifts, sep='\t')
    alignment = pd.read_csv(args.alignments, sep='\t')
    if args.node_key not in mouse.obs:
        raise KeyError(f'{args.node_key!r} not found in saved AnnData observations')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    concordance = sg.pl.geometry_velocity_concordance(
        shift, alignment, state_key=args.state_key,
        alignment_state_key=args.alignment_state_key,
        analysis_rep=args.analysis_rep, show=False, return_result=True,
    )
    outputs['concordance'] = concordance.save(
        args.output_dir / 'geometry_velocity_concordance', formats=('png', 'svg', 'pdf')
    )
    focus = concordance.data['eligible']['_state_id'].tolist()
    node_order = (mouse.obs[args.node_key].cat.categories.tolist()
                  if isinstance(mouse.obs[args.node_key].dtype, pd.CategoricalDtype)
                  else None)
    paga = sg.pl.paga_shift_map(
        mouse, node_key=args.node_key, condition_key=args.condition_key,
        group0=args.group0, group1=args.group1,
        paga_key=args.paga_key, basis=args.basis,
        graph_node_order=node_order,
        focus_states=focus, edge_focus='incident', label_top_n=0,
        return_result=True, show=False,
    )
    outputs['focused_paga'] = paga.save(
        args.output_dir / 'focused_paga', formats=('png', 'svg', 'pdf')
    )
    velocity = sg.pl.velocity_condition_compare(
        mouse, condition_key=args.condition_key,
        group0=args.group0, group1=args.group1,
        basis=args.basis, mode=args.mode, return_result=True, show=False,
    )
    outputs['velocity_conditions'] = velocity.save(
        args.output_dir / 'velocity_conditions', formats=('png', 'svg', 'pdf')
    )
    report = {
        'inputs': inputs,
        'source_cells': mouse.n_obs,
        'eligible_states': len(focus),
        'unavailable_states': concordance.metadata['n_not_assessed'],
        'velocity_support': {
            key: val for key, val in velocity.metadata.items() if key.startswith('n_')
        },
        'outputs': {key: str(value.get('metadata')) for key, value in outputs.items()},
        'interpretation': 'descriptive frozen analysis; no new fit or mouse-level hypothesis test',
    }
    (args.output_dir / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
