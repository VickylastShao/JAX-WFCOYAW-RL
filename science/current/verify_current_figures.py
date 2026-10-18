#!/usr/bin/env python3
"""Verify the 18 current scientific figures and seven controller display names."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

EXPECTED = {
    'zero_yaw': 'Zero yaw', 'neutral_actor_step32768': 'Actor at 32,768',
    'static_les_lookup': 'Static LES lookup',
    'selected_policy_direction_mean': 'Direction-mean control',
    'selected_realtime_policy': 'Real-time policy',
    'selected_probe_mean_ablation': 'Mean-probe ablation',
    'cross_precursor_action_replay': 'Cross-precursor action replay',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    index = json.loads((root / 'manuscript/FIGURE_UPLOAD_INDEX.json').read_text())
    mapping = json.loads((root / 'figure_sources/CONTROLLER_DISPLAY_TERMS.json').read_text())
    assert mapping['controller_id_to_display_name'] == EXPECTED
    assert len(index['figures']) == 18
    files = []
    for figure in index['figures']:
        assert len(figure['formats']) == 3
        assert sorted(Path(f['path']).suffix for f in figure['formats']) == ['.pdf', '.png', '.svg']
        for item in figure['formats']:
            path = root / item['path']
            assert path.is_file() and path.stat().st_size == item['bytes'], item['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['path']
            files.append(item['path'])
    svg = ET.parse(root / 'manuscript/figures/n9_feedback_matrix.svg').getroot()
    visible = [''.join(element.itertext()) for element in svg.iter() if element.tag.split('}')[-1] == 'text']
    text = '\n'.join(visible)
    for label in EXPECTED.values():
        assert label in text, label
    report = dict(schema='current-final-polish-figure-acceptance/1', status='pass',
        figure_count=18, format_count=len(files), controller_count=7, controller_id_to_display_name=EXPECTED,
        current_files_sha_verified=True, original_ids_renamed=False, new_GPU_hours=0)
    if args.output:
        output = args.output.resolve()
        if output.is_relative_to(root):
            raise ValueError('Write the report outside the immutable delivery')
        output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
