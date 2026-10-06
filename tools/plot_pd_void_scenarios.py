"""Export standalone scientific figures from conditional PD scenario JSON."""
from pathlib import Path
import argparse
import json
import math

def make_figures(input_path, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    data = json.loads(Path(input_path).read_text(encoding='utf-8'))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    colors = {0.3: '#b57400', 0.5: '#087f8c', 0.8: '#5266a4'}
    for mr in data['model_results']:
        result = mr['result']
        ratio = result['input']['void']['extinction_ratio']
        if mr['case'] != 'primary':
            continue
        points = []
        for radius in result['radius_cases']:
            scenario = next(s for s in radius['scenarios'] if s['voltage_rms_V'] == 20000)
            points.append((radius['radius_m'] * 1e6, scenario['q_peak_pC']))
        x, y = zip(*points)
        ax.plot(x, y, marker='o', color=colors[ratio], label=f'Primary: Eext/Einc = {ratio:g}')
    ax.axhline(15, color='#be4141', linestyle='--', label='15 pC reference')
    ax.plot([25, 500], [0, 0], color='#717b89', linestyle=':', label='Secondary selected point: no events in these scenarios')
    ax.set(xlabel='Assumed spherical void radius [µm]', ylabel='Maximum first-cycle induced charge [pC]',
           title='Conditional spherical-void estimate at 20 kV RMS')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc='upper left')
    fig.text(.02, .015, 'Assumed air 101325 Pa, host εr=4, immediate seed, no wall-charge decay.\nActual defect/material/test conditions and finite-host corrections are unverified.', fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    radius_path = output / 'pd_charge_vs_radius.svg'
    fig.savefig(radius_path, bbox_inches='tight')
    fig.savefig(output / 'pd_charge_vs_radius.png', dpi=170, bbox_inches='tight')
    plt.close(fig)

    mr = next(m for m in data['model_results'] if m['case'] == 'primary'
              and m['result']['input']['void']['extinction_ratio'] == 0.5)
    radius = next(r for r in mr['result']['radius_cases'] if math.isclose(r['radius_m'], 300e-6))
    scenario = next(s for s in radius['scenarios'] if s['voltage_rms_V'] == 20000)
    events = scenario['events']
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.6, 7), sharex=True)
    amp = scenario['void_peak_field_without_wall_memory_V_per_m']
    grid = np.linspace(0, 2 * math.pi, 1601)
    applied = amp * np.sin(grid)
    memory = np.zeros_like(grid)
    for event in events:
        memory[grid >= event['phase_rad']] = event['wall_memory_after_V_per_m']
    phase = np.degrees(grid)
    ax0.plot(phase, applied / 1e6, color='#a4adba', label='Applied cavity field, without wall charge')
    ax0.plot(phase, (applied + memory) / 1e6, color='#087f8c', label='Cavity field with assumed wall-charge memory')
    inc = radius['inception_field_V_per_m'] / 1e6
    ax0.axhline(inc, color='#be4141', linestyle='--', linewidth=.8)
    ax0.axhline(-inc, color='#be4141', linestyle='--', linewidth=.8)
    ax0.set(ylabel='Cavity field [MV/m]', title='Conditional first virgin cycle: radius 300 µm, 20 kV RMS, 60 Hz')
    ax0.legend(fontsize=8)
    ax0.grid(alpha=.2)
    for event in events:
        q = event['q_app_pC']
        color = '#087f8c' if q >= 0 else '#5266a4'
        ax1.vlines(event['phase_deg'], 0, q, color=color, alpha=.5)
        ax1.scatter([event['phase_deg']], [q], color=color, s=25)
    ax1.axhline(15, color='#be4141', linestyle='--', linewidth=.8)
    ax1.axhline(-15, color='#be4141', linestyle='--', linewidth=.8)
    ax1.set(xlabel='AC phase [degrees]', ylabel='Induced terminal event charge [pC]', xlim=(0, 360))
    ax1.set_xticks(range(0, 361, 45))
    ax1.grid(alpha=.2)
    fig.text(.02, .013, 'First-cycle deterministic pulses; this is not a measured or steady-state PRPD pattern.\nNo statistical time lag, relaxation, multi-void interaction, or measurement-circuit calibration.', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    event_path = output / 'pd_first_cycle_events.svg'
    fig.savefig(event_path, bbox_inches='tight')
    fig.savefig(output / 'pd_first_cycle_events.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    return {'charge_vs_radius': str(radius_path), 'first_cycle_events': str(event_path)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(make_figures(args.input, args.output_dir)))

if __name__ == '__main__':
    main()
