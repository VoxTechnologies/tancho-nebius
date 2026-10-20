"""Generate the figures for the project gallery as SVG, then PNG at 1800x1200 (3:2)."""
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures'
W, H = 1800, 1200
INK, MUTED, LINE = '#111827', '#5b6472', '#d6dae1'
ACCENT, WARN, OK = '#1d4ed8', '#b45309', '#15803d'
FONT = 'Helvetica Neue, Helvetica, Arial, sans-serif'
MONO = 'SFMono-Regular, Menlo, Consolas, monospace'


def text(x, y, s, size=30, fill=INK, weight='400', anchor='start', font=FONT):
    return ('<text x="%g" y="%g" font-family="%s" font-size="%g" fill="%s" '
            'font-weight="%s" text-anchor="%s">%s</text>'
            % (x, y, font, size, fill, weight, anchor, s))


def box(x, y, w, h, fill='#ffffff', stroke=LINE, rx=10, width=2, dash=None):
    d = ' stroke-dasharray="%s"' % dash if dash else ''
    return ('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" fill="%s" stroke="%s" '
            'stroke-width="%g"%s/>' % (x, y, w, h, rx, fill, stroke, width, d))


def arrow(x1, y1, x2, y2, stroke=MUTED):
    return ('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" stroke-width="2.5" '
            'marker-end="url(#a)"/>' % (x1, y1, x2, y2, stroke))


def svg(body, title):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
            'viewBox="0 0 %d %d"><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" '
            'fill="%s"/></marker></defs><rect width="%d" height="%d" fill="#ffffff"/>%s%s</svg>'
            % (W, H, W, H, MUTED, W, H, text(80, 96, title, 44, INK, '600'), body))


def fig_loop():
    steps = [('1  Capture', 'the aircraft takes a frame'),
             ('2  Judge', 'Tancho reads the frame against the goal'),
             ('3  Propose', 'Tancho names the next viewpoint'),
             ('4  Compile', 'deterministic path + limit checks'),
             ('5  Fly', 'PX4 flies the approved path'),
             ('6  Re-capture', 'the new frame arrives'),
             ('7  Judge again', 'goal met, or go round once more')]
    parts, x, y = [], 80, 240
    for i, (head, sub) in enumerate(steps):
        model = i in (1, 2)
        bx = x + (i % 4) * 420
        by = y + (i // 4) * 400
        parts.append(box(bx, by, 360, 230, '#eef2ff' if model else '#ffffff',
                         ACCENT if model else LINE, width=3 if model else 2))
        parts.append(text(bx + 28, by + 70, head, 34, INK, '600'))
        for j, line in enumerate(_wrap(sub, 22)):
            parts.append(text(bx + 28, by + 118 + j * 36, line, 26, MUTED))
        if i % 4 != 3 and i < len(steps) - 1:
            parts.append(arrow(bx + 372, by + 115, bx + 408, by + 115))
    parts.append(arrow(80 + 3 * 420 + 180, y + 240, 80 + 180, y + 390))
    parts.append(text(80, 1060, 'Blue steps are the model. Everything else is deterministic '
                                'and can be audited.', 28, MUTED))
    return svg(''.join(parts), 'One observation cycle')


def fig_limits():
    top, bottom = 430, 960
    parts = [box(620, top, 900, bottom - top, '#f8fafc', LINE, dash='8 8'),
             '<line x1="620" y1="%d" x2="1520" y2="%d" stroke="%s" stroke-width="4"/>'
             % (top, top, WARN),
             text(1540, top + 10, 'ceiling 12.0 m', 28, WARN, '600'),
             text(644, bottom - 36, 'approved volume, fixed before takeoff', 26, MUTED)]
    parts.append('<circle cx="1000" cy="330" r="14" fill="%s"/>' % WARN)
    parts.append(text(1030, 322, 'proposal: 14.39 m, pitch -44.8 deg', 28, WARN, '600'))
    parts.append(text(1030, 362, 'outside the volume, rejected before the aircraft moves',
                      26, MUTED))
    parts.append('<circle cx="1000" cy="660" r="14" fill="%s"/>' % OK)
    parts.append(text(1030, 652, 'accepted: inside the volume, pitch -40 deg', 28, OK, '600'))
    parts.append(text(1030, 692, 'dwell and pixels on target also satisfied', 26, MUTED))
    checks = ['approved volume', 'altitude ceiling', 'obstacle clearance',
              'dwell time', 'pixels on target', 'collision-free round trip']
    parts.append(text(80, 250, 'Every proposal is checked against', 30, INK, '600'))
    for i, c in enumerate(checks):
        parts.append(text(80, 330 + i * 60, '- ' + c, 28, MUTED))
    parts.append(text(80, 1090, 'The model proposes. It never authorises a flight.', 28, MUTED))
    return svg(''.join(parts), 'The compiler rejects a proposal that breaks a limit')


def fig_boundary():
    left = ['scene judgment: is the goal met', 'what is missing, and why',
            'which viewpoint to take next']
    right = ['object detection', 'area extraction', 'route compilation',
             'limit checking', 'flight control', 'capture verification']
    parts = [box(80, 200, 780, 760, '#eef2ff', ACCENT, width=3),
             box(940, 200, 780, 760, '#ffffff', LINE, width=2),
             text(120, 280, 'Tancho (Cosmos 3 Edge fine-tune)', 34, ACCENT, '600'),
             text(980, 280, 'Deterministic components', 34, INK, '600')]
    for i, s in enumerate(left):
        parts.append(text(120, 380 + i * 70, '- ' + s, 30, INK))
    for i, s in enumerate(right):
        parts.append(text(980, 380 + i * 70, '- ' + s, 30, INK))
    parts.append(text(120, 880, '2.44B parameters, 4.537 GiB in BF16', 26, MUTED))
    parts.append(text(980, 880, 'auditable, testable, no sampling', 26, MUTED))
    parts.append(text(80, 1070, 'Measured 2026-09-11: an off-the-shelf COCO detector scored '
                                'F1 0.588 on finding people from the air,', 28, MUTED))
    parts.append(text(80, 1112, 'against 0.024 to 0.065 for the parent model. Detection left '
                                'the language model that day.', 28, MUTED))
    return svg(''.join(parts), 'What the model decides, and what it must not')


def fig_stack():
    rows = [('Parent model', 'NVIDIA Cosmos 3 Edge, through NVIDIA cosmos-framework', ''),
            ('Fine-tuning and frozen evaluation',
             'Nebius AI Cloud Serverless Jobs, gpu-h100-sxm 1gpu-16vcpu-200gb',
             'image pinned by digest in Nebius Container Registry; parent weights on a Nebius compute filesystem'),
            ('Tasking to observation goal',
             'NVIDIA Nemotron 3.5 Lightning on Nebius Token Factory',
             'free text in, a validated goal out; the aircraft never sees free text'),
            ('Published weights', 'v13s/tancho, open weights, 487 files',
             'a token-free download matched every file'),
            ('On-board inference', 'Jetson-class board', 'measurement pending')]
    parts, y = [], 190
    for head, line, sub in rows:
        h = 180 if sub else 130
        parts.append(box(80, y, 1640, h))
        parts.append(text(110, y + 58, head, 30, MUTED, '600'))
        parts.append(text(110, y + 108, line, 32, INK))
        if sub:
            parts.append(text(110, y + 152, sub, 25, MUTED))
        y += h + 20
    return svg(''.join(parts), 'Where NVIDIA and Nebius are used')


def fig_tasking():
    rows = json.loads((ROOT / 'results/summary.json').read_text())['models']
    parts = [text(80, 170, '16 cases, frozen before any model was called. '
                           'A square is one case: filled means the profile, the category and '
                           'every required question were exact.', 26, MUTED)]
    y = 280
    for model, m in sorted(rows.items()):
        detail = json.loads((ROOT / 'results' / (model.replace('/', '_') + '.json')).read_text())
        parts.append(text(80, y, model.split('/')[-1], 32, INK, '600'))
        parts.append(text(80, y + 44, 'median %.2f s' % m['median_seconds'], 26, MUTED))
        for i, row in enumerate(detail['rows']):
            passed = (row.get('schema_valid') and row.get('profile_exact')
                      and row.get('category_exact') and row.get('question_f1') == 1.0)
            x = 600 + i * 60
            parts.append(box(x, y - 34, 44, 44, ACCENT if passed else '#ffffff',
                             ACCENT if passed else LINE, rx=6, width=2))
        parts.append(text(1620, y + 4, '%d / 16' % round(m['all_three_exact'] * 16), 34,
                          INK, '600'))
        y += 170
    parts.append(text(80, y + 90, 'With 16 cases, 15/16 and 14/16 are not meaningfully '
                                  'different. What the measurement supports is that the '
                                  '30B-A3B class is', 28, MUTED))
    parts.append(text(80, y + 132, 'enough for this step, and that the 120B model buys nothing '
                                  'here.', 28, MUTED))
    return svg(''.join(parts), 'Choosing the Nemotron variant for the tasking step')


def _wrap(s, n):
    words, lines, cur = s.split(), [], ''
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = (cur + ' ' + w).strip()
    return lines + ([cur] if cur else [])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figures = {'01-loop': fig_loop, '02-limits': fig_limits, '03-boundary': fig_boundary,
               '04-stack': fig_stack, '05-tasking': fig_tasking}
    for name, builder in figures.items():
        source = OUT / (name + '.svg')
        source.write_text(builder())
        subprocess.run(['rsvg-convert', '-w', str(W), '-h', str(H), '-o',
                        str(OUT / (name + '.png')), str(source)], check=True)
        print(name, (OUT / (name + '.png')).stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
