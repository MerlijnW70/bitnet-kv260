"""Verify the kept partial-product grids in this directory against their exams' confirm rows.

A piece's case is 32 bits, so its full table is 2^32 rows; an exam asks for 65,536 of them, drawn
from across the whole table, and every answer bit of every one of those rows must be right. That
is what the search kept a grid on and what this reproduces: a strong check, not a proof.

usage: python3 verify_pieces.py                  every kept grid pp00fed..pp15fed
       python3 verify_pieces.py --planned        the planner's grids pp00..pp15 instead
       python3 verify_pieces.py GRID EXAM        one grid under one exam file
       python3 verify_pieces.py --cases N ...    fewer or more confirm rows than the exam asks

Standalone: the Python standard library and nothing else, no import from the repository the grids
were made in. A grid is whitespace-separated gate kinds, column by column, rows top to bottom,
columns 1..columns-1 (column 0 is fed, never written). The exam file states rows, columns, steps,
state, hold, the confirm count, the seed and the filler.

The machine: rows 0 and 1 are the input rows, x and y, a bit a step least significant first; rows
2..state+1 are state rows, whose value in the last column is carried into the next step's column 0;
row state+2 is the lane, read at the last column of every step; rows above the bottom one that are
neither state nor lane are fresh (nought at column 0 every step); the bottom row is a one. Within a
step the column advances left to right: a cell holding gate kind k reads the previous column and
writes this one, and a cell holding 0 copies the previous column straight through.

Kinds: 0 wire, 3 xor(up, down), 4 not(straight), 5 maj(up, down, straight), 7 sum(up ^ down ^
straight), 10 up (row - 1), 11 down (row + 1), 13 leap (row + rows/2), 14 hop (row + rows/4), every
neighbour taken modulo rows.

The exam's `serial pp i` filler asks for the partial product: at step k the lane must show bit k of
(x * y_i) << i, over steps + hold steps. The confirm rows are the exam's own: case at, for at in
0..confirm-1, is splitmix64(seed + (at + 1) * 0x9E3779B97F4A7C15) masked to `inputs` bits, and a
case holds x bit t at case bit 2t and y bit t at case bit 2t+1. Every case is run at once, one
machine-word column of the grid per bit position, so 65,536 cases cost one pass.

exit 0 every piece whole on every case, 1 a case came out wrong, 2 a file is missing or malformed.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WIRE, XOR, NOT, MAJ, SUM, UP, DOWN, LEAP, HOP = 0, 3, 4, 5, 7, 10, 11, 13, 14
KNOWN = (WIRE, XOR, NOT, MAJ, SUM, UP, DOWN, LEAP, HOP)
GOLDEN = 0x9E37_79B9_7F4A_7C15


class Malformed(Exception):
    pass


def read_exam(path):
    """The exam's fields: the numbers by name, and 'pp' for the `serial pp i` filler."""
    fields = {}
    try:
        text = open(path, encoding='utf-8').read()
    except OSError as exc:
        raise Malformed(f'{path}: {exc}')
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if parts[0] == 'serial' and len(parts) == 3 and parts[1] == 'pp':
            fields['pp'] = int(parts[2])
        elif len(parts) >= 2 and parts[1].lstrip('-').isdigit():
            fields[parts[0]] = int(parts[1])
    for name in ('inputs', 'outputs', 'rows', 'columns', 'steps', 'state', 'hold', 'confirm', 'seed', 'pp'):
        if name not in fields:
            raise Malformed(f'{path}: no `{name}` line')
    return fields


def read_grid(path, rows, columns):
    """{(column, row): kind} for the nonzero cells, from the flat column-by-column file."""
    try:
        kinds = open(path, encoding='utf-8').read().split()
    except OSError as exc:
        raise Malformed(f'{path}: {exc}')
    want = (columns - 1) * rows
    if len(kinds) != want:
        raise Malformed(f'{path}: {len(kinds)} cells, a {rows} by {columns} grid holds {want}')
    grid = {}
    for at in range(1, columns):
        for row in range(rows):
            kind = int(kinds[(at - 1) * rows + row])
            if kind not in KNOWN:
                raise Malformed(f'{path}: column {at} row {row} holds kind {kind}, which is not a gate')
            if kind:
                grid[(at, row)] = kind
    return grid


def drawn_case(seed, at):
    """splitmix64 of the exam's seed and the row number: the confirm rows the exam walks."""
    mask = (1 << 64) - 1
    z = (seed + (at + 1) * GOLDEN) & mask
    z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & mask
    return z ^ (z >> 31)


def cases_of(seed, count, bits):
    mask = (1 << bits) - 1
    return [drawn_case(seed, at) & mask for at in range(count)]


def columns_of_cases(cases, bits):
    """One integer a case bit, bit j of it holding case j's value: the cases transposed."""
    span = (len(cases) + 7) // 8
    slabs = [bytearray(span) for _ in range(bits)]
    for slot, case in enumerate(cases):
        byte, bit = slot >> 3, 1 << (slot & 7)
        while case:
            low = case & -case
            slabs[low.bit_length() - 1][byte] |= bit
            case ^= low
    return [int.from_bytes(bytes(slab), 'little') for slab in slabs]


def plan_of(grid, rows, columns):
    """The gates a column at a time, each with the rows it reads, so the run touches nothing else."""
    plan = []
    for at in range(1, columns):
        here = []
        for row in range(rows):
            kind = grid.get((at, row))
            if kind:
                here.append((row, kind, (row - 1) % rows, (row + 1) % rows,
                             (row + rows // 2) % rows, (row + rows // 4) % rows))
        plan.append(here)
    return plan


def run(plan, rows, state, lane, feeds, ones):
    """The lane at the last column of every step, one integer a step over every case at once."""
    carried = [0] * rows
    lanes = []
    for bits in feeds:
        column = [0] * rows
        column[0] = bits[0]
        column[1] = bits[1]
        for row in range(2, 2 + state):
            column[row] = carried[row]
        column[rows - 1] = ones
        for here in plan:
            prev = column
            column = list(prev)
            for row, kind, up, down, leap, hop in here:
                if kind == SUM:
                    column[row] = prev[up] ^ prev[down] ^ prev[row]
                elif kind == MAJ:
                    straight = prev[row]
                    column[row] = ((prev[up] ^ straight) & (prev[down] ^ straight)) ^ straight
                elif kind == UP:
                    column[row] = prev[up]
                elif kind == DOWN:
                    column[row] = prev[down]
                elif kind == HOP:
                    column[row] = prev[hop]
                elif kind == LEAP:
                    column[row] = prev[leap]
                elif kind == NOT:
                    column[row] = prev[row] ^ ones
                else:
                    column[row] = prev[up] ^ prev[down]
        lanes.append(column[lane])
        carried = column
    return lanes


def wanted(xs, ys, delay, steps, total):
    """Bit k of (x * y_i) << i a step, one integer a step over every case at once."""
    y = ys[delay]
    return [xs[k - delay] & y if 0 <= k - delay < steps else 0 for k in range(total)]


def failure(got, want, cases, delay, steps, total):
    """The first case that came out wrong: (slot, case, x, y, got, want), or None."""
    slot = None
    for k in range(total):
        bad = got[k] ^ want[k]
        if bad:
            at = (bad & -bad).bit_length() - 1
            slot = at if slot is None else min(slot, at)
    if slot is None:
        return None
    case = cases[slot]
    x = sum(((case >> (2 * t)) & 1) << t for t in range(steps))
    y = sum(((case >> (2 * t + 1)) & 1) << t for t in range(steps))
    shown = sum(((got[k] >> slot) & 1) << k for k in range(total))
    asked = sum(((want[k] >> slot) & 1) << k for k in range(total))
    return slot, case, x, y, shown, asked


def verify(grid_path, exam_path, count=None, log=print):
    """One piece over the exam's confirm rows: True when every case is whole."""
    exam = read_exam(exam_path)
    rows, columns, steps, state, hold = exam['rows'], exam['columns'], exam['steps'], exam['state'], exam['hold']
    delay, bits = exam['pp'], exam['inputs']
    lane = 2 + state
    total = min(exam['outputs'], steps + hold)
    if lane >= rows - 1:
        raise Malformed(f'{exam_path}: a lane on row {lane} does not fit {rows} rows')
    if delay >= steps:
        raise Malformed(f'{exam_path}: `serial pp {delay}` reads a bit of y that {steps} steps do not carry')
    if bits != 2 * steps:
        raise Malformed(f'{exam_path}: `inputs {bits}` does not feed {steps} steps of two rows')
    walked = min(exam['confirm'], 1 << bits) if count is None else count
    grid = read_grid(grid_path, rows, columns)
    started = time.time()
    cases = cases_of(exam['seed'], walked, bits)
    ones = (1 << walked) - 1
    columns_of = columns_of_cases(cases, bits)
    xs = [columns_of[2 * t] for t in range(steps)]
    ys = [columns_of[2 * t + 1] for t in range(steps)]
    feeds = [(xs[t], ys[t]) for t in range(steps)] + [(0, 0)] * hold
    got = run(plan_of(grid, rows, columns), rows, state, lane, feeds, ones)
    want = wanted(xs, ys, delay, steps, total)
    first = failure(got[:total], want, cases, delay, steps, total)
    name = os.path.basename(grid_path)
    took = time.time() - started
    if first is None:
        log(f'{name}: {len(grid)} gates, {rows} rows, {columns} columns, lane {lane}, '
            f'serial pp {delay}: {walked} of {walked} confirm rows whole ({took:.1f}s)')
        return True
    slot, case, x, y, shown, asked = first
    log(f'{name}: {len(grid)} gates, {rows} rows, {columns} columns, lane {lane}, '
        f'serial pp {delay}: WRONG on confirm row {slot}, case 0x{case:08x}: '
        f'x={x} y={y} y_{delay}={(y >> delay) & 1} lane 0x{shown:08x} wants 0x{asked:08x} ({took:.1f}s)')
    return False


def pieces(planned):
    tail = '.grid' if planned else 'fed.grid'
    found = []
    for i in range(16):
        grid = os.path.join(HERE, f'pp{i:02d}{tail}')
        exam = os.path.join(HERE, f'pp{i:02d}' + ('.txt' if planned else 'fed.txt'))
        if not os.path.exists(grid) or not os.path.exists(exam):
            raise Malformed(f'{grid} or {exam}: not there')
        found.append((grid, exam))
    return found


def main():
    args = sys.argv[1:]
    planned = '--planned' in args
    count = None
    rest = []
    at = 0
    while at < len(args):
        if args[at] == '--planned':
            at += 1
            continue
        if args[at] == '--cases':
            count = int(args[at + 1])
            if count < 1:
                raise SystemExit('--cases must be at least 1: checking nothing is not a pass')
            at += 2
            continue
        rest.append(args[at])
        at += 1
    if len(rest) == 1:
        raise SystemExit(__doc__)
    pairs = [(rest[0], rest[1])] if rest else pieces(planned)
    started = time.time()
    whole = 0
    for grid, exam in pairs:
        if verify(grid, exam, count):
            whole += 1
    took = time.time() - started
    print(f'{whole} of {len(pairs)} pieces whole on every confirm row ({took:.1f}s)')
    return 0 if whole == len(pairs) else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Malformed as why:
        print(why, file=sys.stderr)
        sys.exit(2)
