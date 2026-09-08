"""Lay the partial-product piece `serial pp <i>` on oracle's grid by construction, and verify it on
the host model (dotsim's step semantics) before the card is seeded with it.

usage: python ppplan.py <i>.. [--rows=32|64] [--fresh=N] [--columns=N] [--beam=N] [--chains=N] [--reach=N] [--out=DIR]
       python ppplan.py all [--out=DIR]
       python ppplan.py --verify <grid> <i> <rows> <columns> <lane>

The exam: x on row 0 and y on row 1, a bit a step least significant first over W steps and W of
hold, state rows 2..lane-1, the lane row (2 + state) answering pp_i(k) = x(k-i) AND y_i at step k,
rows past the lane fresh (nought at column 0), the bottom row a one. The circuit:

  q         a state row carrying the one from step 1 on: `not` at column 1 gives pulse_0 (one
            only at step 0), and a copy of the one at the last column keeps q;
  P_1..P_i  the one-hot chain, P_j = P_{j-1} carried (P_0 = pulse_0), each a copy at the last
            column, so P_j(k) = [k == j] sits on its row all step long;
  xd_1..xd_i  x delayed j steps, a copy at the last column of the row before (xd_0 = x);
  A         y AND P_i = maj(y, P_i, zero), the zero a row no gate ever writes;
  L         the latch, L' = L XOR A as sum(A, zero, L) on L's own row, so L holds y_i from
            step i+1 on and A is nought at every other step;
  pp        maj(xd_i, L', zero) on the lane, reading the latch as it is written at step i.

Every operand is brought where a gate can read it by dotplan's shortest paths over the time grid
(copies up, down, hop, leap), the chains landing as late as they can so their values stay
readable, and every placement of A, L and pp is scored from one reach table a name and the
cheapest few committed exactly. The kept plan is run on the host model over corners and random
operand pairs for W = 4, 8 and 16 before it is written.

This is the file that laid pp00.grid..pp15.grid, which sit beside it. Nothing here searches: every
gate is placed by a shortest path and the beam is over placements, not over populations. To lay them
again, write into an empty directory and compare, never into this one:

    python ppplan.py all --out=/some/empty/dir

It refuses to write here, because the published grids and exam files are what a reader compares
against. Fourteen of the sixteen come out byte for byte the same; pp08 and pp10 do not, because the
planner was edited after those two were written, and both re-laid grids still verify on all 65,536
confirm rows. The exam text names the files by the paths they had in the repository they were made
in (expeditions/mul/pp<NN>.grid, board/research/ternary/ppplan.py), left exactly as it was so the
reproduction is byte-exact. Those two names are this directory's pp<NN>.grid and this file.
"""
import itertools
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dotplan
import dotsim
from dotplan import Plan, SUM, MAJ, NOT

HERE = os.path.dirname(os.path.abspath(__file__))
FAN = 2


class Geometry:
    def __init__(self, rows, columns, fresh):
        self.rows = rows
        self.columns = columns
        self.last = columns - 1
        self.const = rows - 1
        self.lane = rows - 2 - fresh
        self.state = self.lane - FAN
        self.fresh = fresh
        self.state_rows = frozenset(range(FAN, self.lane))
        self.fresh_rows = frozenset(range(self.lane + 1, self.const))

    def apply(self):
        dotplan.ROWS = self.rows
        dotplan.COLUMNS = self.columns
        dotplan.LAST = self.last
        dotplan.FAN = FAN
        dotplan.STATE = self.state
        dotplan.LANE = self.lane
        dotplan.CONST = self.const
        dotplan.STATE_ROWS = self.state_rows

    def exam_fields(self, wide):
        return {'inputs': 2 * wide, 'outputs': 2 * wide, 'rows': self.rows, 'columns': self.columns,
                'steps': wide, 'state': self.state, 'hold': wide}


def readers(geo, row):
    n = geo.rows
    return [(row + 1) % n, (row - 1) % n, (row - n // 4) % n, (row - n // 2) % n]


def distances(geo, source, allowed):
    """Copies from `source` to every allowed row over the reader graph."""
    seen = {source: 0}
    frontier = [source]
    while frontier:
        nxt = []
        for row in frontier:
            for reader in readers(geo, row):
                if reader in allowed and reader not in seen:
                    seen[reader] = seen[row] + 1
                    nxt.append(reader)
        frontier = nxt
    return seen


def untouched(plan, row):
    return (row not in plan.carries and row not in plan.reads and row not in plan.protect
            and all((c, row) not in plan.gates for c in range(1, dotplan.LAST + 1)))


def make_zero(plan, geo, row):
    """Reserve `row` as a nought forever: an untouched state row (never written, carried nought) or an
    untouched fresh row (nought at column 0, guarded). None when it cannot be."""
    if row == geo.const or row < FAN or not untouched(plan, row):
        return None
    if row in geo.state_rows:
        plan.carries[row] = 'zero'
    else:
        plan.guard(row, 1, geo.last, 'zero')
    return row


def route_late(plan, name, target, deadline):
    """Lay the cheapest copies that put `name` on `target` by `deadline`, landing as late as they can."""
    table = plan.reach(name, deadline)
    options = [(gates, -col, path) for (row, col), (gates, path) in table.items() if row == target and col <= deadline]
    if not options:
        return None
    gates, col, path = min(options)
    col = -col
    for c, r, kind in path:
        if kind == NOT:
            plan.place_not(c, r)
        else:
            plan.place_copy(c, r, kind)
    if plan.hold(col, target) != name:
        return None
    plan.guard(target, col, deadline, name)
    return col


def link(plan, name, target, deadline):
    """A chain link: the state row `target` ends the step holding `name`."""
    held = plan.carries.pop(target)
    at = route_late(plan, name, target, deadline)
    plan.carries[target] = held
    return at


def lay_chains(geo, i, q, d_p, xd1, d_x):
    """The chains on a fresh plan: the q row, P_1..P_i from q in direction d_p, xd_1..xd_i from xd1 in
    direction d_x; the plan and the names of the last pulse and the last delay, or None."""
    p_rows = [q + d_p * j for j in range(1, i + 1)]
    x_rows = [xd1 + d_x * (j - 1) for j in range(1, i + 1)]
    used = [q] + p_rows + x_rows
    if len(set(used)) != len(used) or any(r not in geo.state_rows for r in used):
        return None
    plan = Plan()
    plan.guard(0, 1, geo.last, 'x0')
    plan.guard(1, 1, geo.last, 'x1')
    plan.guard(geo.const, 1, geo.last, 'one')
    plan.guard(geo.lane, 1, geo.last, 'zero')
    plan.carries[q] = 'q'
    for j, r in enumerate(p_rows, 1):
        plan.carries[r] = f'P{j}'
    for j, r in enumerate(x_rows, 1):
        plan.carries[r] = f'xd{j}'
    plan.place_not(1, q)
    for j in range(i, 0, -1):
        if link(plan, f'P{j - 1}' if j > 1 else '~q', p_rows[j - 1], geo.last) is None:
            return None
    if link(plan, 'one', q, geo.last) is None:
        return None
    for j in range(i, 0, -1):
        if link(plan, f'xd{j - 1}' if j > 1 else 'x0', x_rows[j - 1], geo.last) is None:
            return None
    pulse = f'P{i}' if i else '~q'
    delayed = f'xd{i}' if i else 'x0'
    return plan, pulse, delayed


def three_rows(geo, middle):
    return ((middle - 1) % geo.rows, middle, (middle + 1) % geo.rows)


def lane_zero(plan, geo):
    """Whether the lane still reads as nought: no gate on it, and nothing but nought guarded there."""
    lane = geo.lane
    return (all((c, lane) not in plan.gates for c in range(1, geo.last + 1))
            and all(want == 'zero' for _, want in plan.protect.get(lane, ())))


def zero_ok(plan, geo, row, middle):
    """Whether `row` can serve as the nought of a gate on `middle`: untouched, and the middle itself only when fresh."""
    if row == geo.lane:
        return lane_zero(plan, geo)
    if row == middle:
        return row in geo.fresh_rows and untouched(plan, row)
    return row != geo.const and row >= FAN and untouched(plan, row)


def score_gate(plan, geo, tables, middle, names, deadline):
    """The cheapest arrival of the operand names on the three rows around `middle` from reach tables;
    (gates, column) or None. `names` maps row -> name, 'zero' rows checked, not routed."""
    gates, column = 0, 0
    for row, name in names.items():
        if name == 'zero':
            if not zero_ok(plan, geo, row, middle):
                return None
            continue
        if row in plan.carries:
            if plan.carries[row] == name:
                continue
            return None
        table = tables[name]
        best = None
        for (r, col), (g, _) in table.items():
            if r == row and col <= deadline and (best is None or (g, col) < best):
                best = (g, col)
        if best is None:
            return None
        gates += best[0]
        column = max(column, best[1])
    return gates, column


def commit_gate(plan, geo, middle, names, kind, out, deadline):
    """Route the operands to the rows around `middle` and add the gate; the plan's column of the gate or None."""
    arrivals = []
    for row, name in names.items():
        if name == 'zero':
            if make_zero(plan, geo, row) is None and not (row == geo.lane and zero_ok(plan, geo, row, middle)):
                return None
            continue
        if row in plan.carries:
            if plan.carries[row] != name:
                return None
            continue
        at = plan.route(name, row, deadline)
        if at is None:
            return None
        arrivals.append(at)
    k = max(arrivals) + 1 if arrivals else 1
    for row in names:
        plan.trim(row, k - 1)
    up, down = (middle - 1) % geo.rows, (middle + 1) % geo.rows
    reads = [(k - 1, up, plan.hold(k - 1, up)), (k - 1, middle, plan.hold(k - 1, middle)), (k - 1, down, plan.hold(k - 1, down))]
    for (_, row, held) in reads:
        if held != names[row]:
            return None
    if (k, middle) in plan.gates or any(c >= k for c, _ in plan.reads.get(middle, ())):
        return None
    if middle not in plan.carries and not plan.fits(k, middle, out):
        return None
    plan.add(k, middle, kind, out, reads)
    return k


def candidates_and(plan, geo, pulse, deadline):
    tables = {'x1': plan.reach('x1', deadline), pulse: plan.reach(pulse, deadline)}
    found = []
    for middle in range(FAN, geo.const):
        if middle == geo.lane:
            continue
        if any((c, middle) in plan.gates for c in range(1, geo.last + 1)):
            continue
        up, down = (middle - 1) % geo.rows, (middle + 1) % geo.rows
        if middle in plan.carries:
            continue
        for order in itertools.permutations(('x1', pulse, 'zero')):
            names = {up: order[0], middle: order[1], down: order[2]}
            score = score_gate(plan, geo, tables, middle, names, deadline)
            if score is not None:
                found.append((score, middle, names))
    found.sort(key=lambda it: (it[0], it[1]))
    return found


def candidates_latch(plan, geo, deadline):
    table = {'A': plan.reach('A', deadline)}
    found = []
    for L in geo.state_rows:
        if not untouched(plan, L):
            continue
        up, down = L - 1, L + 1
        for a_row, z_row in ((up, down), (down, up)):
            names = {a_row: 'A', z_row: 'zero', L: 'L'}
            if not zero_ok(plan, geo, z_row, L) or a_row in plan.carries or a_row < FAN or a_row == geo.const:
                continue
            score = score_gate(plan, geo, table, L, {a_row: 'A', z_row: 'zero'}, deadline)
            if score is not None:
                found.append((score, L, names))
    found.sort(key=lambda it: (it[0], it[1]))
    return found


def commit_latch(plan, geo, L, names, deadline):
    plan.carries[L] = 'L'
    routed = {row: name for row, name in names.items() if row != L}
    k = commit_gate(plan, geo, L, {**routed, L: 'L'}, SUM, 'L+', deadline)
    return k


def candidates_pp(plan, geo, delayed, deadline):
    tables = {delayed: plan.reach(delayed, deadline), 'L+': plan.reach('L+', deadline)}
    found = []
    lane = geo.lane
    up, down = lane - 1, lane + 1
    if down == geo.const:
        return found
    for order in itertools.permutations((delayed, 'L+', 'zero')):
        names = {up: order[0], lane: order[1], down: order[2]}
        if names[up] == 'zero' and up in plan.carries and plan.carries[up] != 'zero':
            continue
        if names[up] != 'zero' and up in plan.carries:
            continue
        score = score_gate(plan, geo, tables, lane, names, deadline)
        if score is not None:
            found.append((score, lane, names))
    found.sort(key=lambda it: (it[0], it[1]))
    return found


def commit_pp(plan, geo, names, deadline):
    lane = geo.lane
    plan.protect.pop(lane, None)
    for row, name in list(names.items()):
        if name == 'zero' and row == lane:
            plan.guard(lane, 1, deadline, 'zero')
    return commit_gate(plan, geo, lane, names, MAJ, 'pp', deadline)


def lay(geo, i, beam=4, chains=60, reach=2, log=None):
    """The cheapest plan of pp_i on this geometry, verified, or None."""
    geo.apply()
    allowed = set(geo.state_rows)
    from_const = distances(geo, geo.const, allowed | geo.fresh_rows)
    from_x = distances(geo, 0, allowed | geo.fresh_rows)
    q_rows = sorted(r for r in geo.state_rows if from_const.get(r, 99) <= reach)
    x_rows = sorted(r for r in geo.state_rows if from_x.get(r, 99) <= reach)
    frames = []
    for q in q_rows:
        for d_p in (1, -1):
            for xd1 in x_rows:
                for d_x in (1, -1):
                    if i == 0 and (d_p == -1 or d_x == -1):
                        continue
                    laid = lay_chains(geo, i, q, d_p, xd1, d_x)
                    if laid is not None:
                        frames.append((laid[0].cost, len(frames), laid))
    frames.sort(key=lambda it: (it[0], it[1]))
    frames = frames[:chains]
    if log:
        log(f'  pp{i} rows {geo.rows} fresh {geo.fresh} columns {geo.columns}: {len(frames)} chain frames')
    best = None
    for cost, _, (plan0, pulse, delayed) in frames:
        deadline_a = geo.last - 5
        for score, middle, names in candidates_and(plan0, geo, pulse, deadline_a)[:beam]:
            plan1 = plan0.clone()
            if commit_gate(plan1, geo, middle, names, MAJ, 'A', deadline_a) is None:
                continue
            for score2, L, names2 in candidates_latch(plan1, geo, geo.last - 3)[:beam]:
                plan2 = plan1.clone()
                if commit_latch(plan2, geo, L, names2, geo.last - 3) is None:
                    continue
                for score3, lane, names3 in candidates_pp(plan2, geo, delayed, geo.last - 1)[:beam]:
                    plan3 = plan2.clone()
                    if commit_pp(plan3, geo, names3, geo.last - 1) is None:
                        continue
                    used = plan3.columns_used()
                    key = (plan3.cost, used)
                    if best is None or key < best[0]:
                        grid = plan3.grid()
                        whole, first = check(grid, i, geo, 8, trials=64)
                        if whole == 64:
                            best = (key, plan3)
    return best


def cases_of(wide, trials, seed=1):
    top = (1 << wide) - 1
    corners = [0, 1, 2, top, top - 1, 1 << (wide - 1), (1 << (wide - 1)) - 1, 0x5555 & top, 0xAAAA & top]
    cases = [(a, b) for a in corners for b in corners]
    rng = random.Random(seed)
    while len(cases) < trials:
        cases.append((rng.randrange(1 << wide), rng.randrange(1 << wide)))
    return cases[:trials] if trials < len(cases) else cases


def want_of(x, y, i, wide):
    return ((x * ((y >> i) & 1)) << i) & ((1 << (2 * wide)) - 1)


def check(grid, i, geo, wide, trials=500, seed=1):
    """How many of the trials come out whole on the lane over 2W steps, and the first failure."""
    whole, first = 0, None
    for x, y in cases_of(wide, trials, seed):
        words = [[(x >> t) & 1, (y >> t) & 1] for t in range(wide)] + [[0, 0] for _ in range(wide)]
        lasts = dotsim.run(grid, words, geo.rows, FAN, geo.state, geo.columns)
        got = sum(lasts[t][geo.lane] << t for t in range(2 * wide))
        want = want_of(x, y, i, wide)
        if got == want:
            whole += 1
        elif first is None:
            first = (x, y, want, got)
    return whole, first


def verify(grid, i, geo, log=print):
    for wide, trials in ((4, 256), (8, 2000), (16, 2000)):
        whole, first = check(grid, i, geo, wide, trials, seed=wide)
        log(f'  pp{i} W={wide}: {whole} of {trials} whole' + (f' first failure {first}' if first else ''))
        if whole != trials:
            return False
    return True


def exam_text(i, geo, wide, gates, fed):
    f = geo.exam_fields(wide)
    lines = [
        f'# The partial product x(k-{i}) AND y_{i} of a {wide}-bit serial multiplier, laid by construction',
        f'# in board/research/ternary/ppplan.py: x on row 0 and y on row 1 a bit a step, least significant',
        f'# first, a one-hot step chain and an x delay chain of {i} state rows each, a latch of y_{i} set',
        f'# by the pulse of step {i}, and the product on the lane row {geo.lane} at step k, {wide} steps of word',
        f'# and {wide} of hold. {gates} gates, whole on the host model over corners and random operand pairs.',
    ]
    if fed:
        lines = [
            f'# The partial product x(k-{i}) AND y_{i} of a {wide}-bit serial multiplier: the planned grid',
            f'# expeditions/mul/pp{i:02d}.grid ({gates} gates, board/research/ternary/ppplan.py) seeded into',
            f'# the computed table, which the card keeps whole under `lean wires` over the sampled rows and',
            f'# every one of the confirm rows: the piece board/research/ternary/mulcompose.py composes into',
            f'# board/mul16.v.',
        ]
    lines += [
        f'inputs {f["inputs"]}',
        f'outputs {f["outputs"]}',
        f'rows {f["rows"]}',
        f'columns {f["columns"]}',
        f'steps {f["steps"]}',
        f'state {f["state"]}',
        f'hold {f["hold"]}',
        'fold 1 below',
        'gates sum maj not up down leap hop',
        'colony 4096',
        'renew 32',
        'sample 256',
        'confirm 65536',
        'packed',
        'lean wires',
        'seed 4343',
        f'serial pp {i}',
    ]
    if fed:
        lines += [f'start expeditions/mul/pp{i:02d}.grid 4096', f'keep expeditions/mul/pp{i:02d}fed.grid']
    return '\n'.join(lines) + '\n'


def plan_piece(i, rows_options=(32, 64), fresh_options=(1, 2), columns_options=(8, 10, 12, 14, 16), beam=4, chains=60, reach=2, log=print):
    started = time.time()
    for rows in rows_options:
        for columns in columns_options:
            for fresh in fresh_options:
                geo = Geometry(rows, columns, fresh)
                if 2 * i + 2 > geo.state:
                    continue
                best = lay(geo, i, beam, chains, reach, log)
                if best is None:
                    continue
                (cost, used), plan = best
                geo2 = Geometry(rows, used + 1, fresh)
                geo2.apply()
                grid = plan.grid()
                log(f'  pp{i}: {cost} gates, {used + 1} columns, rows {rows}, fresh {fresh}, lane {geo2.lane} ({time.time() - started:.0f}s)')
                if verify(grid, i, geo2, log):
                    return geo2, grid, plan
                log(f'  pp{i}: the plan on {rows} rows {columns} columns fresh {fresh} failed verification')
    return None


def write_piece(i, geo, grid, plan, out_dir, wide=16, log=print):
    os.makedirs(out_dir, exist_ok=True)
    gates = len(grid)
    grid_path = os.path.join(out_dir, f'pp{i:02d}.grid')
    open(grid_path, 'w', encoding='utf-8').write(dotsim.as_grid_file(grid, geo.rows, geo.columns) + '\n')
    open(os.path.join(out_dir, f'pp{i:02d}.txt'), 'w', encoding='utf-8').write(exam_text(i, geo, wide, gates, False))
    open(os.path.join(out_dir, f'pp{i:02d}fed.txt'), 'w', encoding='utf-8').write(exam_text(i, geo, wide, gates, True))
    json.dump({'i': i, 'rows': geo.rows, 'columns': geo.columns, 'lane': geo.lane, 'state': geo.state, 'gates': gates,
               'placed': [[c, r, k, n] for c, r, k, n in sorted(plan.log)]},
              open(os.path.join(out_dir, f'pp{i:02d}.json'), 'w', encoding='utf-8'), indent=1)
    log(f'  wrote {grid_path}: {gates} gates')


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    if args[0] == '--verify':
        grid_path, i, rows, columns, lane = args[1], int(args[2]), int(args[3]), int(args[4]), int(args[5])
        geo = Geometry(rows, columns, rows - 2 - lane)
        geo.apply()
        grid = dotsim.read_grid(grid_path, rows, columns)
        ok = verify(grid, i, geo)
        raise SystemExit(0 if ok else 1)
    out_dir = None
    rows_options = (32, 64)
    fresh_options = (1, 2)
    columns_options = (8, 10, 12, 14, 16)
    which = []
    beam, chains, reach = 4, 60, 2
    for a in args:
        if a.startswith('--out='):
            out_dir = a[6:]
        elif a.startswith('--rows='):
            rows_options = (int(a[7:]),)
        elif a.startswith('--fresh='):
            fresh_options = (int(a[8:]),)
        elif a.startswith('--columns='):
            columns_options = (int(a[10:]),)
        elif a.startswith('--beam='):
            beam = int(a[7:])
        elif a.startswith('--chains='):
            chains = int(a[9:])
        elif a.startswith('--reach='):
            reach = int(a[8:])
        elif a == 'all':
            which = list(range(16))
        else:
            which.append(int(a))
    if which and out_dir is None:
        raise SystemExit(
            'ppplan.py refuses to write into this directory: it holds the published grids and exam\n'
            'files, and laying them again on top would leave nothing to compare against. Give a\n'
            'destination:\n'
            '    python ppplan.py all --out=/some/empty/dir\n'
            'then diff that directory against this one.')
    if os.path.abspath(out_dir or '') == os.path.abspath(HERE):
        raise SystemExit('--out is this directory; choose an empty one so the published files survive')
    for i in which:
        found = plan_piece(i, rows_options, fresh_options, columns_options, beam, chains, reach)
        if found is None:
            raise SystemExit(f'pp{i}: no plan')
        geo, grid, plan = found
        write_piece(i, geo, grid, plan, out_dir)


if __name__ == '__main__':
    main()
