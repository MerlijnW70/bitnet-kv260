"""Lay out a bit-serial ternary dot product on oracle's grid, one adder cell an operand, and
verify it on the host model before the card is seeded with it.

usage: python dotplan.py [out.grid] [out.json] [--beam=N] [--orders=N]
       python dotplan.py --verify <grid>

The exam: `dot 8 1,-1,1,1`, rows 32, columns 12, the four activations on rows 0..3 a bit a step
(least significant first), state rows 4..29, the answer bit read off row 30 at the last column,
row 31 a one. The circuit is a chain of serial adders: a running value R, then R + x, or R - x for
the weight of minus one. Every cell is two three-row places, (P, carry, Q) in either order, one
for the `sum` gate (s = P ^ carry ^ Q) and one for the `maj` gate (carry' = maj(P, carry, Q)); a
place holds a copy of the carry (K, read off the state row C at column 0 by a row that reads C:
C-8 by hop or C-16 by leap) so that the two operands sit either side of it, and the maj place is
either the state row itself or C-16, which reads C by leap and is read back by leap. A subtractor
R - x is the same cell with the maj fed ~R, borrow for carry (d = R ^ x ^ w, w' = maj(~R, x, w)).
Operands travel by copies (up, down, hop, leap) found by a shortest-path search over the time
grid, a gate a copy, and `not` is one more gate on the way. The planner scores every cell
placement by a one-source search per operand and then commits the cheapest few exactly, a beam of
plans a cell, and verifies the whole grid on dotsim over random activations.

This is the layout planner ppplan.py builds on, and nothing but a planner: it decides where a gate
goes by shortest paths, never by search over populations.
"""
import heapq
import itertools
import json
import sys
import time

import dotsim
from dotsim import ROWS, COLUMNS, FAN, STATE, UP, DOWN, LEAP, HOP, SUM, MAJ, NOT

LAST = COLUMNS - 1
LANE = FAN + STATE
CONST = ROWS - 1
STATE_ROWS = frozenset(range(FAN, FAN + STATE))
WEIGHTS = (1, -1, 1, 1)
WIDE, HOLD = 8, 3
KIND_NAME = {UP: 'up', DOWN: 'down', LEAP: 'leap', HOP: 'hop', SUM: 'sum', MAJ: 'maj', NOT: 'not'}
COPIES = (UP, DOWN, HOP, LEAP)


def width_of(wide, live):
    """The bits of the two's complement sum of `live` activations of `wide` bits."""
    return wide + (live - 1).bit_length() + 1


def configure(weights, wide=8):
    """Point the planner at `dot <wide> <weights>` (every weight nonzero, at most four): the activations on
    rows 0..N-1, the state rows below them, the lane always row 30 and the one on row 31."""
    global WEIGHTS, WIDE, HOLD, FAN, STATE, STATE_ROWS, LANE
    WEIGHTS = tuple(weights)
    if not WEIGHTS or any(w not in (1, -1) for w in WEIGHTS) or not any(w > 0 for w in WEIGHTS):
        raise ValueError(f'weights {WEIGHTS}: nonzero, and at least one of them plus one')
    FAN = len(WEIGHTS)
    WIDE = wide
    HOLD = width_of(wide, FAN) - wide
    LANE = ROWS - 2
    STATE = LANE - FAN
    STATE_ROWS = frozenset(range(FAN, FAN + STATE))


def source_of(row, kind):
    if kind == UP:
        return (row - 1) % ROWS
    if kind == DOWN:
        return (row + 1) % ROWS
    if kind == HOP:
        return (row + ROWS // 4) % ROWS
    return (row + ROWS // 2) % ROWS


def readers_of(row):
    return [((row + 1) % ROWS, UP), ((row - 1) % ROWS, DOWN), ((row - ROWS // 4) % ROWS, HOP), ((row - ROWS // 2) % ROWS, LEAP)]


def negated(name):
    return name[1:] if name.startswith('~') else '~' + name


class Plan:
    def __init__(self):
        self.gates = {}
        self.carries = {}
        self.reads = {}
        self.protect = {}
        self.cost = 0
        self.log = []

    def clone(self):
        other = Plan.__new__(Plan)
        other.gates = dict(self.gates)
        other.carries = dict(self.carries)
        other.reads = {row: list(held) for row, held in self.reads.items()}
        other.protect = {row: list(held) for row, held in self.protect.items()}
        other.cost = self.cost
        other.log = list(self.log)
        return other

    def base(self, row):
        if row < FAN:
            return f'x{row}'
        if row in self.carries:
            return self.carries[row]
        if row == CONST:
            return 'one'
        return 'stale' if row in STATE_ROWS else 'zero'

    def hold(self, col, row):
        for c in range(col, 0, -1):
            gate = self.gates.get((c, row))
            if gate:
                return gate[1]
        return self.base(row)

    def next_gate(self, row, after):
        for c in range(after + 1, LAST + 1):
            if (c, row) in self.gates:
                return c
        return LAST + 1

    def fits(self, col, row, name):
        if (col, row) in self.gates:
            return False
        end = self.next_gate(row, col)
        for held in (self.reads.get(row, ()), self.protect.get(row, ())):
            for c, want in held:
                if col <= c < end and want != name:
                    return False
        return True

    def can_place(self, col, row, name):
        return row not in self.carries and self.fits(col, row, name)

    def add(self, col, row, kind, name, reads):
        self.gates[(col, row)] = (kind, name)
        for c, r, want in reads:
            self.reads.setdefault(r, []).append((c, want))
        self.cost += 1
        self.log.append((col, row, KIND_NAME[kind], name))

    def place_copy(self, col, row, kind):
        src = source_of(row, kind)
        name = self.hold(col - 1, src)
        self.add(col, row, kind, name, [(col - 1, src, name)])
        return name

    def place_not(self, col, row):
        name = negated(self.hold(col - 1, row))
        self.add(col, row, NOT, name, [(col - 1, row, negated(name))])
        return name

    def guard(self, row, first, last, name):
        held = self.protect.setdefault(row, [])
        for c in range(first, last + 1):
            held.append((c, name))

    def trim(self, row, last):
        held = self.protect.get(row)
        if held:
            self.protect[row] = [(c, name) for c, name in held if c <= last]

    def reach(self, name, until=LAST):
        """The cheapest gates that hold `name` on (row, col) and keep it there to `until`: {(row, col): (gates, path)}."""
        want = {name, negated(name)}
        best = {}
        heap = []
        counter = itertools.count()
        for row in range(ROWS):
            for col in range(0, until + 1):
                held = self.hold(col, row)
                if held in want:
                    heapq.heappush(heap, (0, col, next(counter), row, held, ()))
        while heap:
            gates, col, _, row, held, path = heapq.heappop(heap)
            key = (row, col, held)
            if key in best:
                continue
            best[key] = (gates, path)
            if col >= until:
                continue
            on_path = {r for _, r, _ in path}
            if (col + 1, row) not in self.gates and (row, col + 1, held) not in best:
                heapq.heappush(heap, (gates, col + 1, next(counter), row, held, path))
            flipped = negated(held)
            if row not in on_path and (row, col + 1, flipped) not in best and self.can_place(col + 1, row, flipped):
                heapq.heappush(heap, (gates + 1, col + 1, next(counter), row, flipped, path + ((col + 1, row, NOT),)))
            for reader, kind in readers_of(row):
                if reader in on_path or reader == row or (reader, col + 1, held) in best:
                    continue
                if self.can_place(col + 1, reader, held):
                    heapq.heappush(heap, (gates + 1, col + 1, next(counter), reader, held, path + ((col + 1, reader, kind),)))
        table = {}
        for (row, col, held), (gates, path) in best.items():
            if held != name:
                continue
            if all((c, row) not in self.gates for c in range(col + 1, until + 1)):
                if (row, col) not in table or table[(row, col)][0] > gates:
                    table[(row, col)] = (gates, path)
        return table

    def route(self, name, target, deadline, table=None):
        """Lay the cheapest copies that put `name` on `target` by `deadline` and keep it there; the arrival column."""
        if table is None:
            table = self.reach(name, deadline)
        options = [(gates, col, path) for (row, col), (gates, path) in table.items() if row == target and col <= deadline]
        if not options:
            return None
        gates, col, path = min(options)
        for c, r, kind in path:
            if kind == NOT:
                self.place_not(c, r)
            else:
                self.place_copy(c, r, kind)
        if self.hold(col, target) != name:
            return None
        self.guard(target, col, deadline, name)
        return col

    def carry_free(self, C):
        return C in STATE_ROWS and C not in self.carries and C not in self.reads and C not in self.protect and all((c, C) not in self.gates for c in range(1, LAST + 1))

    def place_cell(self, P, Q, S, cname, C, M, M1, flip_sum, flip_maj, subtract):
        """A serial adder, or subtractor when `subtract`, of P and Q; (row, column) the sum S lands on, or None."""
        if not self.carry_free(C):
            return None
        sum_rows = ((M - 1) % ROWS, M, (M + 1) % ROWS)
        maj_rows = ((M1 - 1) % ROWS, M1, (M1 + 1) % ROWS)
        for r in sum_rows + maj_rows:
            if r == CONST or (r in self.carries) or (r == C and r != M1):
                return None
        if set(sum_rows) & set(maj_rows):
            return None
        self.carries[C] = cname
        sum_P, sum_Q = (sum_rows[2], sum_rows[0]) if flip_sum else (sum_rows[0], sum_rows[2])
        maj_P, maj_Q = (maj_rows[2], maj_rows[0]) if flip_maj else (maj_rows[0], maj_rows[2])
        maj_P_name = negated(P) if subtract else P
        arrivals_sum, arrivals_maj = [], []
        for name, row, held in ((P, sum_P, arrivals_sum), (Q, sum_Q, arrivals_sum), (maj_P_name, maj_P, arrivals_maj), (Q, maj_Q, arrivals_maj)):
            at = self.route(name, row, LAST - 1)
            if at is None:
                return None
            held.append(at)
        at = self.route(cname, M, LAST - 1)
        if at is None:
            return None
        arrivals_sum.append(at)
        if M1 != C:
            at = self.route(cname, M1, LAST - 1)
            if at is None:
                return None
            arrivals_maj.append(at)
        k = max(arrivals_sum) + 1
        k1 = max(arrivals_maj) + 1
        if k > LAST or k1 > LAST:
            return None
        for r in sum_rows:
            self.trim(r, k - 1)
        if not self.fits(k, M, S):
            return None
        reads = [(k - 1, sum_rows[0], self.hold(k - 1, sum_rows[0])), (k - 1, M, self.hold(k - 1, M)), (k - 1, sum_rows[2], self.hold(k - 1, sum_rows[2]))]
        if sorted([reads[0][2], reads[2][2]]) != sorted([P, Q]) or reads[1][2] != cname:
            return None
        self.add(k, M, SUM, S, reads)
        new = cname + '+'
        for r in maj_rows:
            if r != C:
                self.trim(r, k1 - 1)
        reads = [(k1 - 1, maj_rows[0], self.hold(k1 - 1, maj_rows[0])), (k1 - 1, M1, self.hold(k1 - 1, M1)), (k1 - 1, maj_rows[2], self.hold(k1 - 1, maj_rows[2]))]
        if sorted([reads[0][2], reads[2][2]]) != sorted([maj_P_name, Q]) or reads[1][2] != cname:
            return None
        if M1 == C:
            if (k1, C) in self.gates or any(c >= k1 for c, want in self.reads.get(C, ()) if want == cname):
                return None
            self.add(k1, C, MAJ, new, reads)
            settled = k1
        else:
            if not self.fits(k1, M1, new):
                return None
            self.add(k1, M1, MAJ, new, reads)
            k2 = k1 + 1
            if k2 > LAST or (k2, C) in self.gates or any(c >= k2 for c, want in self.reads.get(C, ()) if want == cname):
                return None
            self.add(k2, C, LEAP, new, [(k2 - 1, M1, new)])
            settled = k2
        held = self.reads.setdefault(C, [])
        for c in range(0, LAST + 1):
            held.append((c, cname if c < settled else new))
        return (M, k)

    def grid(self):
        return {(col, row): kind for (col, row), (kind, _) in self.gates.items()}

    def columns_used(self):
        return max(col for col, _ in self.gates) if self.gates else 0


def placements(C):
    for M in ((C - ROWS // 4) % ROWS, (C - ROWS // 2) % ROWS):
        for M1 in (C, (C - ROWS // 2) % ROWS):
            if M1 == M:
                continue
            for flip_sum in (0, 1):
                for flip_maj in (0, 1):
                    yield M, M1, flip_sum, flip_maj


def estimate(tables, P, Q, cname, C, M, M1, flip_sum, flip_maj, subtract):
    """A cost from the one-source tables alone, before any route is laid."""
    maj_P_name = negated(P) if subtract else P
    sum_P, sum_Q = (M + 1, M - 1) if flip_sum else (M - 1, M + 1)
    maj_P, maj_Q = (M1 + 1, M1 - 1) if flip_maj else (M1 - 1, M1 + 1)
    wants = [(P, sum_P % ROWS), (Q, sum_Q % ROWS), (maj_P_name, maj_P % ROWS), (Q, maj_Q % ROWS), (cname, M)]
    if M1 != C:
        wants.append((cname, M1))
    total, latest = 0, 0
    for name, row in wants:
        table = tables[name]
        options = [(gates, col) for (r, col), (gates, _) in table.items() if r == row and col <= LAST - 1]
        if not options:
            return None
        gates, col = min(options)
        total += gates
        latest = max(latest, col)
    return total, latest


def grow(plan, P, Q, S, cname, subtract, beam):
    """Every way of adding the next cell, the cheapest `beam` of them committed exactly."""
    names = {P, Q, negated(P) if subtract else P}
    tables = {name: plan.reach(name, LAST - 1) for name in names}
    scored = []
    for C in sorted(STATE_ROWS):
        if not plan.carry_free(C):
            continue
        trial = plan.clone()
        trial.carries[C] = cname
        held = dict(tables)
        held[cname] = trial.reach(cname, LAST - 1)
        for M, M1, flip_sum, flip_maj in placements(C):
            guess = estimate(held, P, Q, cname, C, M, M1, flip_sum, flip_maj, subtract)
            if guess is not None:
                scored.append((guess, C, M, M1, flip_sum, flip_maj))
    scored.sort()
    grown = []
    for guess, C, M, M1, flip_sum, flip_maj in scored[:beam * 8]:
        trial = plan.clone()
        landed = trial.place_cell(P, Q, S, cname, C, M, M1, flip_sum, flip_maj, subtract)
        if landed is None:
            continue
        grown.append((trial.cost, landed[1], trial, landed[0]))
    grown.sort(key=lambda it: (it[0], it[1]))
    return grown[:beam]


def orders():
    """Every order of adding the operands that starts on a plus one (the running value is a copy)."""
    names = [f'x{i}' for i in range(FAN)]
    for first in [n for n in names if WEIGHTS[int(n[1:])] > 0]:
        rest = [n for n in names if n != first]
        for perm in itertools.permutations(rest):
            yield first, perm


def plan_order(first, perm, beam):
    beam_plans = [(0, 0, Plan(), first)]
    for stage, x in enumerate(perm):
        subtract = WEIGHTS[int(x[1:])] < 0
        S = f's{stage}'
        cname = f'c{stage}'
        next_beam = []
        for _, _, plan, R in beam_plans:
            for cost, k, grown, landed_row in grow(plan, R, x, S, cname, subtract, beam):
                next_beam.append((cost, k, grown, S))
        next_beam.sort(key=lambda it: (it[0], it[1]))
        beam_plans = next_beam[:beam]
        if not beam_plans:
            return None
    finished = []
    for _, _, plan, R in beam_plans:
        trial = plan.clone()
        at = trial.route(R, LANE, LAST)
        if at is None:
            continue
        finished.append((trial.cost, trial.columns_used(), trial))
    finished.sort(key=lambda it: (it[0], it[1]))
    return finished[0] if finished else None


def verified(grid, trials=200, hold=None):
    return dotsim.check(grid, WEIGHTS, WIDE, HOLD if hold is None else hold, trials=trials, state=STATE)


def best_plan(beam=4, limit=None, log=print):
    """The cheapest whole plan over the orders: (cost, columns, plan, order), or None."""
    best = None
    started = time.time()
    for n, (first, perm) in enumerate(orders()):
        if limit is not None and n >= limit:
            break
        found = plan_order(first, perm, beam)
        if found is None:
            log(f'{first} {" ".join(perm)}: no plan  ({time.time() - started:.0f}s)')
            continue
        cost, columns, plan = found
        whole, failure = verified(plan.grid())
        log(f'{first} {" ".join(perm)}: {cost} gates, {columns} columns, whole on {whole} of 200  ({time.time() - started:.0f}s)')
        if whole == 200 and (best is None or (cost, columns) < (best[0], best[1])):
            best = (cost, columns, plan, (first, perm))
    return best


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--verify' in sys.argv:
        grid = dotsim.read_grid(args[0])
        whole, failure = verified(grid)
        print(f'{args[0]}: {len(grid)} gates, whole on {whole} of 200', failure if failure else '')
        return
    out_grid = args[0] if args else 'dot.grid'
    out_json = args[1] if len(args) > 1 else out_grid.replace('.grid', '.json')
    beam = 4
    limit = None
    for a in sys.argv[1:]:
        if a.startswith('--beam='):
            beam = int(a.split('=')[1])
        if a.startswith('--orders='):
            limit = int(a.split('=')[1])
    best = best_plan(beam, limit, lambda line: print(line, flush=True))
    if best is None:
        print('no plan came out whole')
        return
    cost, columns, plan, order = best
    print(f'best: {order[0]} {" ".join(order[1])}: {cost} gates, {columns} columns')
    for col, row, kind, name in sorted(plan.log):
        print(f'  column {col:2} row {row:2} {kind:5} -> {name}')
    open(out_grid, 'w', encoding='utf-8').write(dotsim.as_grid_file(plan.grid()) + '\n')
    json.dump({'order': [order[0]] + list(order[1]), 'gates': [[c, r, k, n] for c, r, k, n in sorted(plan.log)], 'cost': cost, 'columns': columns},
              open(out_json, 'w', encoding='utf-8'), indent=1)
    print(f'written {out_grid} and {out_json}')


if __name__ == '__main__':
    main()
