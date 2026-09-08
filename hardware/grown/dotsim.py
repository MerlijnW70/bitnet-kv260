"""A host model of oracle's grid for the dot exam: `fan` input rows fed a bit a step, `state` rows
carrying their last column into the next step's column 0, the bottom row a one, the answer read
off the lane row (fan + state) at the last column of every step.

Kinds: 0 wire, 3 xor(up,down), 4 not(straight), 5 maj(up,down,straight), 7 sum(up,down,straight),
10 up (row-1), 11 down (row+1), 13 leap (row+rows/2), 14 hop (row+rows/4).
A grid is {(column, row): kind} with column in 1..columns-1.
"""
import random

ROWS, COLUMNS, FAN, STATE = 32, 12, 4, 26
UP, DOWN, LEAP, HOP, SUM, MAJ, NOT, XOR = 10, 11, 13, 14, 7, 5, 4, 3


def step(grid, carried, bits, rows=ROWS, fan=FAN, state=STATE, columns=COLUMNS):
    column = [0] * rows
    for row in range(fan):
        column[row] = bits[row]
    for row in range(fan, fan + state):
        column[row] = carried[row]
    column[rows - 1] = 1
    history = [column]
    for at in range(1, columns):
        prev = history[-1]
        now = list(prev)
        for row in range(rows):
            kind = grid.get((at, row), 0)
            if kind == 0:
                continue
            up = prev[(row - 1) % rows]
            down = prev[(row + 1) % rows]
            straight = prev[row]
            if kind == UP:
                now[row] = up
            elif kind == DOWN:
                now[row] = down
            elif kind == LEAP:
                now[row] = prev[(row + rows // 2) % rows]
            elif kind == HOP:
                now[row] = prev[(row + rows // 4) % rows]
            elif kind == SUM:
                now[row] = (up + down + straight) & 1
            elif kind == MAJ:
                now[row] = (up + down + straight) // 2
            elif kind == NOT:
                now[row] = 1 - straight
            elif kind == XOR:
                now[row] = up ^ down
            else:
                raise SystemExit(f"kind {kind} not modelled")
        history.append(now)
    return history


def run(grid, words, rows=ROWS, fan=FAN, state=STATE, columns=COLUMNS):
    carried = [0] * rows
    lasts = []
    for bits in words:
        history = step(grid, carried, bits, rows, fan, state, columns)
        lasts.append(history[-1])
        carried = list(history[-1])
    return lasts


def words_of(values, wide, hold):
    fan = len(values)
    fed = [[(values[i] >> t) & 1 for i in range(fan)] for t in range(wide)]
    return fed + [[0] * fan for _ in range(hold)]


def dot(values, weights):
    return sum(w * x for w, x in zip(weights, values))


def check(grid, weights, wide=8, hold=3, trials=200, seed=1, rows=ROWS, state=STATE, columns=COLUMNS, lane=None):
    """How many of the trials come out whole on the lane row, and the first failure."""
    fan = len(weights)
    lane = fan + state if lane is None else lane
    width = wide + hold
    rng = random.Random(seed)
    cases = [[0] * fan, [(1 << wide) - 1] * fan, [(1 << wide) - 1 if w < 0 else 0 for w in weights]]
    while len(cases) < trials:
        cases.append([rng.randrange(1 << wide) for _ in range(fan)])
    whole, first = 0, None
    for values in cases:
        want = dot(values, weights) % (1 << width)
        lasts = run(grid, words_of(values, wide, hold), rows, fan, state, columns)
        got = sum(lasts[t][lane] << t for t in range(width))
        if got == want:
            whole += 1
        elif first is None:
            first = (values, want, got)
    return whole, first


def read_grid(path, rows=ROWS, columns=COLUMNS):
    kinds = [int(k) for k in open(path, encoding='utf-8').read().split()]
    grid = {}
    for at in range(1, columns):
        for row in range(rows):
            kind = kinds[(at - 1) * rows + row]
            if kind:
                grid[(at, row)] = kind
    return grid


def as_grid_file(grid, rows=ROWS, columns=COLUMNS):
    kinds = []
    for at in range(1, columns):
        for row in range(rows):
            kinds.append(str(grid.get((at, row), 0)))
    return ' '.join(kinds)
