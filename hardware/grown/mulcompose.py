"""Compose the sixteen kept partial-product pieces and fifteen serial adders into hardware/mul16.v,
a sixteen by sixteen bit-serial multiplier, and prove it in iverilog and size it with yosys.

usage: python mulcompose.py [--planned] [--vectors N] [--no-sim] [--synth] [--work DIR] [--out FILE]

This is the script that wrote hardware/mul16.v, published so the composition can be read rather
than trusted. It cannot be run end to end from this repository: turning a grid into Verilog is done
by `oracle netlist`, a subcommand of the circuit compiler, and that compiler is not published; nor
is the nine-gate serial adder grid its adders come from. Set ORACLE_NETLIST to a netlister that
takes `<exam> <grid> <out.v>` and put serial_add.txt and serial_add.grid beside this file if you
have them. What is checkable without any of that: the pieces themselves, by verify_pieces.py; the
top module below against the one in hardware/mul16.v; and the testbenches, which are written here in
full and can be run against the published mul16.v with iverilog. hardware/run-testbenches.sh does not
run them: it has two targets, the engine and the glue, and the glue target exercises mul16.v
indirectly at 58,273 elements.

The pieces: pp<i>fed.grid (the card's kept grid; `--planned` takes the planner's pp<i>.grid
instead) under the exam pp<i>.txt, netlisted as module pp<i>, each checked on the host model
against its own table first; the serial adder is the nine-gate piece serial_add.grid (module
serial_add) the ternary neuron builds its trees on. Every generated module resets on
`posedge rst` asynchronously; this script rewrites that one line (`always @(posedge clk or posedge
rst)` to `always @(posedge clk)`) so the reset is a synchronous clear. The research copy of this
script also wrote a header comment into every generated file saying so; this repository allows no
comments in a .v file, so those lines are gone and this docstring says it instead.

The composition, W = 16: x and y arrive a bit a clock, least significant first, sixteen of word and
sixteen of nought; every piece's lane output (y31, the live lane) is registered (r0), then a
balanced tree of serial adders, eight, four, two, one, a register on every node (r1..r4), so the
levels stay in lockstep and the product bit t is on `p` at clock t + 5. `clr` is synchronous:
asserted at the clock that takes the last input bit (bit 31) it clears the pieces at that edge,
and a shift register clears the level-j adders j clocks later, after they have taken their last
bit, so elements run back to back at thirty-two clocks an element; asserted on its own with the
inputs nought it clears the pieces and, over the next four clocks, the tree. The pipeline
registers are never cleared: they carry the last bits out while the next element starts.
"""
import os
import random
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dotsim
import ppplan

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.environ.get('ORACLE_NETLIST', 'oracle')
MUL = HERE
SERIAL_EXAM = os.path.join(HERE, 'serial_add.txt')
SERIAL_GRID = os.path.join(HERE, 'serial_add.grid')
W = 16
LEVELS = 4
LATENCY = LEVELS + 1
ASYNC = 'always @(posedge clk or posedge rst)'
SYNC = 'always @(posedge clk)'


def log(line):
    print(line, flush=True)


def read_exam(path):
    fields = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1].lstrip('-').isdigit():
            fields[parts[0]] = int(parts[1])
    return fields


def netlist(exam, grid, out):
    done = subprocess.run([ORACLE, 'netlist', exam, grid, out], cwd=HERE, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f'{ORACLE} netlist {exam} {grid} {out}:\n{done.stdout}{done.stderr}')


def synchronous(text, module):
    """The module's reset made synchronous: the one always line rewritten."""
    if text.count(ASYNC) != 1:
        raise SystemExit(f'{module}: expected one `{ASYNC}` line, found {text.count(ASYNC)}')
    return text.replace(ASYNC, SYNC)


class Piece:
    def __init__(self, i, exam, grid, work, planned):
        self.i = i
        self.module = f'pp{i:02d}'
        self.exam = exam
        self.grid = grid
        self.fields = read_exam(exam)
        self.rows = self.fields['rows']
        self.columns = self.fields['columns']
        self.lane = 2 + self.fields['state']
        self.kinds = dotsim.read_grid(grid, self.rows, self.columns)
        self.gates = len(self.kinds)
        self.logic = sum(1 for k in self.kinds.values() if k in (dotsim.SUM, dotsim.MAJ, dotsim.NOT, dotsim.XOR))
        self.verilog = os.path.join(work, f'{self.module}.v')
        netlist(exam, grid, self.verilog)
        self.text = synchronous(open(self.verilog, encoding='utf-8').read(), self.module)
        self.out = f'y{2 * W - 1}'

    def prove(self):
        geo = ppplan.Geometry(self.rows, self.columns, self.rows - 2 - self.lane)
        geo.apply()
        for wide, trials in ((8, 500), (16, 2000)):
            whole, first = ppplan.check(self.kinds, self.i, geo, wide, trials, seed=wide)
            if whole != trials:
                raise SystemExit(f'{self.module} ({self.grid}) W={wide}: whole on {whole} of {trials}, first failure {first}')


def pieces_of(work, planned):
    pieces = []
    for i in range(W):
        exam = os.path.join(MUL, f'pp{i:02d}.txt')
        grid = os.path.join(MUL, f'pp{i:02d}.grid' if planned else f'pp{i:02d}fed.grid')
        if not os.path.exists(grid):
            raise SystemExit(f'{grid}: not there')
        piece = Piece(i, exam, grid, work, planned)
        piece.prove()
        pieces.append(piece)
        log(f'  {piece.module}: {piece.gates} gates ({piece.logic} logic), rows {piece.rows}, columns {piece.columns}, lane {piece.lane}, whole on the host model')
    return pieces


def adder_of(work):
    out = os.path.join(work, 'serial_add.v')
    netlist(SERIAL_EXAM, SERIAL_GRID, out)
    text = synchronous(open(out, encoding='utf-8').read(), 'serial_add')
    fields = read_exam(SERIAL_EXAM)
    return text, f'y{fields["outputs"] - 1}', len(dotsim.read_grid(SERIAL_GRID, fields['rows'], fields['columns']))


def top(pieces, adder_out, source):
    lines = [
        'module mul16 (',
        '    input clk,',
        '    input clr,',
        '    input x,',
        '    input y,',
        '    output p',
        ');',
        f'    wire [{W - 1}:0] leaf;',
        f'    reg [{W - 1}:0] r0;',
        f'    reg [{LEVELS}:1] clr_d;',
    ]
    count = W
    for level in range(1, LEVELS + 1):
        count //= 2
        lines.append(f'    wire [{count - 1}:0] s{level};')
        lines.append(f'    reg [{count - 1}:0] r{level};')
    lines.append('')
    for piece in pieces:
        lines.append(f'    {piece.module} leaf{piece.i} (.clk(clk), .rst(clr), .x0(x), .x1(y), .{piece.out}(leaf[{piece.i}]));')
    count = W
    for level in range(1, LEVELS + 1):
        lines.append('')
        for j in range(count // 2):
            lines.append(f'    serial_add add{level}_{j} (.clk(clk), .rst(clr_d[{level}]), .x0(r{level - 1}[{2 * j}]), .x1(r{level - 1}[{2 * j + 1}]), .{adder_out}(s{level}[{j}]));')
        count //= 2
    lines += [
        '',
        f'    assign p = r{LEVELS}[0];',
        '',
        '    always @(posedge clk) begin',
        '        clr_d <= {clr_d[' + f'{LEVELS - 1}:1], clr' + '};',
        '        r0 <= leaf;',
    ]
    for level in range(1, LEVELS + 1):
        lines.append(f'        r{level} <= s{level};')
    lines += ['    end', 'endmodule', '']
    return '\n'.join(lines)


def vectors(trials, seed, out_dir):
    rng = random.Random(seed)
    top_v = (1 << W) - 1
    corners = [0, 1, 2, top_v, top_v - 1, 1 << (W - 1), (1 << (W - 1)) - 1, 0x5555, 0xAAAA, 3, 0x8001]
    pairs = [(a, b) for a in corners for b in corners]
    while len(pairs) < trials:
        pairs.append((rng.randrange(1 << W), rng.randrange(1 << W)))
    pairs = pairs[:max(trials, len(pairs))]
    names = {}
    for name, column in (('x', 0), ('y', 1)):
        path = os.path.join(out_dir, f'mul16_{name}.hex')
        with open(path, 'w', encoding='utf-8') as fh:
            for pair in pairs:
                fh.write(f'{pair[column]:04x}\n')
        names[name] = os.path.basename(path)
    path = os.path.join(out_dir, 'mul16_want.hex')
    with open(path, 'w', encoding='utf-8') as fh:
        for a, b in pairs:
            fh.write(f'{a * b:08x}\n')
    names['want'] = os.path.basename(path)
    return names, len(pairs)


def testbench(names, trials):
    return f'''`timescale 1ns/1ps
module tb;
    localparam T = {trials}, N = {2 * W}, L = {LATENCY};
    reg clk = 1'b0;
    reg clr = 1'b0;
    reg x = 1'b0, y = 1'b0;
    wire p;
    reg [{W - 1}:0] xs [0:T-1];
    reg [{W - 1}:0] ys [0:T-1];
    reg [{2 * W - 1}:0] want [0:T-1];
    reg [{2 * W - 1}:0] got [0:T-1];
    integer k, v, e, fails, alone_fails;
    mul16 dut (.clk(clk), .clr(clr), .x(x), .y(y), .p(p));
    task tick; begin #4; clk = 1'b1; #5; clk = 1'b0; #1; end endtask
    initial begin
        $readmemh("{names['x']}", xs);
        $readmemh("{names['y']}", ys);
        $readmemh("{names['want']}", want);
        fails = 0;
        alone_fails = 0;
        for (v = 0; v < T; v = v + 1) got[v] = {{{2 * W}{{1'b0}}}};
        clr = 1'b1; x = 1'b0; y = 1'b0;
        tick;
        clr = 1'b0;
        for (k = 0; k < N * T + L; k = k + 1) begin
            v = k / N;
            e = k % N;
            if (v < T) begin
                x = (e < {W}) ? xs[v][e] : 1'b0;
                y = (e < {W}) ? ys[v][e] : 1'b0;
                clr = (e == N - 1);
            end else begin
                x = 1'b0; y = 1'b0; clr = 1'b0;
            end
            #4;
            if (k >= L) got[(k - L) / N][(k - L) % N] = p;
            clk = 1'b1; #5; clk = 1'b0; #1;
        end
        for (v = 0; v < T; v = v + 1)
            if (got[v] !== want[v]) begin
                fails = fails + 1;
                if (fails <= 5) $display("pair %0d: %0d * %0d got %h want %h", v, xs[v], ys[v], got[v], want[v]);
            end
        $display("mul16 back to back: %0d of %0d products whole, %0d clocks a product, latency %0d", T - fails, T, N, L);
        for (v = 0; v < (T < 64 ? T : 64); v = v + 1) begin
            clr = 1'b1; x = 1'b0; y = 1'b0;
            tick;
            clr = 1'b0;
            got[v] = {{{2 * W}{{1'b0}}}};
            for (k = 0; k < N + L; k = k + 1) begin
                x = (k < {W}) ? xs[v][k] : 1'b0;
                y = (k < {W}) ? ys[v][k] : 1'b0;
                #4;
                if (k >= L) got[v][k - L] = p;
                clk = 1'b1; #5; clk = 1'b0; #1;
            end
            if (got[v] !== want[v]) begin
                alone_fails = alone_fails + 1;
                if (alone_fails <= 5) $display("alone pair %0d: %0d * %0d got %h want %h", v, xs[v], ys[v], got[v], want[v]);
            end
        end
        $display("mul16 one at a time: %0d of %0d products whole, %0d clocks a product", (T < 64 ? T : 64) - alone_fails, (T < 64 ? T : 64), N + L + 1);
        $finish;
    end
endmodule
'''


def piece_testbench(piece, names, trials, count):
    i = piece.i
    return f'''`timescale 1ns/1ps
module tb_{piece.module};
    localparam T = {trials}, N = {2 * W}, ALL = {count};
    reg clk = 1'b0;
    reg rst = 1'b0;
    reg x0 = 1'b0, x1 = 1'b0;
    wire lane;
    reg [{W - 1}:0] xs [0:ALL-1];
    reg [{W - 1}:0] ys [0:ALL-1];
    reg [{2 * W - 1}:0] got, want;
    integer k, v, fails;
    {piece.module} dut (.clk(clk), .rst(rst), .x0(x0), .x1(x1), .{piece.out}(lane));
    initial begin
        $readmemh("{names['x']}", xs);
        $readmemh("{names['y']}", ys);
        fails = 0;
        for (v = 0; v < T; v = v + 1) begin
            rst = 1'b1; x0 = 1'b0; x1 = 1'b0;
            #4; clk = 1'b1; #5; clk = 1'b0; #1;
            rst = 1'b0;
            got = {{{2 * W}{{1'b0}}}};
            for (k = 0; k < N; k = k + 1) begin
                x0 = (k < {W}) ? xs[v][k] : 1'b0;
                x1 = (k < {W}) ? ys[v][k] : 1'b0;
                #4;
                got[k] = lane;
                clk = 1'b1; #5; clk = 1'b0; #1;
            end
            want = (xs[v] * ys[v][{i}]) << {i};
            if (got !== want) begin
                fails = fails + 1;
                if (fails <= 3) $display("{piece.module} pair %0d: %0d * %0d got %h want %h", v, xs[v], ys[v], got, want);
            end
        end
        $display("{piece.module}: %0d of %0d whole", T - fails, T);
        $finish;
    end
endmodule
'''


def run_sim(work, tb, sources, name):
    vvp = os.path.join(work, f'{name}.vvp')
    started = time.time()
    done = subprocess.run(['iverilog', '-g2012', '-o', vvp, tb] + sources, cwd=work, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f'iverilog {name}: {done.stdout}{done.stderr}')
    done = subprocess.run(['vvp', '-n', vvp], cwd=work, capture_output=True, text=True)
    text = done.stdout.strip()
    log(f'  {text.replace(chr(10), chr(10) + "  ")}  ({time.time() - started:.0f}s)')
    return text


def simulate(work, out, pieces, trials, seed):
    names, count = vectors(trials, seed, work)
    for piece in pieces:
        tb = os.path.join(work, f'tb_{piece.module}.v')
        open(tb, 'w', encoding='utf-8').write(piece_testbench(piece, names, min(count, 300), count))
        text = run_sim(work, tb, [out], f'tb_{piece.module}')
        found = re.search(rf'{piece.module}: (\d+) of (\d+) whole', text)
        if not found or found.group(1) != found.group(2):
            raise SystemExit(f'{piece.module} is not whole in iverilog:\n{text}')
    tb = os.path.join(work, 'tb_mul16.v')
    open(tb, 'w', encoding='utf-8').write(testbench(names, count))
    text = run_sim(work, tb, [out], 'tb_mul16')
    back = re.search(r'back to back: (\d+) of (\d+)', text)
    alone = re.search(r'one at a time: (\d+) of (\d+)', text)
    if not back or back.group(1) != back.group(2) or not alone or alone.group(1) != alone.group(2):
        raise SystemExit(f'mul16 is not whole in iverilog:\n{text}')
    return {'pairs': count, 'log': text}


def synthesize(work, out):
    """yosys synth_xilinx -family xcup on mul16 alone, no IO pads: LUTs, flops, and the deepest path in LUT
    levels by `ltp` over every cell but the FDRE flops (`-noff` alone does not know the mapped flops)."""
    script = os.path.join(work, 'mul16_synth.ys')
    report = os.path.join(work, 'mul16_synth.log')
    stat = os.path.join(work, 'mul16_stat.txt')
    ltp = os.path.join(work, 'mul16_ltp.txt')
    open(script, 'w', encoding='utf-8').write(
        f'read_verilog {out}\nhierarchy -top mul16\nsynth_xilinx -family xcup -top mul16 -flatten -noiopad\n'
        f'tee -o {stat} stat\ntee -o {ltp} ltp -noff t:FDRE %n\n')
    started = time.time()
    done = subprocess.run(['yosys', '-q', '-l', report, '-s', script], cwd=work, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f'yosys: {done.stdout[-3000:]}{done.stderr[-3000:]}')
    cells = {}
    for line in open(stat, encoding='utf-8'):
        found = re.match(r'^\s*(\d+)\s+(\$?[A-Za-z_][A-Za-z0-9_$]*)\s*$', line)
        if found:
            cells[found.group(2)] = int(found.group(1))
    luts = sum(n for cell, n in cells.items() if re.fullmatch(r'LUT\d', cell))
    ffs = sum(n for cell, n in cells.items() if cell.startswith('FD'))
    depth, path = None, []
    for line in open(ltp, encoding='utf-8'):
        found = re.search(r'length=(\d+)', line)
        if found:
            depth = int(found.group(1))
        elif re.match(r'^\s+\d+: ', line):
            path.append(line.strip())
    log(f'  yosys: {luts} LUTs, {ffs} FFs, deepest path {depth} LUT levels (ltp over every cell but the FDRE flops: {"; ".join(path)}) ({time.time() - started:.0f}s)')
    log('  cells: ' + ', '.join(f'{cell} {n}' for cell, n in sorted(cells.items())))
    return {'cells': cells, 'luts': luts, 'ffs': ffs, 'depth': depth, 'path': path}


def main():
    args = sys.argv[1:]
    planned = '--planned' in args
    sim = '--no-sim' not in args
    synth = '--synth' in args
    trials = 2000
    work = os.path.join(tempfile.gettempdir(), 'mulcompose')
    out = os.path.join(work, 'mul16.v')
    i = 0
    while i < len(args):
        if args[i] == '--vectors':
            trials = int(args[i + 1])
            i += 1
        elif args[i] == '--work':
            work = args[i + 1]
            i += 1
        elif args[i] == '--out':
            out = args[i + 1]
            i += 1
        i += 1
    os.makedirs(work, exist_ok=True)
    source = "the planner's grids pp<i>.grid" if planned else "the card's kept grids pp<i>fed.grid"
    pieces = pieces_of(work, planned)
    adder_text, adder_out, adder_gates = adder_of(work)
    text = ''.join(piece.text + '\n' for piece in pieces) + adder_text + '\n' + top(pieces, adder_out, source)
    open(out, 'w', encoding='utf-8').write(text)
    gates = sum(piece.gates for piece in pieces) + 15 * adder_gates
    logic = sum(piece.logic for piece in pieces) + 15 * 2
    log(f'{out}: {len(pieces)} pieces ({sum(piece.gates for piece in pieces)} gates) + 15 serial adders ({adder_gates} gates each) '
        f'= {gates} gates, {logic} of them logic; tree {LEVELS} deep, latency {LATENCY}, {2 * W} clocks an element back to back, '
        f'{2 * W + LATENCY} for one alone')
    if sim:
        simulate(work, out, pieces, trials, seed=1)
    if synth:
        synthesize(work, out)


if __name__ == '__main__':
    main()
