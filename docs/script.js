(() => {
  /* ------------------------------------------------------------------
     DATA: the real measurements, copied from the sketches. Swap freely.
     ------------------------------------------------------------------ */
  const DATA = {
    /* run 1: how well the reader predicts each rule's nudge from the notes (0 = can't, 1 = perfectly) */
    readable: { 1: { apart: .96, line: .87, close: .77 }, 2: { apart: .92, line: .95, close: .95 } },
    /* a machine that never practised */
    unpractised: { 1: { apart: .34, line: .64, close: .85 }, 2: { apart: .31, line: .69, close: .85 } },
    /* keeps[run][round deafened]: how much of each rule's push is left (below 0 = pushed the wrong way, above 1 = more than before) */
    keeps: [
      { 1: { apart: 0.5766, line: 0.4995, close: 0.9415 }, 2: { apart: -0.1117, line: 0.0202, close: 0.0175 } },
      { 1: { apart: 0.2324, line: 0.6292, close: 0.1623 }, 2: { apart: -0.2647, line: -0.0268, close: 0.1622 } },
      { 1: { apart: 0.4188, line: 0.7572, close: 1.2054 }, 2: { apart: -0.1085, line: 0.0188, close: 0.0707 } },
      { 1: { apart: 0.1323, line: 0.3141, close: 0.1124 }, 2: { apart: 0.6295, line: 0.0059, close: 0.6281 } }
    ],
    score: [.993, .992, .994, .990],   // held-out R², clips 700–999: 0.99347, 0.99182, 0.99357, 0.99026
    /* what switching a rule off does in the simulator, measured on 10 flocks of 12 birds */
    toy: {
      apart: 'Keep your distance is off, so birds crowd in. In my simulator the nearest neighbour sits 25% closer.',
      line: 'Line up is off, so nobody agrees on a direction. Heading together drops from 0.97 to 0.22.',
      close: 'Stay close is off, so the flock loosens. The nearest neighbour drifts 27% further away.'
    },
    /* stay close through training, run 1, notes after round two. Step 0 is the never-practised machine */
    practice: {
      total: 40950, pass1: 1365, half: 431, eighty: 1117,
      steps: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,19,21,23,25,27,29,32,35,38,41,45,49,54,59,64,70,76,83,91,99,108,117,128,140,152,166,181,197,215,235,256,279,304,332,362,395,431,470,512,558,609,664,724,790,861,939,1024,1117,1218,1328,1448,1579,1722,1878,2048,2233,2435,2656,2896,3158,3444,3756,4096,4467,4871,5312,5793,6317,6889,7512,8192,8933,9742,10624,11585,12634,13777,15024,16384,17867,19484,21247,23170,25268,27554,30048,32768,35734,38968,40950],
      readable: [0.847,0.843,0.839,0.834,0.832,0.83,0.828,0.825,0.822,0.818,0.815,0.812,0.808,0.805,0.797,0.795,0.793,0.792,0.795,0.798,0.802,0.805,0.809,0.815,0.82,0.824,0.831,0.84,0.848,0.852,0.854,0.856,0.855,0.854,0.86,0.868,0.875,0.881,0.883,0.884,0.886,0.884,0.883,0.884,0.884,0.881,0.88,0.882,0.885,0.887,0.889,0.895,0.896,0.904,0.908,0.911,0.915,0.918,0.922,0.927,0.928,0.928,0.929,0.931,0.935,0.94,0.942,0.943,0.947,0.947,0.945,0.945,0.944,0.942,0.941,0.94,0.938,0.932,0.93,0.926,0.922,0.922,0.922,0.918,0.93,0.922,0.922,0.921,0.922,0.929,0.923,0.93,0.931,0.931,0.931,0.933,0.935,0.939,0.943,0.943,0.954,0.954,0.951,0.949,0.955,0.953,0.949],
      used: [0.056,-0.074,-0.082,-0.03,0.017,0.05,0.074,0.09,0.102,0.113,0.123,0.134,0.144,0.152,0.158,0.157,0.156,0.158,0.166,0.17,0.172,0.174,0.178,0.197,0.216,0.232,0.252,0.257,0.258,0.269,0.28,0.292,0.302,0.333,0.349,0.352,0.348,0.326,0.324,0.332,0.327,0.35,0.342,0.345,0.367,0.389,0.39,0.408,0.418,0.46,0.461,0.448,0.466,0.531,0.542,0.534,0.598,0.627,0.614,0.672,0.715,0.759,0.778,0.796,0.834,0.847,0.851,0.871,0.847,0.875,0.905,0.895,0.907,0.904,0.885,0.922,0.906,0.959,0.919,0.909,0.922,0.931,0.969,0.955,0.928,0.948,0.962,0.958,0.972,0.974,0.997,0.963,0.963,0.976,0.986,0.97,0.988,0.985,0.982,0.994,0.986,0.975,0.99,1.002,0.989,0.983,0.992]
    },
    /* the same two curves for the other two rules (run 1, notes after round two, same checkpoints as above) */
    practiceAll: {
      line: { readable: [0.679,0.684,0.675,0.671,0.668,0.665,0.662,0.658,0.654,0.65,0.647,0.646,0.645,0.645,0.646,0.645,0.645,0.644,0.643,0.643,0.644,0.646,0.648,0.651,0.655,0.661,0.669,0.688,0.706,0.717,0.726,0.736,0.749,0.76,0.766,0.773,0.775,0.773,0.777,0.782,0.794,0.797,0.802,0.805,0.8,0.804,0.811,0.816,0.817,0.816,0.825,0.833,0.836,0.844,0.848,0.855,0.86,0.867,0.876,0.887,0.892,0.899,0.898,0.902,0.904,0.906,0.908,0.912,0.916,0.919,0.922,0.924,0.926,0.927,0.93,0.931,0.932,0.933,0.931,0.931,0.935,0.937,0.938,0.936,0.939,0.94,0.943,0.943,0.94,0.935,0.942,0.943,0.944,0.941,0.943,0.943,0.946,0.946,0.946,0.947,0.949,0.948,0.949,0.944,0.948,0.948,0.947], used: [-0.168,0.097,0.211,0.216,0.197,0.183,0.179,0.188,0.203,0.225,0.25,0.28,0.31,0.34,0.391,0.409,0.422,0.427,0.417,0.398,0.381,0.372,0.368,0.379,0.407,0.443,0.461,0.454,0.445,0.476,0.49,0.496,0.52,0.567,0.578,0.588,0.611,0.612,0.615,0.62,0.616,0.657,0.665,0.662,0.655,0.672,0.709,0.715,0.7,0.688,0.722,0.771,0.741,0.758,0.771,0.784,0.823,0.816,0.81,0.83,0.852,0.872,0.895,0.9,0.892,0.919,0.893,0.949,0.907,0.909,0.96,0.921,0.943,0.932,0.929,0.945,0.976,0.972,0.951,0.972,0.972,0.95,0.978,0.967,0.962,0.976,0.978,0.976,0.975,0.987,0.986,0.979,0.984,0.991,0.995,0.974,0.985,0.976,0.994,0.986,1.0,0.987,0.99,0.99,0.992,0.99,0.996] },
      apart: { readable: [0.315,0.311,0.307,0.31,0.312,0.312,0.311,0.308,0.305,0.301,0.297,0.293,0.289,0.285,0.279,0.278,0.277,0.279,0.283,0.289,0.294,0.299,0.302,0.307,0.312,0.317,0.323,0.332,0.345,0.356,0.369,0.382,0.393,0.402,0.417,0.427,0.431,0.433,0.427,0.431,0.448,0.446,0.436,0.441,0.427,0.417,0.421,0.416,0.43,0.428,0.445,0.458,0.456,0.487,0.502,0.533,0.549,0.575,0.615,0.677,0.72,0.761,0.787,0.798,0.81,0.82,0.825,0.832,0.839,0.849,0.851,0.854,0.862,0.864,0.873,0.879,0.889,0.895,0.895,0.9,0.896,0.895,0.9,0.896,0.902,0.903,0.902,0.906,0.903,0.905,0.903,0.914,0.917,0.921,0.92,0.921,0.923,0.924,0.925,0.924,0.935,0.934,0.93,0.927,0.935,0.929,0.922], used: [-0.004,-0.051,-0.028,0.016,0.046,0.066,0.08,0.093,0.105,0.117,0.128,0.14,0.15,0.158,0.168,0.17,0.171,0.171,0.171,0.167,0.163,0.16,0.161,0.169,0.181,0.193,0.202,0.2,0.194,0.199,0.198,0.195,0.187,0.193,0.199,0.203,0.207,0.201,0.203,0.215,0.208,0.216,0.217,0.224,0.235,0.245,0.24,0.261,0.272,0.289,0.301,0.329,0.33,0.369,0.392,0.416,0.442,0.469,0.496,0.547,0.602,0.689,0.694,0.746,0.761,0.792,0.777,0.824,0.801,0.834,0.873,0.856,0.88,0.845,0.862,0.886,0.893,0.934,0.892,0.897,0.916,0.902,0.935,0.916,0.915,0.941,0.934,0.919,0.97,0.939,0.983,0.958,0.957,0.971,0.987,0.949,0.992,0.971,0.976,0.978,0.988,0.96,0.974,0.982,0.991,0.983,0.975] }
    },
    /* heading together after 400 steps of free flight, runs 1 to 4: [normal, round one deaf, round two deaf] */
    together: [[0.97, 0.81, 0.23], [0.97, 0.9, 0.28], [0.96, 0.75, 0.22], [0.97, 0.73, 0.28]]
  };
  const NAME = { apart: 'keep your distance', line: 'line up', close: 'stay close' };
  const HUE = { apart: '226,61,109', line: '42,107,242', close: '237,164,20' }, TEAL = '25,167,155';
  const NOISE0 = .26, NOISE_GONE = .6, NOISE_LINE = 1.15;      // stand-in flock noise, tuned so run 1 reads about 0.97 / 0.8 / 0.25

  /* A stand-in flock. It does not run the real machine: it flies the three simulator rules, each one turned
     down by the share the real machine was measured to lose. W = how much of each rule is left (1 = all).
     near = how far a bird looks for neighbours (0 = it hears every bird, as the machine does with twelve). */
  const RULES = ['apart', 'line', 'close'];
  let spare = null;
  function gauss() {
    if (spare !== null) { const g = spare; spare = null; return g; }
    let u = 0; while (!u) u = Math.random();
    const m = Math.sqrt(-2 * Math.log(u)), v = Math.random(); spare = m * Math.sin(6.2832 * v); return m * Math.cos(6.2832 * v);
  }
  const my0 = (f) => { let y = 0; for (const b of f.b) y += b.y; return y / f.n; };
  class Flock {
    constructor(n, o) {
      o = o || {};
      this.n = n; this.b = []; this.w = 0; this.h = 0; this.s = 1;
      this.near = o.near || 0; this.inner = o.inner || 0; this.tug = o.tug || 1.6; this.jit = o.jit || 0; this.body = o.body || 0; this.gap = o.gap || 36; this.reach = o.reach || 84; this.pace = o.pace || 92; this.err = o.err === undefined ? 1 : o.err;
      this.W = { apart: 1, line: 1, close: 1 }; this.Wt = { apart: 1, line: 1, close: 1 }; this.c = 1; this.ct = 1;
      this.wander = Math.random() * 6.28; this.pol = 1; this.follow = 0;
      this.rec = []; this.pool = []; this.lag = 0;   // a short recording, so a scene can show the flock a moment late and already know where it goes next
      this.ask = { apart: [0, 0], line: [0, 0], close: [0, 0] }; this.rms = { apart: .6, line: .25, close: .5 };
      for (let i = 0; i < n; i++) this.b.push({ x: 0, y: 0, hx: 1, hy: 0, nx: 1, ny: 0, ph: Math.random() * 6.28, hist: [] });
    }
    seed(b) {
      const a = -.3 + gauss() * .15, spread = Math.sqrt(this.n / 12);
      b.x = this.w * .4 + gauss() * 30 * this.s * spread; b.y = this.h * .5 + gauss() * 18 * this.s * spread;
      b.x = Math.max(8, Math.min(this.w - 8, b.x)); b.y = Math.max(8, Math.min(this.h - 8, b.y));
      b.hx = Math.cos(a); b.hy = Math.sin(a); b.nx = b.hx; b.ny = b.hy; b.hist.length = 0;
    }
    fit(w, h, s) {
      const fresh = !this.w;
      if (w === this.w && h === this.h && s === this.s) return;
      if (!fresh) {
        const kx = w / this.w, ky = h / this.h;
        for (const b of this.b) { b.x *= kx; b.y *= ky; b.hist.length = 0; }
        for (const r of this.rec) for (let o = 0; o < r.a.length; o += 5) { r.a[o] *= kx; r.a[o + 1] *= ky; }
      }
      this.w = w; this.h = h; this.s = s;
      if (fresh) for (const b of this.b) this.seed(b);
    }
    aim(W, c) { this.Wt = W; this.ct = c; }
    record(t, dt, lagTo) {
      const R = this.rec, n = this.n;
      if (R.length && t - R[R.length - 1].t > .3) { while (R.length) this.pool.push(R.pop().a); this.lag = 0; }   // the recording has a hole in it: start again
      const a = this.pool.pop() || new Float32Array(n * 5);
      for (let i = 0; i < n; i++) { const b = this.b[i], o = i * 5; a[o] = b.x; a[o + 1] = b.y; a[o + 2] = b.hx; a[o + 3] = b.hy; a[o + 4] = b.ph; }
      R.push({ t, a });
      while (R.length > 1 && R[0].t < t - lagTo - 1.3) this.pool.push(R.shift().a);
      this.lag = Math.min(lagTo, this.lag + dt * .5);   // slip behind gently, at half speed, instead of freezing
    }
    shown(t) {   // index of the recorded moment being shown at time t
      const R = this.rec; let k = R.length - 1; while (k > 0 && R[k].t > t - this.lag) k--; return k;
    }
    step(dt, t) {
      const { w, h, s, n } = this, B = this.b;
      const av = this.avoid || null, gustY = (this.gust || 0) * Math.max(0, Math.min(1, (h * .8 - my0(this)) / (h * .5)));
      const k = Math.min(1, dt * 4.5);
      for (const r of RULES) this.W[r] += (this.Wt[r] - this.W[r]) * k;
      this.c += (this.ct - this.c) * k;
      const held = this.polGoal != null;   // a flock that has to hit a measured tightness keeps enough pull to be steered there
      const c = this.c, wS = c * this.W.apart, wA = c * (held ? Math.max(this.W.line, .6) : this.W.line), wC = c * this.W.close, wAp = Math.max(0, wA);
      const cl = (v) => Math.max(0, Math.min(1, v));
      const gone = (3 - cl(this.W.apart) - cl(this.W.line) - cl(this.W.close)) / 3;
      let sigma = 1.0 + (NOISE0 - 1.0) * c + c * this.err * (NOISE_GONE * gone + NOISE_LINE * (1 - cl(this.W.line))) + this.jit;
      if (this.polGoal != null) {   // steer the wobble until "heading together" settles on the measured value
        const err = this.pol - this.polGoal;
        this.sigCtl = Math.max(.04, Math.min(2.4, (this.sigCtl === undefined ? Math.min(sigma, 1.2) : this.sigCtl) + err * dt * 1.5));
        sigma = Math.max(.04, this.sigCtl + err * 2.2);   // the slow part finds the level, the quick part damps the swings
      } else this.sigCtl = undefined;
      const speed = this.pace * s * (.62 + .38 * c), v = this.pace / 64;   // faster birds need firmer steering at the edges
      const sepR = this.gap * s, m = Math.min(w, h) * .22, M = Math.min(w, h) * .37, near = this.near * s, near2 = near * near, reach = this.reach * s, inner = this.inner * s, body = this.body * s;
      this.wander += gauss() * .55 * Math.sqrt(dt);
      const wx = Math.cos(this.wander), wy = Math.sin(this.wander);
      let mx = 0, my = 0, ahx = 0, ahy = 0;
      for (const b of B) {
        if (!(isFinite(b.x) && isFinite(b.y) && isFinite(b.hx) && isFinite(b.hy))) this.seed(b);   // never lose a bird to a bad number
        mx += b.x; my += b.y; ahx += b.hx; ahy += b.hy;
      }
      let gx = 0, gy = 0;   // a small flock turns as one body before single birds reach an edge
      if (!near) {
        const gxc = mx / n, gyc = my / n;
        if (gxc < M) gx = (M - gxc) / M; else if (gxc > w - M) gx = -(gxc - (w - M)) / M;
        if (gyc < M) gy = (M - gyc) / M; else if (gyc > h - M) gy = -(gyc - (h - M)) / M;
      }
      /* the safe zone: the followed bird starts turning back a little before the edge of it. A flock that is flying
         together is leaned the same way, so it turns as one body instead of leaving the followed bird behind */
      const sf = this.safe || null; let zx = 0, zy = 0;
      if (sf) {
        const F = B[this.follow], qx = Math.min(30 * s, (sf.x1 - sf.x0) / 3) || 1, qy = Math.min(30 * s, (sf.y1 - sf.y0) / 3) || 1;
        if (F.x < sf.x0 + qx) zx = Math.min(2, (sf.x0 + qx - F.x) / qx); else if (F.x > sf.x1 - qx) zx = -Math.min(2, (F.x - (sf.x1 - qx)) / qx);
        if (F.y < sf.y0 + qy) zy = Math.min(2, (sf.y0 + qy - F.y) / qy); else if (F.y > sf.y1 - qy) zy = -Math.min(2, (F.y - (sf.y1 - qy)) / qy);
      }
      for (let i = 0; i < n; i++) {
        const b = B[i];
        let sx = 0, sy = 0, bx = 0, by = 0, lhx = 0, lhy = 0, lcx = 0, lcy = 0, ln = 0;
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          const o = B[j], dx = o.x - b.x, dy = o.y - b.y, d2 = dx * dx + dy * dy;
          if (near && d2 > near2) continue;
          if (near) { ln++; lhx += o.hx; lhy += o.hy; lcx += dx; lcy += dy; }
          if (d2 < sepR * sepR && d2 > 1e-6) {
            const dist = Math.sqrt(d2), f = 1 - dist / sepR; sx -= dx / dist * f; sy -= dy / dist * f;
            if (dist < body) { const g = 1 - dist / body; bx -= dx / dist * g; by -= dy / dist * g; }   // two birds can't share one spot, rule or no rule
          }
        }
        let ax, ay, cx, cy;
        if (near) { if (ln) { ax = lhx / ln - b.hx; ay = lhy / ln - b.hy; cx = lcx / ln; cy = lcy / ln; } else ax = ay = cx = cy = 0; }
        else { ax = (ahx - b.hx) / (n - 1) - b.hx; ay = (ahy - b.hy) / (n - 1) - b.hy; cx = (mx - b.x) / (n - 1) - b.x; cy = (my - b.y) / (n - 1) - b.y; }
        const cd = Math.hypot(cx, cy) || 1, pull = Math.max(0, Math.min(1, (cd - inner) / reach));   // stay close only pulls once a bird has drifted out
        const qS = [9 * sx, 9 * sy], qA = [4.8 * ax, 4.8 * ay], qC = [this.tug * cx / cd * pull, this.tug * cy / cd * pull];   // what each rule asks of this bird
        if (i === this.follow) {
          const e = Math.min(1, dt / .3), e2 = Math.min(1, dt / 3);
          for (const [r, q] of [['apart', qS], ['line', qA], ['close', qC]]) {
            const a = this.ask[r]; a[0] += (q[0] - a[0]) * e; a[1] += (q[1] - a[1]) * e;
            this.rms[r] = Math.sqrt(this.rms[r] * this.rms[r] * (1 - e2) + (q[0] * q[0] + q[1] * q[1]) * e2) || .01;
          }
        }
        let fx = wS * qS[0] + wA * qA[0] + wC * qC[0] + wAp * (.5 * wx + 5.5 * v * gx) + 14 * bx;
        let fy = wS * qS[1] + wA * qA[1] + wC * qC[1] + wAp * (.5 * wy + 5.5 * v * gy) + 14 * by;
        if (b.x < m) fx += (m - b.x) / m * 6.5 * v; else if (b.x > w - m) fx -= (b.x - (w - m)) / m * 6.5 * v;
        if (b.y < m) fy += (m - b.y) / m * 6.5 * v; else if (b.y > h - m) fy -= (b.y - (h - m)) / m * 6.5 * v;
        fy += gustY;
        if (zx || zy) { const zk = (i === this.follow ? 10 : 4.5 * Math.min(1, wAp)) * v; fx += zx * zk; fy += zy * zk; }
        if (av) {   // a label sits over this patch of sky, so birds drift back toward the middle instead of hiding behind it
          const ox = Math.max(av.x0 - b.x, 0, b.x - av.x1), oy = Math.max(av.y0 - b.y, 0, b.y - av.y1), od = Math.hypot(ox, oy);
          if (od < av.m) { const tx = w * .42 - b.x, ty = h * .58 - b.y, tl = Math.hypot(tx, ty) || 1, p = (1 - od / av.m) * 11 * v; fx += tx / tl * p; fy += ty / tl * p; }
        }
        const nx = b.hx + fx * dt, ny = b.hy + fy * dt;
        const th = gauss() * sigma * Math.sqrt(dt), ct = Math.cos(th), st = Math.sin(th);
        const rx = nx * ct - ny * st, ry = nx * st + ny * ct, len = Math.hypot(rx, ry) || 1;
        b.nx = rx / len; b.ny = ry / len;
      }
      let px = 0, py = 0;
      const keep = n > 30 ? 28 : 60;
      for (const b of B) {
        b.hx = b.nx; b.hy = b.ny;
        const sp = speed * (1 + .2 * (1 - c) * Math.sin(b.ph * .2 + t * 2.1));
        b.x += b.hx * sp * dt; b.y += b.hy * sp * dt;
        if (b.x < 5) { b.x = 5; b.hx = Math.abs(b.hx); } else if (b.x > w - 5) { b.x = w - 5; b.hx = -Math.abs(b.hx); }
        if (b.y < 5) { b.y = 5; b.hy = Math.abs(b.hy); } else if (b.y > h - 2) { b.y = h - 2; b.hy = -Math.abs(b.hy); }
        b.ph += dt * (13 + 4 * c);
        b.hist.push(b.x, b.y); if (b.hist.length > keep) b.hist.splice(0, b.hist.length - keep);
        px += b.hx; py += b.hy;
      }
      if (sf) {   // and it never crosses the edge: there it turns back, and from further out (a newly picked bird) it is brought in quickly
        const F = B[this.follow], lim = speed * dt * 2.5;
        if (F.x < sf.x0) { F.x = Math.min(sf.x0, F.x + lim); F.hx = Math.abs(F.hx); } else if (F.x > sf.x1) { F.x = Math.max(sf.x1, F.x - lim); F.hx = -Math.abs(F.hx); }
        if (F.y < sf.y0) { F.y = Math.min(sf.y0, F.y + lim); F.hy = Math.abs(F.hy); } else if (F.y > sf.y1) { F.y = Math.max(sf.y1, F.y - lim); F.hy = -Math.abs(F.hy); }
      }
      const p = Math.hypot(px, py) / n;
      this.pol += (p - this.pol) * Math.min(1, dt / .6);
      if (!isFinite(this.pol)) this.pol = p || 0;
    }
  }

  /* The big flock at the top and bottom of the page. Copied from murmuration-free-flight.html, then reworked (its own route, bursts, birds drifting along the body, ripples, splits of its own) and with one change (autoFalcon: the falcon can be told to stay away until
     someone clicks, so it never tears up a shape nobody asked it to). The overall shape is
     choreographed, and every bird runs separation, alignment and cohesion against its seven nearest neighbours */
function mountMurmuration(canvas, options) {
  const opt = Object.assign({
    sky: 'dusk',              // 'dusk', 'beige', 'none' (see-through, your page shows behind), or your own [[0, '#top'], [1, '#bottom']] stops
    ground: true,             // the dark treeline along the bottom
    groundColor: '#10151C',
    birdColor: '#0E1116',
    falconColor: '#b3341b',   // the one bird you should be able to follow: rust red, a colour nothing else on the page uses
    speed: 2,                 // playback rate; 1 is slower and calmer
    birds: 'auto',            // 'auto' = 12,000 on a big canvas, 3,800 on a small one; or a number
    falcon: true,             // a falcon that dives through the flock now and then
    autoFalcon: true,         // false: it stays out of sight until a click sends it, makes one pass, and leaves
    birdSize: 1,              // 1.3 draws every bird 30% bigger
    interactive: true,        // tap or click the sky to send the falcon there
    onCount: null             // called with the number of birds, at start and if a slow device thins the flock
  }, options);
  const SKIES = {
    dusk: {
      stops: [[0, '#5F8DB6'], [0.30, '#8FA6BC'], [0.52, '#C2B8B4'], [0.72, '#EBB58A'], [0.90, '#EE8E48'], [1, '#E27A38']],
      sun: ['rgba(255, 214, 150, 0.75)', 'rgba(255, 190, 120, 0.32)', 'rgba(255, 190, 120, 0)'],
      hills: 'rgba(58, 48, 54, 0.4)'
    },
    beige: {
      stops: [[0, '#EFE7D4'], [0.5, '#EBE1CB'], [0.8, '#E6D9BF'], [1, '#E0D0B2']],
      sun: ['rgba(255, 246, 225, 0.6)', 'rgba(255, 246, 225, 0.25)', 'rgba(255, 246, 225, 0)'],
      hills: 'rgba(96, 80, 56, 0.28)'
    }
  };
  const sky = Array.isArray(opt.sky) ? { stops: opt.sky, sun: null, hills: null } : (SKIES[opt.sky] || null);
  const ctx = canvas.getContext('2d', { alpha: !sky });
  const reduce = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  let running = true, visible = true, raf = 0;

  // ---------- world ----------
  const K = 7;                 // each bird tracks its 7 nearest neighbours, whatever the distance
  const C = 12;                // grid cell size
  const TABLE = 32768;         // hashed grid buckets
  const V0 = 72;               // cruise speed, world units per second
  const SPEED = opt.speed;      // playback rate
  const CAM = 1500;            // camera distance from the roost
  const STAGGER = 6;           // each bird refreshes its neighbour list every sixth frame

  let N = 0, frame = 0;
  let px, py, pz, vx, vy, vz, ax, ay, az, bank, fear, order, keyOf, nb, nbc;
  const cellStart = new Int32Array(TABLE + 1);
  const kd = new Float32Array(K);
  const kj = new Int32Array(K);
  let time = 31;               // start partway through the dance, in a good pose
  let comX = 0, comY = 0, comZ = 0;
  let camX = 0, camY = 0;      // the camera pans after the flock, lazily
  const FOLLOW = 0.8;          // 0 = fixed camera, 1 = flock always dead centre
  const tear = { uc: 0.5, t0: -100 };   // where and when the falcon last went through

  const falcon = { x: opt.falcon && opt.autoFalcon ? 420 : 1e6, y: 150, z: -120, vx: -60, vy: 0, vz: 40, mode: opt.autoFalcon ? 'loiter' : 'away', t: 0, until: 4, target: -1, kc: 48, wx: 380, wy: 150, wz: 0,
    pass: 0, side: 1, ang: 0, dir: 1, tuck: 0, flapA: 0, ph: 0, bank: 0, head: null, trail: [] };

  function alloc(n) {
    N = n;
    px = new Float32Array(n); py = new Float32Array(n); pz = new Float32Array(n);
    vx = new Float32Array(n); vy = new Float32Array(n); vz = new Float32Array(n);
    ax = new Float32Array(n); ay = new Float32Array(n); az = new Float32Array(n);
    bank = new Float32Array(n);
    fear = new Float32Array(n);
    order = new Int32Array(n);
    keyOf = new Int32Array(n);
    nb = new Int32Array(n * K);
    nbc = new Uint8Array(n);
  }

  function gauss() {
    return (Math.random() + Math.random() + Math.random() + Math.random() - 2) * 0.87;
  }

  // ---------- the ribbon ----------
  // The flock is steered as one long, flat, twisting ribbon that trails behind a point wandering
  // over the roost. Each bird owns a place along the ribbon (u: 0 = head, 1 = tail). When the
  // ribbon turns back on itself it folds, and where it folds or shows us its edge it goes dark.
  const M = 96;
  const rqx = new Float32Array(M), rqy = new Float32Array(M), rqz = new Float32Array(M);
  const rtx = new Float32Array(M), rty = new Float32Array(M), rtz = new Float32Array(M);
  const rax = new Float32Array(M), ray = new Float32Array(M), raz = new Float32Array(M);
  const rbx = new Float32Array(M), rby = new Float32Array(M), rbz = new Float32Array(M);
  const rwa = new Float32Array(M), rwb = new Float32Array(M), rsp = new Float32Array(M);
  let phaseU, homeU, seatPh, seatW, wv;
  const waves = [];   // dark ripples running down the body: { u0, t0 }
  let nextWave = 0, nextSplit = 0;

  function taper(u) {
    const s = Math.pow(Math.sin(Math.PI * u), 0.5);
    return s < 0.22 ? 0.22 : s;
  }

  function buildRibbon(t) {
    // seconds of trail: it draws out into a long ribbon, then gathers back into a cloud
    const T0 = 11.5 + 3 * Math.sin(0.2 * t + 0.4) + 2.4 * Math.sin(0.71 * t + 2.0);
    const T = T0 < 7.5 ? 7.5 : T0;
    const Tdot = T0 < 7.5 ? 0 : 3 * 0.2 * Math.cos(0.2 * t + 0.4) + 2.4 * 0.71 * Math.cos(0.71 * t + 2.0);
    const A0 = P.width * Math.sqrt(10 / T) * Math.sqrt(N / 12000);
    for (let k = 0; k < M; k++) {
      const u = k / (M - 1);
      const s = t - u * T;
      // where the leading point was, s seconds in
      const x = 240 * Math.sin(0.236 * s + 1.9) + 80 * Math.sin(0.531 * s + 2.6) + 28 * Math.sin(1.07 * s);
      const y = 62 * Math.sin(0.317 * s + 0.6) + 28 * Math.sin(0.613 * s + 1.8) - 14;
      const z = 135 * Math.sin(0.371 * s + 2.2) + 40 * Math.sin(0.147 * s + 1.0);
      let dx = 240 * 0.236 * Math.cos(0.236 * s + 1.9) + 80 * 0.531 * Math.cos(0.531 * s + 2.6) + 28 * 1.07 * Math.cos(1.07 * s);
      let dy = 62 * 0.317 * Math.cos(0.317 * s + 0.6) + 28 * 0.613 * Math.cos(0.613 * s + 1.8);
      let dz = 135 * 0.371 * Math.cos(0.371 * s + 2.2) + 40 * 0.147 * Math.cos(0.147 * s + 1.0);
      const sp = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
      dx /= sp; dy /= sp; dz /= sp;
      // a frame across the ribbon: side = tangent x up, then lift = side x tangent
      let bx = -dz, by = 0, bz = dx;
      const bl = Math.sqrt(bx * bx + bz * bz) || 1;
      bx /= bl; bz /= bl;
      const nx = by * dz - bz * dy, ny = bz * dx - bx * dz, nz = bx * dy - by * dx;
      // twist it
      const th = 1.35 * Math.sin(0.19 * t + 3.1 * u) + 0.7 * Math.sin(0.113 * t - 2.2 * u + 1.0);
      const ct = Math.cos(th), st = Math.sin(th);
      rqx[k] = x; rqy[k] = y; rqz[k] = z;
      rtx[k] = dx; rty[k] = dy; rtz[k] = dz;
      rax[k] = ct * bx + st * nx; ray[k] = ct * by + st * ny; raz[k] = ct * bz + st * nz;
      rbx[k] = -st * bx + ct * nx; rby[k] = -st * by + ct * ny; rbz[k] = -st * bz + ct * nz;
      // width swells and pinches along its length, and the swellings travel
      const bulge = 0.78 + 0.34 * Math.pow(Math.sin(Math.PI * (1.1 * u - 0.05 * t)), 2);
      const w = A0 * taper(u) * bulge;
      rwa[k] = w;
      rwb[k] = w * (0.2 + 0.06 * Math.sin(0.15 * t + 4.0 * u));
      const want = sp * (1 - u * Tdot);          // birds further back speed up or ease off as the trail changes length
      rsp[k] = want < 34 ? 34 : (want > 175 ? 175 : want);
    }
  }

  function seed() {
    phaseU = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      // more birds where the ribbon is wide
      let u = Math.random();
      for (let tries = 0; tries < 8; tries++) {
        const tp = taper(u);
        if (Math.random() < tp * tp) break;
        u = Math.random();
      }
      phaseU[i] = u;
    }
    homeU = Float32Array.from(phaseU); seatPh = new Float32Array(N); seatW = new Float32Array(N); wv = new Float32Array(N);
    for (let i = 0; i < N; i++) { seatPh[i] = Math.random() * 6.2832; seatW[i] = 0.25 + 0.4 * Math.random(); }
    buildRibbon(time);
    for (let i = 0; i < N; i++) {
      const k = Math.min(M - 1, Math.round(phaseU[i] * (M - 1)));
      const r = Math.sqrt(Math.random()), an = Math.random() * Math.PI * 2;
      const a = Math.cos(an) * r * rwa[k], b = Math.sin(an) * r * rwb[k];
      const c = gauss() * 6;
      px[i] = rqx[k] + a * rax[k] + b * rbx[k] + c * rtx[k];
      py[i] = rqy[k] + a * ray[k] + b * rby[k] + c * rty[k];
      pz[i] = rqz[k] + a * raz[k] + b * rbz[k] + c * rtz[k];
      vx[i] = rtx[k] * rsp[k] + gauss() * 3;
      vy[i] = rty[k] * rsp[k] + gauss() * 3;
      vz[i] = rtz[k] * rsp[k] + gauss() * 3;
      bank[i] = 0.2;
    }
  }

  // ---------- simulation ----------
  function bucket(gx, gy, gz) {
    return ((Math.imul(gx, 73856093) ^ Math.imul(gy, 19349663) ^ Math.imul(gz, 83492791)) >>> 0) & (TABLE - 1);
  }

  function buildGrid() {
    cellStart.fill(0);
    let sx = 0, sy = 0, sz = 0;
    for (let i = 0; i < N; i++) {
      const gx = Math.floor(px[i] / C) + 512, gy = Math.floor(py[i] / C) + 512, gz = Math.floor(pz[i] / C) + 512;
      keyOf[i] = gx | (gy << 10) | (gz << 20);
      cellStart[bucket(gx, gy, gz) + 1]++;
      sx += px[i]; sy += py[i]; sz += pz[i];
    }
    comX = sx / N; comY = sy / N; comZ = sz / N;
    for (let c = 0; c < TABLE; c++) cellStart[c + 1] += cellStart[c];
    for (let i = N - 1; i >= 0; i--) {
      const k = keyOf[i];
      const c = bucket(k & 1023, (k >> 10) & 1023, (k >> 20) & 1023);
      order[--cellStart[c + 1]] = i;
    }
    for (let c = 0; c < TABLE; c++) cellStart[c] = cellStart[c + 1];
    cellStart[TABLE] = N;
  }

  function findNeighbours(i) {
    const x = px[i], y = py[i], z = pz[i];
    const k0 = keyOf[i];
    const gx = k0 & 1023, gy = (k0 >> 10) & 1023, gz = (k0 >> 20) & 1023;
    let cnt = 0;
    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const cx = gx + dx, cy = gy + dy, cz = gz + dz;
          const key = cx | (cy << 10) | (cz << 20);
          const c = bucket(cx, cy, cz);
          let s = cellStart[c];
          const e = cellStart[c + 1];
          let seen = 0;
          for (; s < e; s++) {
            const j = order[s];
            if (keyOf[j] !== key || j === i) continue;
            if (++seen > 5) break;
            const ex = px[j] - x, ey = py[j] - y, ez = pz[j] - z;
            const d2 = ex * ex + ey * ey + ez * ez;
            if (cnt < K) {
              let p = cnt++;
              while (p > 0 && kd[p - 1] > d2) { kd[p] = kd[p - 1]; kj[p] = kj[p - 1]; p--; }
              kd[p] = d2; kj[p] = j;
            } else if (d2 < kd[K - 1]) {
              let p = K - 1;
              while (p > 0 && kd[p - 1] > d2) { kd[p] = kd[p - 1]; kj[p] = kj[p - 1]; p--; }
              kd[p] = d2; kj[p] = j;
            }
          }
        }
      }
    }
    const o = i * K;
    for (let k = 0; k < cnt; k++) nb[o + k] = kj[k];
    nbc[i] = cnt;
  }

  /* The falcon. Sent by a click, it stoops in from above the picture with its wings tucked, cuts through the flock,
     throws itself up into a climbing turn, circles once to look the flock over, stoops a second time, and then
     climbs away in a long curve. Every change of direction is a turn it has to fly (its pull is limited), so it
     never snaps round. (With autoFalcon it loiters near the flock instead of leaving.) */
  const FK = { stoopA: 430, climbA: 250, circleA: 200, leaveA: 170 };
  function seek(f, tx, ty, tz, speed, maxA, dt) {
    let dx = tx - f.x, dy = ty - f.y, dz = tz - f.z;
    const d = Math.hypot(dx, dy, dz) || 1;
    dx = dx / d * speed - f.vx; dy = dy / d * speed - f.vy; dz = dz / d * speed - f.vz;
    const m = Math.hypot(dx, dy, dz) || 1, k = Math.min(1, maxA * dt / m);
    f.vx += dx * k; f.vy += dy * k; f.vz += dz * k;
    return d;
  }
  function holdSpeed(f, speed, dt) {
    const v = Math.hypot(f.vx, f.vy, f.vz) || 1, r = 1 + (speed / v - 1) * Math.min(1, dt * 4);
    f.vx *= r; f.vy *= r; f.vz *= r;
  }
  function stepFalcon(dt) {
    const f = falcon;
    if (f.mode === 'away') return;
    f.t += dt;
    const viewX = W / (2 * S), viewY = H / (2 * S);
    let flap = 0, tuck = 0;
    if (f.mode === 'stoop' || f.mode === 'hunt') {   // the dive: faster and faster, wings folded back
      let tx, ty, tz;
      if (f.target >= 0 && f.target < N) { tx = px[f.target]; ty = py[f.target]; tz = pz[f.target]; }
      else { tx = rqx[f.kc]; ty = rqy[f.kc]; tz = rqz[f.kc]; }
      const sp = V0 * (2.3 + 1.3 * Math.min(1, f.t / .9));
      const d = seek(f, tx, ty, tz, sp, FK.stoopA, dt);
      tuck = Math.min(1, f.t / .5);
      if (d < 55 || f.t > 6) {
        f.mode = 'through'; f.t = 0; f.until = .45 + Math.random() * .25; f.pass++;
        if (d < 55) { tear.uc = f.target >= 0 && f.target < N ? phaseU[f.target] : f.kc / (M - 1); tear.t0 = time; }
      }
    } else if (f.mode === 'through') {   // carried straight on by its own speed, wings opening
      holdSpeed(f, V0 * 2.6, dt); tuck = Math.max(0, .7 - f.t * 2);
      if (f.t > f.until) {
        f.mode = 'climb'; f.t = 0;
        f.side = f.vx >= 0 ? 1 : -1;   // it pulls up the way it was already going
      }
    } else if (f.mode === 'climb') {   // pulls up and over into a turn
      const tx = f.x + f.side * 160, ty = Math.max(f.y, comY) + 260, tz = comZ;
      seek(f, tx, ty, tz, V0 * 1.9, FK.climbA, dt); flap = 1;
      if (f.t > 1.15) {
        if (f.pass >= 2 && !opt.autoFalcon) { f.mode = 'leave'; f.t = 0; f.side = f.vx >= 0 ? 1 : -1; }
        else { f.mode = 'circle'; f.t = 0; f.until = 1.8 + Math.random() * .8; f.ang = Math.atan2(f.y - comY, f.x - comX); f.dir = f.vx * (f.y - comY) - f.vy * (f.x - comX) > 0 ? -1 : 1; }
      }
    } else if (f.mode === 'circle') {   // a wide turn round the flock, gliding with a few beats
      f.ang += f.dir * dt * .95;
      const R = Math.min(330, viewX * .75), a = f.ang + f.dir * .7;
      seek(f, comX + Math.cos(a) * R, comY + 40 + Math.sin(a) * R * .55, comZ, V0 * 1.6, FK.circleA, dt);
      flap = Math.sin(f.t * 2.3) > .55 ? .8 : 0;
      if (f.t > f.until) { f.mode = 'stoop'; f.t = 0; f.target = -1; f.kc = Math.round((.3 + Math.random() * .4) * (M - 1)); }
    } else if (f.mode === 'leave') {   // one more climbing curve, and out of the picture
      seek(f, camX + f.side * (viewX + 500), camY + viewY + 420, comZ, V0 * 2.2, FK.leaveA, dt); flap = 1;
      if (Math.abs(f.x - camX) > viewX + 160 || f.y - camY > viewY + 140 || f.t > 7) { f.mode = 'away'; f.x = 1e6; f.trail.length = 0; return; }
    } else {   // autoFalcon: loiter near the flock, then stoop
      seek(f, f.wx, f.wy, f.wz, V0 * 1.4, 120, dt); flap = .6;
      if (f.t > f.until) { f.mode = 'stoop'; f.t = 0; f.target = -1; f.kc = Math.round((.3 + Math.random() * .4) * (M - 1)); f.pass = 0; }
    }
    if (f.mode === 'circle' && f.pass >= 2 && opt.autoFalcon) {   // the automatic one goes back to loitering instead of leaving
      f.mode = 'loiter'; f.t = 0; f.until = 2 + Math.random() * 3.5; f.pass = 0;
      const a = Math.random() * Math.PI * 2; f.wx = comX + Math.cos(a) * 380; f.wz = comZ + Math.sin(a) * 200; f.wy = comY + 90 + Math.random() * 110;
    }
    const e = Math.min(1, dt * 7);
    f.tuck += (tuck - f.tuck) * e; f.flapA += (flap - f.flapA) * Math.min(1, dt * 4);
    f.ph += dt * (13 + 4 * f.flapA);
    f.x += f.vx * dt; f.y += f.vy * dt; f.z += f.vz * dt;
    f.trail.push(f.x, f.y, f.z); if (f.trail.length > 66) f.trail.splice(0, 3);
  }

  // tunables
  const P = {
    rs: 14, ws: 140,            // separation radius and strength
    wa: 2.4,                   // alignment with the 7 neighbours
    wc: 1.0,                   // cohesion toward their centroid
    width: 205,                // half-width of the ribbon at its fullest
    wq: 1.5,                   // how firmly the ribbon's edge holds
    wl: 0.55,                  // how firmly a bird keeps its place along it
    wt: 0.55,                  // lean toward the ribbon's direction of travel
    wf: 9,                     // restless air
    jit: 7,                    // individual restlessness
    rp: 125, wp: 400,          // falcon: flee radius and strength
    gap: 150,                  // how far the two halves pull apart after a pass
    flow: 0.07,                // 1. how far each bird drifts along the ribbon from its own place, as a fraction of its length
    waveSpeed: 0.85,           // 2. how fast a ripple runs down the body (ribbon lengths per second)
    waveWidth: 0.035,          //    how wide the dark band is
    waveEvery: [4, 7],     //    seconds between ripples
    splitEvery: [14, 24],      // 4. seconds between splits the flock makes on its own
    amax: 125
  };



  function step(dt) {
    time += dt;
    frame++;
    buildGrid();
    const pan = frame < 3 ? 1 : 1 - Math.exp(-dt / 1.1);
    camX += (comX * FOLLOW - camX) * pan;
    camY += (comY * FOLLOW - camY) * pan;
    buildRibbon(time);
    if (opt.falcon) stepFalcon(dt);
    // 1. the inside flows: every bird drifts back and forth around its own place along the ribbon, each at its own pace,
    //    so birds keep streaming past one another without ever piling into one band
    for (let i = 0; i < N; i++) {
      let nu = homeU[i] + P.flow * Math.sin(seatW[i] * time + seatPh[i]);
      phaseU[i] = nu < 0.015 ? 0.015 : nu > 0.985 ? 0.985 : nu;
    }
    // 2. ripples: every few seconds one edge of the flock turns and the turn runs down the body as a dark band
    if (!nextWave) nextWave = time + 1.2;
    if (time > nextWave) { waves.push({ u0: Math.random() < 0.5 ? 0.04 + 0.2 * Math.random() : 0.76 + 0.2 * Math.random(), t0: time }); nextWave = time + P.waveEvery[0] + Math.random() * (P.waveEvery[1] - P.waveEvery[0]); }
    while (waves.length && time - waves[0].t0 > 2.6) waves.shift();
    for (let i = 0; i < N; i++) {
      let s = 0; const u = phaseU[i];
      for (const wq of waves) {
        const age = time - wq.t0, d = (Math.abs(u - wq.u0) - P.waveSpeed * age) / P.waveWidth;
        if (d > -3 && d < 3) s += Math.exp(-d * d) * Math.min(1, age * 6) * Math.max(0, 1 - age / 2.6);
      }
      wv[i] += ((s > 1 ? 1 : s) - wv[i]) * Math.min(1, dt * 14);
    }
    // 4. now and then it pinches in two by itself and closes up again, falcon or not
    if (!nextSplit) nextSplit = time + 7;
    if (time > nextSplit && (falcon.mode === 'away' || falcon.mode === 'loiter' || falcon.mode === 'circle')) { tear.uc = 0.32 + 0.36 * Math.random(); tear.t0 = time; nextSplit = time + P.splitEvery[0] + Math.random() * (P.splitEvery[1] - P.splitEvery[0]); }
    const phase = frame % STAGGER;
    if (frame < 3) { for (let i = 0; i < N; i++) findNeighbours(i); }
    else { for (let i = phase; i < N; i += STAGGER) findNeighbours(i); }
    const f = falcon;
    const t = time;
    const rs = P.rs, rs2 = rs * rs, rp2 = P.rp * P.rp;
    const fade = Math.exp(-dt / 1.6);   // fear wears off over a couple of seconds
    // the flock parts where the falcon went through, holds the gap, then closes it again
    const since = time - tear.t0;
    const gap = since < 0 ? 0 : Math.min(1, since / 0.45) * Math.exp(-Math.max(0, since - 1.6) / 1.9) * P.gap;
    const s1 = 0.23 * t, s2 = 1.3 - 0.17 * t, s3 = 0.19 * t + 0.7, s4 = 0.13 * t, s5 = 2.1 - 0.21 * t, s6 = 0.15 * t;
    for (let i = 0; i < N; i++) {
      const x = px[i], y = py[i], z = pz[i];
      const ux = vx[i], uy = vy[i], uz = vz[i];
      const sp = Math.sqrt(ux * ux + uy * uy + uz * uz) || 1e-4;
      let fx = 0, fy = 0, fz = 0;
      // ---- the 7 neighbours: don't collide, match heading, stay close ----
      const cnt = nbc[i];
      if (cnt > 0) {
        let mx = 0, my = 0, mz = 0, mvx = 0, mvy = 0, mvz = 0;
        const o = i * K;
        for (let k = 0; k < cnt; k++) {
          const j = nb[o + k];
          const ex = px[j] - x, ey = py[j] - y, ez = pz[j] - z;
          mx += ex; my += ey; mz += ez;
          mvx += vx[j]; mvy += vy[j]; mvz += vz[j];
          const d2 = ex * ex + ey * ey + ez * ez;
          if (d2 < rs2 && d2 > 1e-6) {
            const d = Math.sqrt(d2);
            const w = P.ws * (1 - d / rs) / d;
            fx -= ex * w; fy -= ey * w; fz -= ez * w;
          }
        }
        const inv = 1 / cnt;
        fx += mx * inv * P.wc + (mvx * inv - ux) * P.wa;
        fy += my * inv * P.wc + (mvy * inv - uy) * P.wa;
        fz += mz * inv * P.wc + (mvz * inv - uz) * P.wa;
      }
      // ---- the ribbon ----
      const fi = phaseU[i] * (M - 1);
      const k0 = fi | 0, k1 = k0 < M - 1 ? k0 + 1 : k0, fr = fi - k0;
      const ex = x - (rqx[k0] + (rqx[k1] - rqx[k0]) * fr);
      const ey = y - (rqy[k0] + (rqy[k1] - rqy[k0]) * fr);
      const ez = z - (rqz[k0] + (rqz[k1] - rqz[k0]) * fr);
      const tx = rtx[k0], ty = rty[k0], tz = rtz[k0];
      const a = ex * rax[k0] + ey * ray[k0] + ez * raz[k0];
      const b = ex * rbx[k0] + ey * rby[k0] + ez * rbz[k0];
      const c = ex * tx + ey * ty + ez * tz;
      const na = a / rwa[k0], nb2 = b / rwb[k0];
      const m2 = na * na + nb2 * nb2;
      const calm = 1 - 0.9 * fear[i];
      const du = phaseU[i] - tear.uc;
      const part = gap * (du < 0 ? 1 : -1) * Math.exp(-Math.abs(du) * 3.2);   // ahead of the cut surge on, behind it hang back
      if (m2 > 1) {
        const g = calm * P.wq * (1 - 1 / Math.sqrt(m2));
        fx -= g * (a * rax[k0] + b * rbx[k0]);
        fy -= g * (a * ray[k0] + b * rby[k0]);
        fz -= g * (a * raz[k0] + b * rbz[k0]);
      }
      const v0 = rsp[k0];
      fx += -P.wl * (calm * c - part) * tx + (tx * v0 - ux) * P.wt;
      fy += -P.wl * (calm * c - part) * ty + (ty * v0 - uy) * P.wt;
      fz += -P.wl * (calm * c - part) * tz + (tz * v0 - uz) * P.wt;
      // ---- restless air and a restless bird ----
      fx += (Math.sin(0.021 * y + s1) + Math.sin(0.017 * z + s2)) * P.wf + (Math.random() - 0.5) * P.jit;
      fy += (Math.sin(0.019 * z + s3) + Math.sin(0.023 * x + s4)) * P.wf * 0.7 + (Math.random() - 0.5) * P.jit;
      fz += (Math.sin(0.018 * x + s5) + Math.sin(0.022 * y + s6)) * P.wf + (Math.random() - 0.5) * P.jit;
      // ---- hold cruise speed ----
      const sc = 2.0 * (v0 - sp) / sp;
      fx += ux * sc; fy += uy * sc; fz += uz * sc;
      // ---- limit ----
      const m = Math.sqrt(fx * fx + fy * fy + fz * fz);
      if (m > P.amax) { const r = P.amax / m; fx *= r; fy *= r; fz *= r; }
      // ---- falcon: fear outranks everything, so it is added after the limit ----
      const qx = x - f.x, qy = y - f.y, qz = z - f.z;
      const q2 = qx * qx + qy * qy + qz * qz;
      if (q2 < rp2) {
        const q = Math.sqrt(q2) || 1e-3;
        const w = 1 - q / P.rp;
        const s = P.wp * w * w / q;
        fx += qx * s; fy += qy * s; fz += qz * s;
        if (w > fear[i]) fear[i] = w;
      }
      fear[i] *= fade;
      ax[i] = fx; ay[i] = fy; az[i] = fz;
      // how hard this bird is turning: used to flash its wings
      const al = (fx * ux + fy * uy + fz * uz) / sp;
      const lx = fx - al * ux / sp, ly = fy - al * uy / sp, lz = fz - al * uz / sp;
      let lat = Math.sqrt(lx * lx + ly * ly + lz * lz) / 70;
      if (lat > 1) lat = 1;
      bank[i] += (lat - bank[i]) * 0.12;
    }
    for (let i = 0; i < N; i++) {
      vx[i] += ax[i] * dt; vy[i] += ay[i] * dt; vz[i] += az[i] * dt;
      px[i] += vx[i] * dt; py[i] += vy[i] * dt; pz[i] += vz[i] * dt;
    }
  }

  // ---------- drawing ----------
  let W = 0, H = 0, DPR = 1, S = 1, CY = 0, RC = 1, RS = 0;
  const bg = document.createElement('canvas');

  function mulberry(a) {
    return () => {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function paintBackground() {
    bg.width = W; bg.height = H;      // also clears it
    const b = bg.getContext('2d');
    const horizon = H * 0.905;
    if (sky) {
      const g = b.createLinearGradient(0, 0, 0, H);
      for (const [at, colour] of sky.stops) g.addColorStop(at, colour);
      b.fillStyle = g;
      b.fillRect(0, 0, W, H);
      if (sky.sun) {
        // the last of the sun, low and off to one side
        const sx = W * 0.27, sy = H * 0.93;
        const sun = b.createRadialGradient(sx, sy, 0, sx, sy, Math.max(W, H) * 0.55);
        sun.addColorStop(0, sky.sun[0]);
        sun.addColorStop(0.25, sky.sun[1]);
        sun.addColorStop(1, sky.sun[2]);
        b.fillStyle = sun;
        b.fillRect(0, 0, W, H);
      }
      if (sky.hills && opt.ground) {
        // far hills
        b.fillStyle = sky.hills;
        b.beginPath();
        b.moveTo(0, H);
        for (let x = 0; x <= W; x += 6 * DPR) {
          const u = x / W;
          b.lineTo(x, horizon - H * 0.022 * (1 + Math.sin(u * 5.1 + 0.6) * 0.7 + Math.sin(u * 13.3) * 0.25));
        }
        b.lineTo(W, H);
        b.closePath();
        b.fill();
      }
    }
    if (opt.ground) {
      // near land: reedbed and a ragged treeline, the roost
      const rnd = mulberry(7);
      b.fillStyle = opt.groundColor;
      b.beginPath();
      b.moveTo(0, H);
      const stepX = 3 * DPR;
      for (let x = 0; x <= W + stepX; x += stepX) {
        const u = x / W;
        const clump = Math.max(0, Math.sin(u * 9.0 + 1.2) * Math.sin(u * 3.1 + 0.4));
        const tree = clump * (0.022 + 0.03 * rnd()) * H;
        const reeds = (0.004 + 0.006 * rnd()) * H;
        b.lineTo(x, horizon + H * 0.012 - tree - reeds);
      }
      b.lineTo(W, H);
      b.closePath();
      b.fill();
    }
  }

  function resize() {
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    if (!cw || !ch) return false;
    const dpr = Math.min(opt.maxDpr || 2, window.devicePixelRatio || 1);
    const w = Math.round(cw * dpr), h = Math.round(ch * dpr);
    if (w === W && h === H) return true;
    W = w; H = h; DPR = dpr;
    canvas.width = W; canvas.height = H;
    const tall = H > W * 1.15;      // on a tall screen the whole scene is tilted a little so the ribbon has room
    const tilt = tall ? 0.5 : 0;
    RC = Math.cos(tilt); RS = Math.sin(tilt);
    S = tall ? Math.min(W / 560, H / 800) : Math.min(W / 800, H / 500);
    CY = H * (tall ? 0.43 : 0.4);
    paintBackground();
    return true;
  }

  function draw() {
    ctx.globalAlpha = 1;
    if (!sky) ctx.clearRect(0, 0, W, H);
    if (sky || opt.ground) ctx.drawImage(bg, 0, 0);
    ctx.fillStyle = opt.birdColor;
    const F = CAM * S;
    const cx = W / 2;
    const base = 1.1 * DPR * Math.max(0.64, Math.min(1.25, S / DPR)) * opt.birdSize;
    for (let i = 0; i < N; i++) {
      const zc = pz[i] + CAM;
      if (zc < 120) continue;
      const k = F / zc;
      const rx = px[i] - camX, ry = py[i] - camY;
      const hx = rx * RC - ry * RS, hy = rx * RS + ry * RC;
      const sx = cx + hx * k;
      const sy = CY - hy * k;
      if (sx < -4 || sx > W + 4 || sy < -4 || sy > H + 4) continue;
      const sp = Math.sqrt(vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]) || 1;
      const wx = vx[i] * RC - vy[i] * RS, wy = vx[i] * RS + vy[i] * RC;
      const s = base * (CAM / zc) * (0.82 + 0.6 * bank[i]) * (1 + 0.55 * wv[i]);
      const w = s * (1 + 0.75 * Math.abs(wx) / sp);
      const h = s * (0.8 + 0.55 * Math.abs(wy) / sp);
      let a = 0.7 - pz[i] * 0.0006;
      if (a > 0.86) a = 0.86; else if (a < 0.4) a = 0.4;
      a += 0.22 * wv[i]; if (a > 1) a = 1;
      ctx.globalAlpha = a;
      ctx.fillRect(sx - w * 0.5, sy - h * 0.5, w, h);
    }
    // the falcon
    const f = falcon;
    if (opt.falcon && f.mode !== 'away' && f.z + CAM > 120) {
      const proj = (x, y, z) => { const k = F / (z + CAM), rx = x - camX, ry = y - camY; return [cx + (rx * RC - ry * RS) * k, CY - (rx * RS + ry * RC) * k]; };
      // a faint wake, so you can see the turns it just flew
      const T = f.trail;
      if (T.length > 6) {
        ctx.lineCap = 'round'; ctx.lineJoin = 'round';
        let [ax, ay] = proj(T[0], T[1], T[2]);
        for (let k = 3; k < T.length; k += 3) {
          const [bx, by] = proj(T[k], T[k + 1], T[k + 2]), u = k / T.length;
          ctx.strokeStyle = opt.falconColor; ctx.globalAlpha = .32 * u * u; ctx.lineWidth = (.6 + 1.6 * u) * DPR;
          ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke(); ax = bx; ay = by;
        }
      }
      const [sx, sy] = proj(f.x, f.y, f.z), zc = f.z + CAM;
      const sp = Math.hypot(f.vx, f.vy, f.vz) || 1;
      let hx = (f.vx * RC - f.vy * RS) / sp, hy = -(f.vx * RS + f.vy * RC) / sp;
      const hn = Math.hypot(hx, hy) || 1; hx /= hn; hy /= hn;
      // the drawn heading follows the real one smoothly, and how fast it is swinging round is how hard it banks
      const ang = Math.atan2(hy, hx);
      if (f.head === null) f.head = ang;
      let da = ang - f.head; da = Math.atan2(Math.sin(da), Math.cos(da));
      const dtd = 1 / 60, turn = da * 9;
      f.head += da * .35;
      f.bank += (Math.max(-1, Math.min(1, turn * .55)) - f.bank) * .12;
      const ux = Math.cos(f.head), uy = Math.sin(f.head);
      const r = 8.4 * DPR * Math.max(0.64, Math.min(1.25, S / DPR)) * (CAM / zc);
      const tk = f.tuck, beat = f.flapA * Math.sin(f.ph), open = 1 - .62 * tk;
      const at = (a, b) => [sx + (ux * a - uy * b) * r, sy + (uy * a + ux * b) * r];
      const half = (side) => {   // one side of the bird's outline, nose to tail, as points (the curves sampled)
        const bk = 1 + side * f.bank * .32 * (1 - tk);            // the outer wing of a turn looks longer, the inner one shorter
        const flapIn = 1 - .26 * f.flapA * (.5 - .5 * Math.cos(f.ph));
        const span = (1.95 - 1.5 * tk) * bk * flapIn;            // tucked for the dive, the wings fold flat along the body
        const tipX = -.48 - .58 * tk + .2 * beat, fan = .17 + .09 * Math.abs(f.bank) - .06 * tk;
        const out = [[1.24, .1], [1.04, .2], [.62, .25], [.38, .26]];   // a rounded head and a solid chest
        const quad = (c, e) => { const s0 = out[out.length - 1]; for (let q = 1; q <= 8; q++) { const u = q / 8, v = 1 - u; out.push([v * v * s0[0] + 2 * v * u * c[0] + u * u * e[0], v * v * s0[1] + 2 * v * u * c[1] + u * u * e[1]]); } };
        quad([.3 - .3 * tk, span * .8 + .05 * tk], [tipX, span]);                 // the long leading edge, swept back to a point
        quad([-.18 - .62 * tk, span * .32 + .12 * tk], [-.38 - .3 * tk, .21]);    // the trailing edge back to the body
        out.push([-.64, .13], [-1.12, fan], [-1.2, fan * .45], [-1.22, 0]);        // a narrow, square-ended tail that fans a little in a turn
        return out.map(([x, y]) => [x, y * side]);
      };
      const L = half(1), Rt = half(-1).reverse();
      ctx.beginPath(); ctx.moveTo(...at(1.32, 0));
      for (const [x, y] of L) ctx.lineTo(...at(x, y));
      for (const [x, y] of Rt) ctx.lineTo(...at(x, y));
      ctx.closePath();
      ctx.globalAlpha = 1; ctx.lineJoin = 'round';
      ctx.strokeStyle = 'rgba(254,254,254,.95)'; ctx.lineWidth = 2.2 * DPR; ctx.stroke();   // a white edge lifts it off the black of the flock
      ctx.fillStyle = opt.falconColor; ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  // tap: send the falcon at the bird nearest the tap
  function sendFalcon(best) {
    if (!opt.falcon || !(best >= 0 && best < N)) return;
    const f = falcon;
    if (f.mode === 'away') {   // it drops in from above the picture, on the side nearer the bird, already diving
      const vx = W / (2 * S), vy = H / (2 * S), side = px[best] - camX < 0 ? -1 : 1;
      f.x = camX + side * vx * (.55 + .3 * Math.random()); f.y = camY + vy + 90; f.z = pz[best];
      const dx = px[best] - f.x, dy = py[best] - f.y, d = Math.hypot(dx, dy) || 1, v = V0 * 2.3;
      f.vx = dx / d * v; f.vy = dy / d * v; f.vz = 0;
      f.trail.length = 0; f.head = null; f.tuck = 1; f.flapA = 0;
    }
    f.mode = 'stoop'; f.t = 0; f.target = best; f.pass = 0;
  }
  function onTap(e) {
    const r = canvas.getBoundingClientRect();
    if (!r.width) return;
    const tx = (e.clientX - r.left) / r.width * W;
    const ty = (e.clientY - r.top) / r.height * H;
    const F = CAM * S;
    let best = -1, bd = Infinity;
    for (let i = 0; i < N; i += 3) {
      const k = F / (pz[i] + CAM);
      const rx = px[i] - camX, ry = py[i] - camY;
      const dx = W / 2 + (rx * RC - ry * RS) * k - tx, dy = CY - (rx * RS + ry * RC) * k - ty;
      const d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    if (best >= 0) sendFalcon(best);
    if (reduce) { for (let i = 0; i < 90; i++) step(1 / 60); draw(); }
  }
  const tappable = opt.interactive && opt.falcon;
  if (tappable) canvas.addEventListener('pointerdown', onTap);

  // ---------- run ----------
  function start() {
    if (!running) return;
    if (!resize()) { raf = requestAnimationFrame(start); return; }   // wait until the canvas has a size
    const area = canvas.clientWidth * canvas.clientHeight;
    alloc(opt.birds === 'auto' ? (area < 420000 ? 3800 : 12000) : Math.max(500, opt.birds | 0));
    seed();
    if (opt.onCount) opt.onCount(N);
    for (let i = 0; i < 80; i++) step(1 / 30);   // let the seeded shape settle before anyone sees it
    draw();
    if (reduce) return;                          // visitors who ask for less motion get a still
    let last = performance.now();
    let slow = 0, frames = 0;
    const tick = (now) => {
      if (!running) return;
      raf = requestAnimationFrame(tick);
      if (document.hidden || !visible) { last = now; return; }   // no work while nobody can see it
      if (opt.maxFps && now - last < 1000 / opt.maxFps - 2) return;
      resize();
      let dt = (now - last) / 1000;
      last = now;
      if (dt > 1 / 30) dt = 1 / 30; else if (dt < 1 / 120) dt = 1 / 120;
      const t0 = performance.now();
      step(Math.min(dt * SPEED, 1 / 24));
      draw();
      const cost = performance.now() - t0;
      // if this device cannot keep up, thin the flock rather than stutter
      frames++;
      if (cost > 22) slow++;
      if (frames >= 60) {
        if (slow > 30 && N > 2500) {
          N = Math.round(N * 0.75);
          if (opt.onCount) opt.onCount(N);
        }
        frames = 0; slow = 0;
      }
    };
    raf = requestAnimationFrame(tick);
  }

  // pause while the canvas is scrolled out of view
  let watcher = null;
  if ('IntersectionObserver' in window) {
    watcher = new IntersectionObserver((entries) => { visible = entries[entries.length - 1].isIntersecting; });
    watcher.observe(canvas);
  }

  start();

  return {
    get birds() { return N; },
    // send the falcon at the body of the flock, as a click in the middle of it would
    send() { let best = -1, bd = 9; for (let i = 0; i < N; i += 5) { const d = Math.abs(phaseU[i] - .5); if (d < bd) { bd = d; best = i; } } sendFalcon(best); },
    // stop everything and let go of the canvas (call this if the page removes it)
    destroy() {
      running = false;
      cancelAnimationFrame(raf);
      if (watcher) watcher.disconnect();
      if (tappable) canvas.removeEventListener('pointerdown', onTap);
    }
  };
}

  const $ = (s) => document.querySelector(s), $$ = (s) => [...document.querySelectorAll(s)];
  const body = document.body;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const pct = (v) => Math.round(v * 100) + '%';
  const INK = '13,17,23';
  function mulberry(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  /* part 7: scroll position T runs along the chart's x axis. A narrow band on the left is step 0, the rest is a log axis from step 1 to the last step */
  const R7 = ['close', 'line', 'apart'];   // part 7's rule picker, in button order
  const STORY7 = {   // run 1, notes after round two, straight from the curves below
    close: 'The reader finds stay close at <strong>85% before any practice</strong>. The machine starts out using it at <strong>6%</strong> and builds up from there: above half from step 431 and above 80% from step 1,117, both inside the first of its 30 passes through the recordings. It ends at 99%.',
    line: 'Line up is <strong>the first thing the machine learns</strong>. It starts out using none of it, passes half by step 70 and 80% by step 558, long before the other two rules. The reader finds it at 68% before any practice and 80% from step 166. It ends at 100%.',
    apart: 'Keep your distance is <strong>the honest one</strong>. The reader finds only 31% of it before any practice, and reading it and using it grow up together: the reader passes half at step 470, the machine at step 724, and the machine passes 80% at step 1,448, just after the first pass. It ends at 98%.'
  };
  const HALF7 = { close: 431, line: 70, apart: 724 };   // where the machine's use of each rule passes half
  const USE7 = { close: '214,144,10', line: '42,107,242', apart: '226,61,109' };   // the machine's curve takes the rule's colour (amber one step darker, so a thin line holds on white)
  const curve7 = () => { const r = R7[state.rule7 || 0]; return r === 'close' ? DATA.practice : DATA.practiceAll[r]; };
  const PR = DATA.practice, BAND = .09, LN_TOTAL = Math.log(PR.total), fmtN = (n) => Math.round(n).toLocaleString('en-US');
  const stepX = (st) => st < 1 ? BAND / 2 : BAND + (1 - BAND) * Math.log(st) / LN_TOTAL;
  function practiceAt(T) {
    const C = curve7();
    if (T < BAND) return { step: 0, read: C.readable[0], use: C.used[0] };
    const st = Math.min(PR.total, Math.exp((T - BAND) / (1 - BAND) * LN_TOTAL)), S = PR.steps;
    let i = 1; while (i < S.length - 2 && S[i + 1] < st) i++;
    const u = clamp((Math.log(st) - Math.log(S[i])) / (Math.log(S[i + 1]) - Math.log(S[i])), 0, 1);
    return { step: st, read: C.readable[i] + (C.readable[i + 1] - C.readable[i]) * u, use: C.used[i] + (C.used[i + 1] - C.used[i]) * u };
  }

  /* each scene keeps its own switches, so the picture always matches the words beside it when you scroll back */
  const state = {
    scene: 0, page: 0, T: 0,
    on: { apart: true, line: false, close: true }, lastOff: 'line',   // line up starts off: the flock looks wrong, and that invites a press
    its: false, fifty: 0, raw: false, deaf7: 0, deaf9: 0, rule7: 0, split: 1, hold4: 0,
    toyed: false, autoLine: false, scrubbed: false
  };
  const stage = $('#stage'), skywin = $('#skywin'), padEl = $('#pad'), readerEl = $('#reader'), keyEl = $('#key'), capEl = $('#caption');
  const pages = [null, $('#p1'), $('#p2')];
  const togs = $$('.tog'), scenes = $$('.scene'), segs = $$('.segc[data-seg]'), says = $$('.say');
  const sayOn = (b) => {   // is the picture showing what this phrase says, right now?
    const [k, v] = b.dataset.say.split(':'), n = +v;
    if (+b.closest('.scene').dataset.scene !== state.scene) return false;
    return k === 'page' ? readingOn() && state.page === n : k === 'its' ? !!state.its : state[k] === n;
  };
  const main = new Flock(12), fifty = new Flock(50, { gap: 30, reach: 70, pace: 126 }), quad = [0, 1, 2, 3].map(() => new Flock(12));
  const toy = new Flock(120, { near: 86, inner: 11, gap: 24, reach: 42, pace: 100, err: 0, body: 17, jit: .12 });
  const flock = () => state.scene <= 2 ? toy : state.scene === 3 && state.fifty ? fifty : main;   // which flock the sky is showing

  /* ---------------- what the numbers say right now ---------------- */
  const deafNow = () => state.scene === 7 ? state.deaf7 : 0;
  const rawNow = () => state.scene === 6 && state.raw;
  const readingOn = () => !!state.page && state.scene >= 5 && state.scene <= 8;
  /* the reader's score is only measured on notes that deafening hasn't touched */
  const ringsOn = () => readingOn() && !(deafNow() && !(deafNow() === 2 && state.page === 1));
  function readable(rule) {
    if (!state.page) return null;
    if (state.scene === 8) return rule === R7[state.rule7] ? practiceAt(state.T).read : null;
    return (rawNow() ? DATA.unpractised : DATA.readable)[state.page][rule];
  }
  function left(run, deaf) {   // how much of each rule a run still has
    if (!deaf) return { apart: 1, line: 1, close: 1 };
    return DATA.keeps[run][deaf];
  }
  function aimFlocks() {
    const c = rawNow() ? 0 : state.scene === 8 ? Math.pow(state.T, 1.6) : 1;
    main.aim(left(0, deafNow()), c);
    quad.forEach((f, m) => {   // each pane flies as tightly as that run was measured to; with round two deaf there is no flock to hold
      f.aim(left(m, state.deaf9), 1); f.polGoal = state.deaf9 === 2 ? null : DATA.together[m][state.deaf9];
    });
    toy.aim({ apart: +state.on.apart, line: +state.on.line, close: +state.on.close }, 1);
  }

  /* ---------------- painting the DOM ---------------- */
  const geo = { padX: 0, padY: 0, gx: 0, gy: 0, top: 44, low: 1e4 };
  let readerSpot = '', chipKind = '', keySig = '', guessSig = '';
  const kick = (el, cls) => { el.classList.remove(cls); void el.offsetWidth; el.classList.add(cls); };   // play a one-shot animation again from the start
  function hopReader() { readerEl.classList.remove('hopping'); void readerEl.offsetWidth; readerEl.classList.add('hopping'); }
  function measure() {
    const st = stage.getBoundingClientRect(); if (!st.width) return;
    body.style.setProperty('--stageH', Math.round(stage.offsetHeight) + 'px');
    const sk = skywin.getBoundingClientRect(), pd = padEl.getBoundingClientRect();
    geo.padX = pd.right - sk.left - 6; geo.padY = pd.top - sk.top + 4;
    const onPage = readingOn();
    const b = (onPage ? pages[state.page] : skywin).getBoundingClientRect();
    const rw = readerEl.offsetWidth || 46, rh = readerEl.offsetHeight || 38;
    readerEl.style.setProperty('--rx', (b.left - st.left + b.width * (onPage ? .5 : .13) - rw / 2).toFixed(1) + 'px');
    readerEl.style.setProperty('--ry', ((onPage ? pd.top : b.top) - st.top - rh + 3).toFixed(1) + 'px');
    if (onPage) { geo.gx = b.left - sk.left + b.width * .5 + rw * .33; geo.gy = pd.top - sk.top - rh * .7 + 3; }   // the reader's beak, where its guesses are thrown from
    const spot = onPage ? 'p' + state.page : 'sky';
    if (spot !== readerSpot) { readerSpot = spot; hopReader(); }
  }
  /* the safe zone, part one: how far down the labels at the top of the sky reach, and how far up the ones at the bottom do
     (the notes with the reader perched on them, and the key) */
  function measureSafe() {
    const sk = skywin.getBoundingClientRect(); if (!sk.height) return;
    let top = 0, low = sk.height;
    for (const el of [$('.gauge'), $('#chip'), sk.width > 600 ? keyEl : null, padEl]) {
      if (!el) continue;
      const r = el.getBoundingClientRect(); if (!r.width) continue;
      if (r.top - sk.top < sk.height / 2) top = Math.max(top, r.bottom - sk.top);
      else low = Math.min(low, r.top - sk.top - (el === padEl && readingOn() ? readerEl.offsetHeight : 0));
    }
    geo.top = top; geo.low = low;
  }
  /* part two: where the ringed bird has to stay so that its ring, its arrows and the reader's rings are never cut by the
     frame or hidden under those labels. In part 2 there is only a ring to keep clear; everywhere else, the longest arrow */
  function setSafe(f) {
    const sc = state.scene;
    if (sc === 9 || !f.w) { f.safe = null; return; }
    const reach = sc === 3 ? (f === fifty ? Math.max(5.4, 7.6 * f.s) : Math.max(6.6, 9.2 * f.s)) * 1.75 + 8
      : 67 * view.s * (f === toy ? .8 : 1) + 6;
    const yTop = sc === 3 && state.its ? Math.max(geo.top, 96) : geo.top, yLow = Math.min(f.h, geo.low, view.h);
    const rx = Math.min(reach, f.w * .3), ry = Math.min(reach, Math.max(0, yLow - yTop) * .3);   // on a small picture the arrows may brush a label, but the bird always has room to fly
    let x0 = rx, x1 = f.w - rx, y0 = yTop + ry, y1 = yLow - ry;
    if (y0 > y1) y0 = y1 = (y0 + y1) / 2;
    f.safe = { x0, y0, x1, y1 };
  }
  /* the key inside the sky: what each colour is, what the reader reads, what the machine keeps */
  const keyCells = {};
  (function buildKey() {
    const arrow = '<svg viewBox="0 0 22 10" aria-hidden="true"><path d="M1 5h14" stroke="var(--c)" stroke-width="2.4" stroke-linecap="round"/><path d="M21 5l-7-4v8Z" fill="var(--c)"/></svg>';
    keyEl.innerHTML = '<span class="kh"></span><span class="kh kn"></span><span class="kh k1" id="kh1">reads</span><span class="kh k2" id="kh2">keeps</span>'
      + RULES.map((r) => '<span data-rule="' + r + '">' + arrow + '</span><span class="kn" data-rule="' + r + '">' + NAME[r] + '</span><span class="kv k1" data-rule="' + r + '"></span><span class="kv k2" data-rule="' + r + '"></span>').join('');
    for (const r of RULES) { const c = [...keyEl.querySelectorAll('[data-rule="' + r + '"]')]; keyCells[r] = { all: c, v1: c[2], v2: c[3] }; }
  })();
  function paintKey() {
    const s = state.scene, d = deafNow(), raw = rawNow() && !!state.page, c1 = ringsOn(), c2 = (s === 7 && d > 0) || s === 8 || raw;
    keyEl.classList.toggle('c1', c1); keyEl.classList.toggle('c2', c2); keyEl.classList.toggle('cmp', raw);
    const small = skywin.clientWidth <= 420;
    $('#kh1').textContent = raw ? (small ? 'trained' : 'practised') : 'reads';
    $('#kh2').textContent = s === 8 ? 'uses' : raw ? (small ? 'never' : 'never practised') : 'keeps';
    for (const r of RULES) {
      const K = keyCells[r], hide = s === 8 && r !== R7[state.rule7];
      K.all.forEach((el) => el.classList.toggle('off', hide));
      const R = raw ? DATA.readable[state.page][r] : readable(r); K.v1.textContent = c1 && R !== null ? pct(R) : '';
      let v = null;
      if (raw) v = DATA.unpractised[state.page][r];
      else if (s === 7 && d) v = clamp(DATA.keeps[0][d][r], 0, 1); else if (s === 8 && r === R7[state.rule7]) v = Math.max(0, practiceAt(state.T).use);
      K.v2.textContent = v === null ? '' : pct(v);
      K.v2.classList.toggle('gone', s === 7 && v !== null && v < .1);
      K.v2.classList.toggle('hit', raw && v >= R);
    }
    const ks = [s, d, raw, c1, c2, state.page].join();   // when what the key is saying changes, it says so
    if (ks !== keySig) { keySig = ks; kick(keyEl, 'flash'); }
  }
  const CAP4 = ['The ringed bird is about to move.', 'Round one. Every bird hears where the others are, and the ringed bird writes its first page.',
    'Round two. Each bird passes on its first page, which already holds what it heard, and the ringed bird writes its second page.', 'Now it moves, using only its two pages.'];
  let capKind = '';
  function paintCaption() {
    paintCaptionText();
    const kind = state.scene === 8 ? 'step' + state.rule7 : capEl.textContent;   // the step counter ticks on its own; a new sentence announces itself
    if (kind !== capKind) { if (capKind) kick(capEl, 'fresh'); capKind = kind; }
  }
  function paintCaptionText() {
    const s = state.scene, d = deafNow(), off = RULES.filter((r) => !state.on[r]);
    capEl.textContent = s <= 2 ? (off.length ? DATA.toy[off.includes(state.lastOff) ? state.lastOff : off[0]] : 'All three rules are on. Each arrow shows one rule pulling on the ringed bird.')
      : s === 3 ? (!state.its ? 'This is the machine flying ' + (state.fifty ? 'fifty birds, a flock size it never saw.' : 'twelve birds.')
          : state.fifty ? 'Fifty birds, a size it never practised on. It flies them as one flock, about as tightly as the simulator. I only scored its guesses on twelve, so there are no rings here.'
          : 'Left is what you see. Right is what the machine gets. A dot is a bird, a ring is where the machine guesses that bird will be next, and then the bird flies into it.')
      : s === 4 ? CAP4[beat.phase || 0]
      : s === 5 ? (!state.page ? 'The reader is waiting on the edge of the sky.'
          : 'The reader is on the ' + (state.page === 1 ? 'first' : 'second') + ' page of notes. A teal ring that lands on the tip of an arrow is a good guess.')
      : s === 6 ? (state.raw ? 'This machine never practised. The dashed arrows are what the rules ask for, and it does none of it. The reader still scores, and on stay close in this run it scores higher.'
          : 'This is the practised machine, and the reader\u2019s rings land on its arrows.')
      : s === 7 ? (d === 2 ? 'Round two is deaf. Dashed arrows are what each rule asks for, and solid arrows are what the machine still does. The solid ones have almost vanished, and the reader’s rings still land on the dashed ones.'
          : d === 1 ? 'Round one is deaf. Most of each solid arrow is still there.'
          : 'Both rounds are listening. The solid arrows are the machine doing what each rule asks.')
      : s === 8 ? 'Step ' + fmtN(practiceAt(state.T).step) + ' of ' + fmtN(PR.total) + '. The ring is the reader finding ' + NAME[R7[state.rule7]] + ', and the solid arrow is how much the machine uses it.'
      : (state.deaf9 === 1 ? 'Round one is deaf in all four runs. They still fly as flocks, only looser. ' : state.deaf9 === 2 ? 'Round two is deaf in all four runs, and none of them flies as a flock. ' : 'Four training runs of the same machine. ')
          + 'The number on each run is how tightly it flew when I measured it, where 1 is perfectly together.';
  }
  function paintChip() {
    const s = state.scene, d = deafNow(), el = $('#chip');
    const text = s <= 2 ? 'my simulator'
      : s === 3 ? (state.its ? 'what the machine gets' : 'the machine, ' + (state.fifty ? 50 : 12) + ' birds')
      : d ? 'round ' + (d === 1 ? 'one' : 'two') + ' is deaf'
      : rawNow() ? 'never practised'
      : s === 8 ? 'step ' + fmtN(practiceAt(state.T).step) + ' of ' + fmtN(PR.total)
      : 'the machine';
    if (el.textContent !== text) el.textContent = text;
    const kind = s === 8 ? 'step' : text;   // the step counter ticks on its own; everything else announces itself
    if (kind !== chipKind) { chipKind = kind; kick(el, 'pop'); }
  }
  let story7At = -1;
  function paintPrac() {
    const r7 = R7[state.rule7];
    if (story7At !== state.rule7) {
      const el = $('#payoff7'), nm = $('#rule7name'); el.innerHTML = STORY7[r7]; el.querySelectorAll('strong').forEach((x) => x.classList.add('lit'));
      nm.textContent = NAME[r7]; nm.className = 'rule ' + r7;
      if (story7At >= 0) { kick(el, 'fresh'); kick(nm, 'fresh'); }
      story7At = state.rule7;
    }
    $('#prac').style.setProperty('--use7', 'rgb(' + USE7[r7] + ')');
    const P = practiceAt(state.T);
    $('#vRead').textContent = pct(P.read); $('#vUse').textContent = pct(Math.max(0, P.use));
    drawChart7();
  }
  /* the part 7 chart: two measured curves for stay close in run 1. What is behind the marker is solid, what is ahead is faint.
     Colours follow the page: teal is the reader; amber is stay close, one step darker here so a thin line holds on white */
  const chart7 = $('#chart7'), cv7 = chart7.querySelector('canvas'), c7 = cv7.getContext('2d'), tip7 = $('#tip7');
  let hover7 = -1;
  function drawChart7() {
    const w = cv7.clientWidth, h = cv7.clientHeight; if (!w || !h) return;
    const dpr = Math.min(2.5, devicePixelRatio || 1);
    if (cv7.width !== Math.round(w * dpr) || cv7.height !== Math.round(h * dpr)) { cv7.width = Math.round(w * dpr); cv7.height = Math.round(h * dpr); }
    const c = c7, L = 46, R = 10, T0 = 30, B = 40, pw = w - L - R, ph = h - T0 - B, X = (f) => L + f * pw, Y = (v) => T0 + (1 - clamp(v, 0, 1)) * ph;
    c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
    c.lineCap = 'round'; c.lineJoin = 'round';
    /* grid and axes: recessive hairlines */
    c.strokeStyle = '#e1e5ea'; c.lineWidth = 1; c.beginPath();
    for (const v of [0, .5, 1]) { c.moveTo(L, Math.round(Y(v)) + .5); c.lineTo(w - R, Math.round(Y(v)) + .5); }
    c.stroke();
    c.font = '500 12.5px "DM Mono", ui-monospace, Menlo, monospace'; c.fillStyle = '#39414c'; c.textBaseline = 'middle'; c.textAlign = 'right';
    for (const [v, t] of [[0, '0%'], [.5, '50%'], [1, '100%']]) c.fillText(t, L - 8, Y(v) + .5);
    c.textBaseline = 'alphabetic'; c.textAlign = 'center';
    c.strokeStyle = '#b9c0c8'; c.beginPath();
    for (const st of [0, 1, 10, 100, 1000, 10000]) { const x = Math.round(X(stepX(st))) + .5; c.moveTo(x, T0 + ph); c.lineTo(x, T0 + ph + 5); c.fillText(st ? fmtN(st) : '0', x, T0 + ph + 18); }
    c.stroke();
    c.font = '600 13px "Bricolage Grotesque", Arial, sans-serif'; c.fillStyle = 'rgb(' + INK + ')'; c.textAlign = 'right';
    c.fillText(pw > 300 ? 'training step, on a log scale' : 'training step (log scale)', w - R, h - 3);
    /* the end of the first pass through the recordings */
    const xp = Math.round(X(stepX(PR.pass1))) + .5;
    c.strokeStyle = 'rgb(' + INK + ')'; c.lineWidth = 1; c.beginPath(); c.moveTo(xp, T0 - 5); c.lineTo(xp, T0 + ph); c.stroke();
    { const t = 'end of the first pass, of 30', tw = c.measureText(t).width; c.textAlign = 'right'; c.fillText(t, Math.min(w - R, xp + tw / 2), T0 - 10); }
    { const r7 = R7[state.rule7], hs = HALF7[r7], xh = X(stepX(hs)), yh = Y(.5), col = 'rgb(' + USE7[r7] + ')';   // where the machine passes half of this rule
      c.save(); c.setLineDash([3, 4]); c.strokeStyle = col; c.globalAlpha = .7; c.beginPath(); c.moveTo(xh, yh); c.lineTo(xh, T0 + ph); c.stroke(); c.restore();
      c.font = '600 12.5px "Bricolage Grotesque", Arial, sans-serif'; c.fillStyle = col; c.textAlign = 'left'; c.textBaseline = 'middle';
      const lbl = 'half used, step ' + fmtN(hs), lw = c.measureText(lbl).width, lx = Math.min(xh + 8, w - R - lw);
      c.fillStyle = 'rgba(254,254,254,.9)'; c.fillRect(lx - 3, yh + 12, lw + 6, 18); c.fillStyle = col; c.fillText(lbl, lx, yh + 21); c.textBaseline = 'alphabetic'; }
    /* the two curves */
    const xm = X(state.T), S = PR.steps;
    const C7 = curve7(), useCol = USE7[R7[state.rule7]];
    for (const [arr, col] of [[C7.readable, '25,167,155'], [C7.used, useCol]]) {
      for (const pass of [0, 1]) {   // faint all the way, then solid up to the marker
        c.save(); if (pass) { c.beginPath(); c.rect(0, 0, xm, h); c.clip(); }
        c.strokeStyle = 'rgba(' + col + ',' + (pass ? 1 : .28) + ')'; c.lineWidth = 2.6; c.beginPath();
        for (let i = 1; i < S.length; i++) { const x = X(stepX(S[i])), y = Y(arr[i]); if (i === 1) c.moveTo(x, y); else c.lineTo(x, y); }
        c.stroke();
        c.fillStyle = 'rgba(' + col + ',' + (pass ? 1 : .28) + ')'; c.beginPath(); c.arc(X(stepX(0)), Y(arr[0]), 3.2, 0, 6.2832); c.fill();   // step 0 sits apart from the log axis
        c.restore();
      }
    }
    /* the marker scroll moves, and a second one under the pointer */
    const mark = (f, strong) => {
      const P = practiceAt(f), x = X(f);
      c.strokeStyle = strong ? 'rgb(' + INK + ')' : 'rgba(' + INK + ',.45)'; c.lineWidth = strong ? 1.5 : 1; c.beginPath(); c.moveTo(x, T0 - 3); c.lineTo(x, T0 + ph); c.stroke();
      for (const [v, col] of [[P.read, '#19a79b'], [P.use, 'rgb(' + useCol + ')']]) {
        c.fillStyle = '#fefefe'; c.beginPath(); c.arc(x, Y(v), strong ? 6.5 : 5.5, 0, 6.2832); c.fill();
        c.fillStyle = col; c.beginPath(); c.arc(x, Y(v), strong ? 4.5 : 3.5, 0, 6.2832); c.fill();
      }
      return P;
    };
    if (hover7 >= 0) {
      const P = mark(hover7, false), x = X(hover7);
      tip7.replaceChildren(); const b = document.createElement('b'); b.textContent = 'step ' + fmtN(P.step); tip7.append(b, '  finds ' + pct(P.read) + '  uses ' + pct(Math.max(0, P.use)));
      tip7.hidden = false; tip7.style.left = clamp(x - tip7.offsetWidth / 2, 0, w - tip7.offsetWidth) + 'px';
    } else tip7.hidden = true;
    mark(state.T, true);
  }
  const at7 = (e) => { const r = cv7.getBoundingClientRect(); hover7 = clamp((e.clientX - r.left - 46) / (r.width - 56), 0, 1); drawChart7(); };
  /* a mouse reads wherever it hovers. A finger reads while it is down and for a moment after it lifts, and the
     reading goes as soon as the finger starts scrolling, so it never sits on the chart while you scroll through practice */
  let tip7Timer = 0, tip7Touch = false;
  const clear7 = () => { clearTimeout(tip7Timer); if (hover7 >= 0) { hover7 = -1; drawChart7(); } };
  cv7.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse' || e.buttons) at7(e); });
  cv7.addEventListener('pointerdown', (e) => { tip7Touch = e.pointerType !== 'mouse'; clearTimeout(tip7Timer); at7(e); });
  cv7.addEventListener('pointerup', (e) => { if (e.pointerType !== 'mouse') { clearTimeout(tip7Timer); tip7Timer = setTimeout(clear7, 1600); } });
  cv7.addEventListener('pointercancel', clear7);   // the browser took the finger for a scroll
  cv7.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') clear7(); });
  document.addEventListener('pointerdown', (e) => { if (e.target !== cv7) clear7(); });
  addEventListener('scroll', () => { if (tip7Touch && hover7 >= 0) clear7(); }, { passive: true });
  new ResizeObserver(drawChart7).observe(cv7);
  function paint() {
    const s = state.scene, d = deafNow();
    body.classList.toggle('scrubbed', state.scrubbed); body.classList.toggle('toyed', state.toyed);
    body.dataset.scene = s;
    body.classList.toggle('s4to8', s >= 4 && s <= 8);
    body.classList.toggle('reading', readingOn());
    body.classList.toggle('split', s === 3 && state.its);
    for (const n of [1, 2]) { pages[n].setAttribute('aria-pressed', readingOn() && state.page === n); pages[n].classList.toggle('deaf', d === n); pages[n].classList.toggle('writing', s === 4 && beat.phase === n); }
    for (const g of segs) for (const b of g.querySelectorAll('button')) b.setAttribute('aria-pressed', +b.dataset.v === state[g.dataset.seg]);
    togs.forEach((b) => b.setAttribute('aria-pressed', state.on[b.dataset.rule]));
    for (const b of says) b.setAttribute('aria-pressed', sayOn(b));
    for (const sc of scenes) sc.classList.toggle('ask', +sc.dataset.scene === s && !rest.touched.has(s) && !(s === 2 && state.toyed));   // until someone presses something in this part, its controls ask to be pressed
    aimFlocks(); measure(); paintKey(); paintCaption(); paintChip(); paintPrac(); paintPanes(); measureSafe();
    const gs = [s, state.page, +rawNow(), d, +ringsOn(), s === 8 ? state.rule7 : 0].join();
    if (gs !== guessSig) { guessSig = gs; throwGuesses(performance.now() / 1000, true); }
  }

  /* ---------------- actions ---------------- */
  function setPage(n) { state.page = n; paint(); }
  function setRule(r, v) { state.on[r] = v; if (!v) state.lastOff = r; state.toyed = true; state.autoLine = false; startle(); paint(); }
  const replay7 = { on: false, t0: 0, dur: 2 };
  function setT(v) { state.Ts = clamp(v, 0, 1); applyT(); }
  function applyT() {
    let v = state.Ts || 0;
    if (replay7.on) { const u = clamp((performance.now() - replay7.t0) / (replay7.dur * 1000), 0, 1); v *= u * u * (3 - 2 * u); if (u >= 1) replay7.on = false; }
    if (Math.abs(v - state.T) < .0005) return;
    state.T = v;
    const past = (state.Ts || 0) >= stepX(PR.half); if (past !== state.scrubbed) { state.scrubbed = past; body.classList.toggle('scrubbed', past); if (past) kick($('.scrub .payoff'), 'fresh'); }
    aimFlocks(); paintKey(); paintCaption(); paintChip(); paintPrac();
  }

  /* ---------------- scrolling is the remote control ----------------
     A paragraph marked data-beat changes the picture when it scrolls up to the reading line, and changes it back if
     you scroll up again. The rule switches and the two "try it yourself" controls are optional extras: the story
     plays the same whether or not anyone touches them. */
  const beatEls = $$('[data-beat]'), scene8 = scenes.find((el) => el.dataset.scene === '8'), pin8 = scene8.querySelector('.pin');
  const anchors = scenes.map((el) => ({ s: +el.dataset.scene, el: el.querySelector('.slate') }));
  const isWide = () => matchMedia('(min-width:900px)').matches;
  function readingLine() {   // where on the screen a paragraph "arrives"
    const vh = innerHeight;
    if (isWide()) return Math.min(vh * .6, 520);
    const sb = stage.getBoundingClientRect().bottom; return sb + (vh - sb) * .4;
  }
  /* Most people scroll to a part, stop, read all of it, and only then move on. A beat that waits for its paragraph to
     reach the reading line would then fire on the way out, where nobody is looking. Two things cover that. In part 1,
     where the words promise it, the beat plays by itself once the page has been still for a reading's worth of time
     (rest.forced). Everywhere else the phrase that changes the picture is a button in the sentence, and it pulses,
     along with the part's other controls, until one of them has been pressed. */
  const rest = { forced: new Set(), touched: new Set(), since: 0, moving: 0, y: scrollY, peak: scrollY };
  /* a part someone has pressed something in belongs to them: the scroll no longer changes the picture there until they leave it */
  const took = new Set();
  function touch(s) { rest.touched.add(s); took.add(s); }
  let beatSig = '';
  function applyScroll(entered, line) {
    const s = state.scene, B = {};
    const bl = isWide() ? line : (() => { const sb = stage.getBoundingClientRect().bottom; return sb + (innerHeight - sb) * .72; })();
    for (const el of beatEls) B[el.dataset.beat] = el.getBoundingClientRect().top <= bl || rest.forced.has(el.dataset.beat);
    const sig = s + JSON.stringify(B);
    if (sig !== beatSig || entered) {
      beatSig = sig;
      const was = state.fifty, d7 = state.deaf7;
      if (s === 2) {   // if nobody has touched the switches, show what losing "line up" does
        if (B.toyline && !state.toyed && state.on.line) { state.on.line = false; state.lastOff = 'line'; state.autoLine = true; startle(); }
        else if (!B.toyline && state.autoLine) { state.on.line = true; state.autoLine = false; }
      }
      const mine = took.has(s) && !entered;
      if (s === 3 && !mine) { state.its = B.its; state.fifty = B.fifty ? 1 : 0; }
      if (s === 5 && !mine) state.page = B.read2 ? 2 : 1;
      if (s === 6) { state.raw = true; if (!mine) state.page = 1; }
      if (s === 7 && (B.d72 || B.d71)) unveil7(true);   // part 6 only changes when you press something; the scroll just lifts the blur
      if (s === 9 && !mine) state.deaf9 = B.d92 ? 2 : B.d91 ? 1 : 0;
      paint(); if (was !== state.fifty) sizeSky();
    }
    if (s === 8) {   // practice is scrubbed by the scroll itself, while the text stays pinned
      const r = scene8.getBoundingClientRect();
      if (isWide()) { const travel = r.height - pin8.offsetHeight - parseFloat(getComputedStyle(scene8).paddingBottom); setT((-r.top - 30) / Math.max(1, travel - 190)); }
      else { const pr = pin8.querySelector('.prac'), sb = stage.getBoundingClientRect().bottom; setT(1 - (pin8.getBoundingClientRect().bottom - sb - pr.offsetHeight - 16) / (innerHeight * 1.1)); }   // narrow: 0 when the chart pins under the picture, 1 when the part lets go of it
    }
  }

  /* ---------------- scenes ---------------- */
  function enterScene(s) {
    const from = state.scene; state.scene = s;
    rest.forced.clear(); rest.since = performance.now(); took.clear();
    if (s >= 5 && s <= 8 && !state.page) state.page = 1;
    main.gap = s === 4 ? 52 : 36;   // in part 3 the birds keep a little further apart, so the pages they pass can be told apart
    if (s === 4 && from !== 4) beat.origin = performance.now() / 1000 - .05;   // the listening starts now, not whenever the clock comes round
    if (s === 7 && from !== 7) state.page = 1;
    if (s === 8) state.page = 2;
    paint(); sizeSky();
  }
  let ticking = false;
  function pickScene() {
    ticking = false;
    const line = readingLine();
    let s = 0;
    for (const a of anchors) if (a.el.getBoundingClientRect().top <= line) s = a.s;
    const entered = s !== state.scene;
    if (entered) enterScene(s);
    applyScroll(entered, line);
    /* how far through this part the reading line is: the sky's hour follows it */
    const i = anchors.findIndex((a) => a.s === s);
    if (i < 0) todTo = 0;
    else {
      const a = anchors[i].el.getBoundingClientRect().top, b = i + 1 < anchors.length ? anchors[i + 1].el.getBoundingClientRect().top : filmEl.getBoundingClientRect().bottom;
      todTo = TOD[s] + (TOD[s + 1] - TOD[s]) * clamp((line - a) / Math.max(1, b - a), 0, 1);
    }
    paintRail(s);
    /* scrolling leans on the flock like a gust of wind */
    const now = performance.now(), y = scrollY, v = (y - gust.y) / Math.max(16, now - gust.at) * 1000; gust.y = y; gust.at = now;
    gust.v = clamp(v / 1400, -1, 1);
  }
  const railEl = $('#rail'), railBird = $('#railBird'), railBtns = anchors.map((a, i) => {
    const b = document.createElement('button'), t = a.el.textContent.replace(/\s*\/\s*/, ': ').trim(), m = t.match(/^Part (\d+): (.*)$/);
    b.type = 'button'; b.setAttribute('aria-label', 'Go to ' + t);
    const lb = document.createElement('span'); lb.className = 'rl'; lb.setAttribute('aria-hidden', 'true');
    if (m) { const k = document.createElement('i'), nm = document.createElement('span'); k.textContent = m[1]; nm.className = 'rn'; nm.textContent = m[2]; lb.append(k, nm); } else lb.textContent = t;
    b.appendChild(lb);
    b.addEventListener('click', () => { const sk = isWide() ? skywin.getBoundingClientRect().top : 24; scrollTo({ top: a.el.getBoundingClientRect().top + scrollY - Math.max(24, Math.min(sk, readingLine() - 80)), behavior: reduce ? 'auto' : 'smooth' }); });
    railEl.appendChild(b); return b;
  });
  let railAt = -2;
  function paintRail(s) {
    const i = anchors.findIndex((a) => a.s === s), fr = filmEl.getBoundingClientRect();
    const on = i >= 0 && fr.bottom > innerHeight * .55;
    railEl.classList.toggle('on', on);
    if (i === railAt || i < 0) return; railAt = i;
    railBtns.forEach((b, j) => { b.classList.toggle('done', j <= i); b.classList.toggle('cur', j === i); });
    railEl.style.setProperty('--by', (railBtns[i].offsetTop + railBtns[i].offsetHeight / 2).toFixed(1) + 'px');   // the bird and its tag both read this
    $('#railNow').textContent = 'Part ' + (i + 1);
  }
  const gust = { y: scrollY, at: performance.now(), v: 0, now: 0, slow: 0 }, filmEl = $('#film');
  function onScroll() {
    const y = scrollY;
    if (Math.abs(y - rest.y) > 2) {
      rest.y = y; rest.moving = performance.now();
      if (y > rest.peak) rest.peak = y;
      else if (y < rest.peak - 40) { rest.peak = y; if (rest.forced.size) { rest.forced.clear(); beatSig = ''; } }   // scrolling back up rewinds the picture, as it always did
    }
    if (!ticking) { ticking = true; requestAnimationFrame(pickScene); }
  }
  const WPS = 3.4;   // words a second: a little under the usual reading pace, so the picture changes just as the words get there
  function wordsBetween(a, b) { const r = document.createRange(); r.setStartBefore(a); r.setEndBefore(b); return r.toString().trim().split(/\s+/).filter(Boolean).length; }
  function restNext() {   // the next beat of this part that has not played yet, if its paragraph is on screen
    if (state.scene !== 2 || rest.touched.has(2)) return null;
    const sc = scenes.find((el) => +el.dataset.scene === state.scene); if (!sc) return null;
    const line = readingLine(); let prev = sc.querySelector('h2') || sc;
    for (const el of sc.querySelectorAll('[data-beat]')) {
      const r = el.getBoundingClientRect();
      if (!(r.top <= line || rest.forced.has(el.dataset.beat))) return r.top < innerHeight - 30 && r.bottom > 0 ? { el, prev } : null;
      prev = el;
    }
    return null;
  }
  setInterval(() => {
    const now = performance.now();
    if (document.hidden || now - rest.moving < 900) return;   // still scrolling: the scroll is in charge
    const nx = restNext(); if (!nx) return;
    const wait = clamp(wordsBetween(nx.prev, nx.el) / WPS, 3, 20) * 1000;
    if (now - Math.max(rest.since, rest.moving) < wait) return;
    rest.forced.add(nx.el.dataset.beat); rest.since = now;
    nx.el.classList.remove('now'); void nx.el.offsetWidth; nx.el.classList.add('now');   // a tick beside the paragraph the picture just caught up with
    applyScroll(true, readingLine());
  }, 300);
  addEventListener('scroll', onScroll, { passive: true });

  /* ---------------- controls ---------------- */
  /* part 6: a guess before the answer. The guess deafens the round it names, so the sky answers it */
  const bet7 = $('#bet7'), bet7r = $('#bet7r'), scene7 = bet7.closest('.scene');
  let betSkipped = false, betMade = false;
  function unveil7(byScroll) { scene7.classList.add('bet-done'); if (byScroll && !betMade) betSkipped = true; }
  addEventListener('scroll', () => {
    if (!betSkipped || betMade || bet7.hidden) return;
    if (bet7.getBoundingClientRect().bottom > -20) return;   // still on screen, or only just off it
    const next = bet7.nextElementSibling, y0 = next.getBoundingClientRect().top;
    bet7.hidden = true;
    const dy = next.getBoundingClientRect().top - y0; if (dy) scrollBy(0, dy);   // keep what you are reading exactly where it was
  }, { passive: true });
  bet7.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-bet]'); if (!b || b.disabled) return;
    const n = +b.dataset.bet;
    if (state.scene !== 7) { railBtns[anchors.findIndex((a) => a.s === 7)].click(); return; }
    touch(7);
    for (const o of bet7.querySelectorAll('button[data-bet]')) { o.disabled = true; o.setAttribute('aria-pressed', o === b); }
    bet7r.innerHTML = n === 1 ? '<b>Good call.</b> Without round one it gets sloppier, but it is still a flock. Now see what losing round two does.'
      : '<b>Nope.</b> That is what the reader’s scores suggest, but look at the sky: without round two, the flock falls apart.';
    bet7r.hidden = false; kick(bet7r, 'fresh');
    betMade = true; state.deaf7 = n; startle(); hopReader(); unveil7(); paint();
  });

  /* the one hint on the page sits on the first rust phrase, and leaves after the first press anywhere */
  const hint = $('#hint'), hintP = hint && hint.parentElement;
  let hintGone = false;
  function placeHint() {
    if (!hint || hintGone) return;
    const b = hintP.querySelector('.say'), pr = hintP.getBoundingClientRect(), br = b.getBoundingClientRect();
    const x = Math.max(0, br.left - pr.left + Math.min(26, br.width * .25) - 5);
    hint.style.setProperty('--hx', x.toFixed(1) + 'px'); hint.style.setProperty('--hw', Math.max(140, pr.width - x).toFixed(1) + 'px');
    hintP.style.paddingTop = (hint.offsetHeight + 16) + 'px';
  }
  function dropHint() {
    if (!hint || hintGone) return; hintGone = true;
    hint.classList.add('gone'); hintP.style.paddingTop = '';
    setTimeout(() => hint.remove(), 600);
  }
  if (hint) {
    placeHint();
    new IntersectionObserver((es, io) => { if (es[0].isIntersecting) { hint.classList.add('show'); io.disconnect(); } }, { rootMargin: '0px 0px -25% 0px' }).observe(hintP);
    document.addEventListener('click', (e) => { if (e.target.closest('.say, .segc button, .tog')) dropHint(); }, true);
    addEventListener('resize', placeHint);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(placeHint);
  }
  togs.forEach((b) => b.addEventListener('click', () => { touch(state.scene); setRule(b.dataset.rule, !state.on[b.dataset.rule]); }));
  for (const g of segs) g.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    touch(state.scene);
    state[g.dataset.seg] = +b.dataset.v; if (+b.dataset.v) startle();
    if (g.dataset.seg === 'rule7') { Object.assign(replay7, { on: true, t0: performance.now(), dur: .8 + 2.2 * (state.Ts || 0) }); state.T = 0; }   // the new rule's practice plays from the start
    paint(); if (g.dataset.seg === 'fifty') sizeSky();
  });
  for (const b of says) b.addEventListener('click', () => {
    const [k, v] = b.dataset.say.split(':'), n = +v, s = +b.closest('.scene').dataset.scene;
    if (s !== state.scene) { railBtns[anchors.findIndex((a) => a.s === s)].click(); return; }   // a phrase from the part above or below: go there first
    touch(s);
    if (k === 'page') { setPage(n); return; }
    if (k === 'its') state.its = !state.its; else state[k] = state[k] === n ? 0 : n;
    if (k === 'hold4') { const now = performance.now() / 1000; beat.holdT0 = now; beat.origin = now - .05; beat.n = -1; }   // a held step starts over at once; letting go restarts the whole loop
    if (state[k]) startle();
    paint(); if (k === 'fifty') sizeSky();
  });
  for (const n of [1, 2]) pages[n].addEventListener('click', () => { if (state.scene >= 5 && state.scene <= 7) { touch(state.scene); setPage(n); } });
  readerEl.addEventListener('click', () => {   // the reader is clickable too: each click sends it to the other page
    if (state.scene >= 5 && state.scene <= 7) { touch(state.scene); setPage(readingOn() && state.page === 1 ? 2 : 1); } else hopReader();
  });

  /* ---------------- drawing helpers ---------------- */
  /* a flying bird seen from above, nose along +x: scythe wings and a forked tail. flap runs 0 (tucked) to 1 (spread) */
  function flyer(c, dpr, x, y, hx, hy, size, flap, ox, oy) {
    const W = size * (.5 + .62 * flap), tx = -size * (.5 + .28 * (1 - flap));
    c.setTransform(dpr * hx, dpr * hy, -dpr * hy, dpr * hx, dpr * (x + (ox || 0)), dpr * (y + (oy || 0)));
    c.beginPath();
    c.moveTo(size, 0);
    c.quadraticCurveTo(size * .74, -size * .17, size * .42, -size * .17);
    c.bezierCurveTo(size * .44, -W * .55, size * .1, -W * .95, tx, -W);
    c.bezierCurveTo(tx + size * .38, -W * .78, size * .02, -W * .42, -size * .1, -size * .16);
    c.quadraticCurveTo(-size * .5, -size * .1, -size * 1.02, -size * .3);
    c.lineTo(-size * .68, 0);
    c.lineTo(-size * 1.02, size * .3);
    c.quadraticCurveTo(-size * .5, size * .1, -size * .1, size * .16);
    c.bezierCurveTo(size * .02, W * .42, tx + size * .38, W * .78, tx, W);
    c.bezierCurveTo(size * .1, W * .95, size * .44, W * .55, size * .42, size * .17);
    c.quadraticCurveTo(size * .74, size * .17, size, 0);
    c.fill();
  }
  /* a starling seen from above, nose along +x: round head, short square tail, and wings that swing around the shoulder,
     out to the side on the downstroke and folded back along the body on the upstroke. The wings are bowed like a shallow
     crescent. spread runs 0 (folded) to 1 (out); bank leans the bird into a turn, so the inside wing reads shorter */
  function starling(c, dpr, x, y, hx, hy, size, spread, bank, ox, oy) {
    const L = size, ang = (70 - 58 * spread) * Math.PI / 180;
    c.setTransform(dpr * hx, dpr * hy, -dpr * hy, dpr * hx, dpr * (x + (ox || 0)), dpr * (y + (oy || 0)));
    c.beginPath();
    c.moveTo(L * .98, 0);
    c.quadraticCurveTo(L * .86, -L * .16, L * .6, -L * .18);
    for (const sg of [-1, 1]) {
      if (sg === 1) { c.lineTo(-L * .56, -L * .13); c.lineTo(-L * .9, -L * .19); c.quadraticCurveTo(-L * .95, 0, -L * .9, L * .19); c.lineTo(-L * .56, L * .13); }
      const R = L * (.95 + .2 * spread) * (1 - .32 * Math.max(0, sg * bank)) * (1 + .12 * Math.max(0, -sg * bank));
      const lx = L * .34, ly = sg * L * .2, tx = -L * .38, ty = sg * L * .22;
      const px = -Math.sin(ang) * R - L * .02, py = sg * (L * .2 + Math.cos(ang) * R);
      const ex = px - lx, ey = py - ly, el = Math.hypot(ex, ey) || 1, nx = -ey / el * -sg, ny = ex / el * -sg;   // the side of the wing that faces forward
      const bow = L * (.2 + .06 * spread), dip = L * .13;   // the front edge bulges forward, the back edge follows it: a gentle crescent
      const fx = (lx + px) / 2 + nx * bow, fy = (ly + py) / 2 + ny * bow, bx = (px + tx) / 2 + nx * dip, by = (py + ty) / 2 + ny * dip;
      if (sg === -1) { c.lineTo(lx, ly); c.quadraticCurveTo(fx, fy, px, py); c.quadraticCurveTo(bx, by, tx, ty); }
      else { c.lineTo(tx, ty); c.quadraticCurveTo(bx, by, px, py); c.quadraticCurveTo(fx, fy, lx, ly); }
    }
    c.lineTo(L * .6, L * .18);
    c.quadraticCurveTo(L * .86, L * .16, L * .98, 0);
    c.fill();
  }
  /* each bird's wings: bursts of quick beats, then a glide with the wings held out, every bird on its own rhythm; and a lean
     that follows how fast it is turning */
  const wings = new WeakMap(), clock = { t: 0, dt: 0 };
  function wingsOf(key, hx, hy) {
    const t = clock.t, dt = clock.dt;
    let w = wings.get(key);
    if (!w) { w = { per: 1.5 + Math.random() * 1.1, off: Math.random() * 10, beat: 9 + Math.random() * 3, ph: Math.random() * 6.28, hx, hy, ang: Math.atan2(hy, hx), bank: 0, sp: .9, at: -1 }; wings.set(key, w); }
    if (w.at === t) return w;   // already moved on this frame
    w.at = t;
    const flapping = ((t + w.off) % w.per) / w.per < .55;
    w.ph += dt * w.beat * (flapping ? 1 : .15);
    w.sp += ((flapping ? .5 + .5 * Math.sin(w.ph) : 1) - w.sp) * Math.min(1, dt * (flapping ? 30 : 6));
    { const k = Math.min(1, dt / .14); w.hx += (hx - w.hx) * k; w.hy += (hy - w.hy) * k; const l = Math.hypot(w.hx, w.hy) || 1; w.hx /= l; w.hy /= l; }   // a bird that jitters still looks like it is flying somewhere
    const ang = Math.atan2(w.hy, w.hx); let da = ang - w.ang; if (da > Math.PI) da -= 6.2832; if (da < -Math.PI) da += 6.2832; w.ang = ang;
    w.bank += (Math.max(-1, Math.min(1, (dt > 0 ? da / dt : 0) / 2.2)) - w.bank) * Math.min(1, dt * 3);
    return w;
  }
  function arrow(c, x0, y0, x1, y1, head) {
    c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
    const a = Math.atan2(y1 - y0, x1 - x0);
    c.setLineDash([]); c.beginPath(); c.moveTo(x1 + Math.cos(a) * head * .5, y1 + Math.sin(a) * head * .5);
    c.lineTo(x1 - head * Math.cos(a - .55), y1 - head * Math.sin(a - .55)); c.lineTo(x1 - head * Math.cos(a + .55), y1 - head * Math.sin(a + .55)); c.closePath(); c.fill();
  }

  /* ---------------- the framed sky ---------------- */
  const cv = $('#sky'), ctx = cv.getContext('2d');
  const landA = document.createElement('canvas'), landB = document.createElement('canvas'), landC = document.createElement('canvas'), landD = document.createElement('canvas');   // the land in the afternoon, at golden hour, at dusk and at night
  const view = { w: 0, h: 0, H: 0, s: 1, dpr: 1 };
  const beat = { n: -1, len: 0, phase: 0, ph: 0, origin: 0, holdT0: 0, f1: true, f2: true, t1: -9, t2: -9, on1: false, on2: false };
  const clouds = [{ x: .08, y: .2, k: 1.25, v: 5 }, { x: .5, y: .37, k: .85, v: 3.2 }, { x: .82, y: .13, k: 1, v: 4.2 }];
  let sitters = [];          // the birds sitting on the telephone wire
  let aboveGround = null;    // everything above the near ground, so the train can pass behind the hill
  let win = null;            // the window of the house, lit after dusk
  /* the day gets later as the story goes on: one value per scene, 0 is afternoon and 1 is night */
  const TOD = [0, 0, 0, .12, .26, .38, .56, .7, .84, .96, 1];   // where each scene starts; the sun moves as you scroll through it
  let todTo = 0;
  /* the sky's colour at the top, where the birds fly, and at the horizon. The warm colours stay low, under the flock,
     so the three rule colours are still the loudest thing in the picture */
  const SKY = [
    [0,   [196, 224, 247], [226, 239, 250], [253, 243, 222]],
    [.32, [201, 222, 246], [234, 237, 244], [255, 231, 194]],
    [.55, [198, 206, 238], [233, 224, 238], [255, 209, 170]],
    [.68, [178, 182, 228], [224, 204, 228], [255, 188, 152]],
    [.82, [132, 142, 202], [176, 166, 212], [230, 174, 184]],
    [1,   [88, 102, 168],  [118, 128, 190], [164, 160, 210]]
  ];
  const DAY = { far: '#b7d2dc', mid: '#d0deb6', near: '#a9c890', bush: '#84ad7b', pole: '#55616b', wall: '#f5ecdc', roof: '#c68a72', trunk: '#7d6a5c' };
  const GOLD = { far: '#c5c6d8', mid: '#e0d6a6', near: '#c4ba7c', bush: '#9c9c66', pole: '#57525c', wall: '#f7e2c8', roof: '#bb7860', trunk: '#7a6150' };
  const DUSK = { far: '#b3a6cf', mid: '#a598c5', near: '#877bb0', bush: '#72679f', pole: '#3f3a63', wall: '#c9b6d2', roof: '#7b6496', trunk: '#57487a' };
  const NIGHT = { far: '#5d6ca8', mid: '#4e5d98', near: '#3c4a84', bush: '#324078', pole: '#1f2850', wall: '#5a689f', roof: '#2c376a', trunk: '#27315e' };
  let tod = 0;
  const mix = (a, b, u) => a + (b - a) * u, rgb = (c) => 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
  const srnd = mulberry(77), stars = Array.from({ length: 54 }, () => ({ x: srnd(), y: srnd() * .6, r: .6 + srnd() * .9, p: srnd() * 6.28 }));
  function skyAt(u) {
    let i = 1; while (i < SKY.length - 1 && u > SKY[i][0]) i++;
    const a = SKY[i - 1], b = SKY[i], k = clamp((u - a[0]) / (b[0] - a[0]), 0, 1), m = (p, q) => p.map((v, j) => Math.round(v + (q[j] - v) * k));
    return { top: m(a[1], b[1]), mid: m(a[2], b[2]), low: m(a[3], b[3]) };
  }

  /* the land: far hills, a flat plain with a railway, a near hill with a house and a tree, poles and a wire */
  function paintLand(L, pal, record) {
    const { w, h, dpr, s } = view;
    L.width = cv.width; L.height = cv.height;
    const c = L.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.lineCap = 'round';
    const rnd = mulberry(5);
    const far = (x) => h * (.835 - .05 * (.6 * Math.sin(x / w * 5.1 + 1.3) + .4 * Math.sin(x / w * 11.7 + .4)));
    const near = (x) => { const u = x / w; return h * (.925 - .018 * (Math.sin(u * 3.3 + 2) + .4 * Math.sin(u * 9 + 1)) - .18 * Math.exp(-Math.pow((u - .2) / .14, 2)) - .05 * Math.exp(-Math.pow((u - .92) / .1, 2))); };
    const ridge = (fn, fill) => { c.fillStyle = fill; c.beginPath(); c.moveTo(0, h); for (let x = 0; x <= w + 4; x += 4) c.lineTo(x, fn(x)); c.lineTo(w, h); c.fill(); };
    ridge(far, pal.far);
    ridge(() => h * .88, pal.mid);
    c.fillStyle = pal.bush;
    for (let i = 0; i < 4; i++) { const x = w * (.875 + .03 * i + .008 * rnd()), ry = h * (.05 + .035 * rnd()); c.beginPath(); c.ellipse(x, near(x) - ry * .8, h * .013, ry, 0, 0, 6.2832); c.fill(); }
    for (let i = 0; i < 6; i++) { const x = w * (.3 + .09 * rnd()), rr = h * (.018 + .02 * rnd()); c.beginPath(); c.ellipse(x, near(x) - rr * .5, rr * 1.2, rr, 0, 0, 6.2832); c.fill(); }
    const tx = w * .268, ty = near(tx), tr = h * .033;   // the tree
    c.strokeStyle = pal.trunk; c.lineWidth = Math.max(1.4, 1.8 * s); c.beginPath(); c.moveTo(tx, ty + 3); c.lineTo(tx, ty - tr * 1.3); c.stroke();
    c.fillStyle = pal.bush; for (const [ox, oy, k] of [[0, -1.95, 1], [-.72, -1.4, .74], [.76, -1.35, .7]]) { c.beginPath(); c.arc(tx + ox * tr, ty + oy * tr, tr * k, 0, 6.2832); c.fill(); }
    ridge(near, pal.near);
    const u = h * .0115, hx = w * .172, hy = near(hx) + 1;   // the house
    c.fillStyle = pal.wall; c.fillRect(hx - 3.2 * u, hy - 4 * u, 6.4 * u, 4 * u + 3);
    c.fillStyle = pal.roof;
    c.beginPath(); c.moveTo(hx - 4.1 * u, hy - 4 * u); c.lineTo(hx, hy - 6.9 * u); c.lineTo(hx + 4.1 * u, hy - 4 * u); c.closePath(); c.fill();
    c.fillRect(hx + 1.7 * u, hy - 6.7 * u, 1.05 * u, 2.1 * u); c.fillRect(hx + .8 * u, hy - 2.4 * u, 1.4 * u, 2.4 * u + 3);
    const wn = { x: hx - 2.3 * u, y: hy - 3.05 * u, w: 1.6 * u, h: 1.7 * u }; c.fillRect(wn.x, wn.y, wn.w, wn.h);
    const tops = [.42, .535, .65, .765].map((f) => ({ x: w * f, y: near(w * f) - h * .16 }));
    c.strokeStyle = pal.pole; c.lineWidth = Math.max(1.2, 1.5 * s);
    for (const p of tops) { c.beginPath(); c.moveTo(p.x, near(p.x) + 2); c.lineTo(p.x, p.y - h * .01); c.moveTo(p.x - h * .02, p.y); c.lineTo(p.x + h * .02, p.y); c.stroke(); }
    c.lineWidth = Math.max(.8, .8 * s);
    const sag = h * .024, pts = [{ x: w * .345, y: tops[0].y + h * .035 }, ...tops, { x: w * .85, y: tops[3].y + h * .036 }], sit = [];
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1], mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2 + sag;
      c.beginPath(); c.moveTo(a.x, a.y); c.quadraticCurveTo(mx, my, b.x, b.y); c.stroke();
      if (i === 1 || i === 2) for (const t of (i === 1 ? [.3, .43, .52] : [.36, .62])) {
        const v = 1 - t; sit.push({ x: v * v * a.x + 2 * v * t * mx + t * t * b.x, y: v * v * a.y + 2 * v * t * my + t * t * b.y, at: -9, next: 3 + rnd() * 9 });
      }
    }
    if (record) {
      sitters = sit; win = wn;
      aboveGround = new Path2D(); aboveGround.moveTo(0, 0); aboveGround.lineTo(w, 0);
      for (let x = w; x >= -4; x -= 4) aboveGround.lineTo(x, near(x)); aboveGround.closePath();
    }
  }
  function paintScenery() { paintLand(landA, DAY, true); paintLand(landB, GOLD, false); paintLand(landC, DUSK, false); paintLand(landD, NIGHT, false); }
  function sizeSky() {
    const w = cv.clientWidth, h = cv.clientHeight; if (!w || !h) return;
    const dpr = Math.min(2.5, devicePixelRatio || 1);
    const changed = w !== view.w || h !== view.h || dpr !== view.dpr;
    view.w = w; view.h = h; view.dpr = dpr; view.H = h * .74;
    view.s = clamp(Math.min(w, view.H) / 235, .75, 1.8);
    if (changed) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); paintScenery(); }
    main.fit(w, view.H, view.s); fifty.fit(w, view.H, view.s * .72); toy.fit(w, view.H, view.s * .62);
    const pw = w / 2, ph = h / 2, ps = clamp(Math.min(pw, ph) / 200, .62, 1.3);
    quad.forEach((f) => f.fit(pw, ph * .96, ps));
  }
  new ResizeObserver(() => { sizeSky(); measure(); paintKey(); measureSafe(); onScroll(); }).observe(cv);
  let skyShown = true;   // the framed sky only draws while some of it is on screen, so the top flock has the phone to itself
  if ('IntersectionObserver' in window) new IntersectionObserver((es) => { skyShown = es[es.length - 1].isIntersecting; }, { rootMargin: '0px' }).observe(skywin);

  cv.addEventListener('pointerdown', (e) => {
    const s = state.scene; if (s === 3 || s === 9) return;
    const f = flock(), r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    let best = -1, bd = 54;
    f.b.forEach((b, i) => { const d = Math.hypot(b.x - x, b.y - y); if (d < bd) { bd = d; best = i; } });
    if (best >= 0 && best !== f.follow) { f.follow = best; for (const r of RULES) f.ask[r] = [0, 0]; }
  });
  function startle() { const t = performance.now() / 1000; sitters.forEach((p, i) => { p.at = t + i * .06; p.twice = true; }); }

  function drawFlock(f, ox, oy, size) {
    const dpr = view.dpr;
    if (!reduce) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.lineCap = 'round';
      for (const b of f.b) {
        const hst = b.hist, n = hst.length / 2;
        for (let k = 1; k < n; k++) {
          const u = k / n;
          ctx.strokeStyle = 'rgba(' + INK + ',' + (Math.pow(u, 1.6) * .26).toFixed(3) + ')'; ctx.lineWidth = f.s * (.3 + 1.5 * u);
          ctx.beginPath(); ctx.moveTo(ox + hst[2 * k - 2], oy + hst[2 * k - 1]); ctx.lineTo(ox + hst[2 * k], oy + hst[2 * k + 1]); ctx.stroke();
        }
      }
    }
    ctx.fillStyle = 'rgb(' + INK + ')';
    for (const b of f.b) { const w = wingsOf(b, b.hx, b.hy); starling(ctx, dpr, b.x, b.y, w.hx, w.hy, size, w.sp, w.bank, ox, oy); }
  }
  /* the sky itself: colour for the time of day, stars, sun, moon, clouds, and a balloon drifting by */
  const shoot = { at: -9, next: 0, x: .5, y: .1 };
  function drawSkyBack(t, plain) {
    const { w, h, s, dpr } = view, col = skyAt(tod);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const g = ctx.createLinearGradient(0, 0, 0, h); g.addColorStop(0, rgb(col.top)); g.addColorStop(.56, rgb(col.mid)); g.addColorStop(.86, rgb(col.low)); g.addColorStop(1, rgb(col.low));
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
    const night = clamp((tod - .7) / .22, 0, 1);
    if (night > 0) {
      ctx.fillStyle = '#fff';
      for (const st of stars) { ctx.globalAlpha = night * (.45 + .45 * Math.sin(t * 1.3 + st.p)); ctx.beginPath(); ctx.arc(st.x * w, st.y * h, st.r * Math.max(1, s * .8), 0, 6.2832); ctx.fill(); }
      if (night > .6 && !reduce) {   // now and then, a shooting star
        if (!shoot.next) shoot.next = t + 4;
        if (t > shoot.next) { shoot.at = t; shoot.next = t + 10 + Math.random() * 12; shoot.x = .25 + .55 * Math.random(); shoot.y = .06 + .2 * Math.random(); }
        const u = (t - shoot.at) / .7;
        if (u > 0 && u < 1) { const x = (shoot.x + u * .12) * w, y = (shoot.y + u * .07) * h; ctx.globalAlpha = Math.sin(Math.PI * u) * .9; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - .05 * w, y - .029 * h); ctx.stroke(); }
      }
      ctx.globalAlpha = 1;
    }
    const sa = 1 - clamp((tod - .7) / .14, 0, 1);   // the sun sinks through the afternoon and is gone by dusk
    if (sa > 0) {
      const sx = w * (.6 + .05 * tod), sy = h * (.3 + .78 * tod), r = Math.min(h * .085, w * .075);
      const lo = clamp(tod / .66, 0, 1), sun = [255, Math.round(mix(238, 172, lo)), Math.round(mix(168, 104, lo))].join(',');   // pale yellow up high, orange as it sets
      const halo = ctx.createRadialGradient(sx, sy, r * .8, sx, sy, r * 3.4);
      halo.addColorStop(0, 'rgba(' + sun + ',' + (.7 * sa).toFixed(2) + ')'); halo.addColorStop(1, 'rgba(' + sun + ',0)');
      ctx.fillStyle = halo; ctx.fillRect(sx - r * 3.5, sy - r * 3.5, r * 7, r * 7);
      ctx.globalAlpha = sa; ctx.fillStyle = 'rgb(' + sun + ')'; ctx.beginPath(); ctx.arc(sx, sy, r, 0, 6.2832); ctx.fill(); ctx.globalAlpha = 1;
    }
    const ma = clamp((tod - .74) / .14, 0, 1);   // then the moon comes up
    if (ma > 0) {
      const mx = w * .2, my = h * mix(.42, .2, clamp((tod - .74) / .26, 0, 1)), r = Math.min(h * .045, w * .045);
      const halo = ctx.createRadialGradient(mx, my, r * .8, mx, my, r * 3);
      halo.addColorStop(0, 'rgba(255,255,255,' + (.5 * ma).toFixed(2) + ')'); halo.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = halo; ctx.fillRect(mx - r * 3.1, my - r * 3.1, r * 6.2, r * 6.2);
      ctx.globalAlpha = ma; ctx.fillStyle = '#fbfcfe'; ctx.beginPath(); ctx.arc(mx, my, r, 0, 6.2832); ctx.fill();
      ctx.fillStyle = 'rgba(150,160,205,.42)'; for (const [ox, oy, k] of [[-.3, -.2, .22], [.28, .25, .16], [.1, -.45, .1]]) { ctx.beginPath(); ctx.arc(mx + ox * r, my + oy * r, k * r, 0, 6.2832); ctx.fill(); }
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = 'rgba(' + col.low.map((v) => Math.round(mix(v, 255, .62))).join(',') + ',' + mix(.85, .3, tod).toFixed(2) + ')';   // clouds pick up the colour of the horizon
    for (const cl of clouds) {
      const L = 60 * cl.k * s, x = ((cl.x * w + (reduce ? 0 : t * cl.v)) % (w + 4 * L)) - 2 * L, y = cl.y * h;
      for (const [ox, oy, rx, ry] of [[0, 0, 1, .2], [.5, -.12, .6, .2], [-.55, .06, .55, .14], [1.05, .08, .5, .12]]) {
        ctx.beginPath(); ctx.ellipse(x + ox * L, y + oy * L, rx * L, ry * L, 0, 0, 6.2832); ctx.fill();
      }
    }
    const ba = plain ? 0 : 1 - clamp((tod - .52) / .2, 0, 1);   // a hot-air balloon, out for the afternoon
    if (ba > 0) {
      const R = 8.5 * s, bx = ((w * .78 + (reduce ? 0 : t * 4.5)) % (w + 8 * R)) - 4 * R, by = h * (.22 + .012 * Math.sin(t * .5));
      ctx.globalAlpha = ba;
      ctx.strokeStyle = 'rgba(' + INK + ',.5)'; ctx.lineWidth = .8; ctx.beginPath();
      ctx.moveTo(bx - R * .2, by + R * 1.28); ctx.lineTo(bx - R * .17, by + R * 1.6); ctx.moveTo(bx + R * .2, by + R * 1.28); ctx.lineTo(bx + R * .17, by + R * 1.6); ctx.stroke();
      ctx.fillStyle = '#8f7bc8'; ctx.beginPath(); ctx.arc(bx, by, R, Math.PI * 5 / 6, Math.PI * 13 / 6); ctx.lineTo(bx + R * .22, by + R * 1.3); ctx.lineTo(bx - R * .22, by + R * 1.3); ctx.closePath(); ctx.fill();
      ctx.fillStyle = 'rgba(255,246,226,.9)'; ctx.beginPath(); ctx.ellipse(bx, by + R * .04, R * .3, R * .97, 0, 0, 6.2832); ctx.fill();
      ctx.fillStyle = 'rgba(' + INK + ',.62)'; ctx.fillRect(bx - R * .2, by + R * 1.58, R * .4, R * .3);
      ctx.globalAlpha = 1;
    }
  }
  /* a small train that crosses the plain now and then, passing behind the hill */
  const train = { at: -99, next: 0, dur: 15 };
  function drawTrain(t) {
    if (reduce || !aboveGround) return;
    if (!train.next) train.next = t + 6;
    if (t > train.next) { train.at = t; train.next = t + 26 + Math.random() * 20; }
    const u = (t - train.at) / train.dur; if (u < 0 || u > 1) return;
    const { w, h, dpr } = view, k = Math.max(.9, h * .0034), y = h * .88, cars = 5, cl = 9 * k, gap = 1.3 * k, total = cars * (cl + gap), x0 = -total + (w + total) * u;
    const lit = clamp((tod - .5) / .18, 0, 1);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.save(); ctx.clip(aboveGround);
    for (let i = 0; i < cars; i++) {
      const x = x0 + i * (cl + gap), eng = i === cars - 1;
      ctx.fillStyle = 'rgb(' + Math.round(mix(92, 30, tod)) + ',' + Math.round(mix(104, 38, tod)) + ',' + Math.round(mix(122, 80, tod)) + ')';
      ctx.beginPath(); if (ctx.roundRect) ctx.roundRect(x, y - 3 * k, cl, 3 * k, k); else ctx.rect(x, y - 3 * k, cl, 3 * k); ctx.fill();
      if (eng) ctx.fillRect(x + cl * .1, y - 4.2 * k, cl * .34, 1.6 * k);
      if (lit > 0) { ctx.fillStyle = 'rgba(255,247,214,' + lit.toFixed(2) + ')'; for (let j = 0; j < 3; j++) ctx.fillRect(x + cl * (.16 + .27 * j), y - 2.3 * k, cl * .15, k); }
    }
    ctx.restore();
  }
  function drawScenery(t) {
    const { s, dpr } = view;
    drawSkyBack(t, false);
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.drawImage(landA, 0, 0);
    for (const [L, from, to] of [[landB, .06, .3], [landC, .52, .64], [landD, .68, .9]]) {   // each later hour fades in over the one before
      const a = clamp((tod - from) / (to - from), 0, 1); if (a > 0) { ctx.globalAlpha = a; ctx.drawImage(L, 0, 0); }
    }
    ctx.globalAlpha = 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const lit = clamp((tod - .5) / .18, 0, 1);
    if (lit > 0 && win) {   // someone is home
      const cx = win.x + win.w / 2, cy = win.y + win.h / 2, gl = ctx.createRadialGradient(cx, cy, 0, cx, cy, win.w * 4);
      gl.addColorStop(0, 'rgba(255,247,214,' + (.5 * lit).toFixed(2) + ')'); gl.addColorStop(1, 'rgba(255,247,214,0)');
      ctx.fillStyle = gl; ctx.fillRect(cx - win.w * 4, cy - win.w * 4, win.w * 8, win.w * 8);
      ctx.fillStyle = 'rgba(255,247,214,' + lit.toFixed(2) + ')'; ctx.fillRect(win.x, win.y, win.w, win.h);
    }
    drawTrain(t);
    /* the birds on the wire: they sit, and now and then one hops */
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = '#161b22';
    const k = Math.max(1, 1.15 * s);
    for (const p of sitters) {
      if (t > p.next && !reduce) { p.at = t; p.next = t + 5 + Math.random() * 12; }
      let u = (t - p.at) / .32, lift = 0;
      if (u >= 0 && u < 1) lift = Math.sin(Math.PI * u) * 3.5 * k; else if (p.twice && u >= 1 && u < 2) lift = Math.sin(Math.PI * (u - 1)) * 2.5 * k; else if (u >= 2) p.twice = false;
      ctx.beginPath(); ctx.ellipse(p.x, p.y - 2.3 * k - lift, 2.1 * k, 2.6 * k, 0, 0, 6.2832); ctx.fill();
      ctx.beginPath(); ctx.arc(p.x + .9 * k, p.y - 5.1 * k - lift, 1.5 * k, 0, 6.2832); ctx.fill();
    }
  }
  /* our bird: a ring, one arrow per rule, and (when the reader is reading) its teal guess rings.
     The arrows are eased so they swing and stretch but never blink in and out */
  const arrows = RULES.map((r) => ({ rule: r, dx: 1, dy: 0, len: .2, e: [0, 0], g: null }));
  /* The reader's guesses. Each time the page it is sitting on is rewritten, it throws one ring per rule at the tip of that
     rule's arrow. How far off a ring lands is drawn from the real score, so a rule it reads well lands on the tip and one it
     reads badly lands wide. A ring flies, lands, shows whether it hit, and fades before the next throw */
  const FLY = .7, HOLD = 1.4, FADE = .4;
  function throwGuesses(t, quick) {
    if (!quick && arrows.some((A) => A.g && t - A.g.t0 < FLY + .8)) return;   // the last throw has barely landed: let it be seen
    let i = 0;
    for (const A of arrows) {
      const R = ringsOn() ? readable(A.rule) : null;
      if (R === null) { A.g = null; continue; }
      const a = Math.random() * 6.2832, m = Math.sqrt(Math.max(0, 1 - R)) * 34 * Math.min(1.5, .4 + .6 * Math.abs(gauss()));
      A.g = { t0: t + (quick ? .65 : .5) + i * .12, ox: Math.cos(a) * m, oy: Math.sin(a) * m, lead: !i, hopped: false };
      i++;
    }
  }
  function doneShare(f, r) {   // how much of what the rule asks actually gets done
    if (f === toy) return f.W[r];
    if (state.scene === 8) return clamp(practiceAt(state.T).use, 0, 1.3);
    return clamp(f.c * f.W[r], -1, 1.3);
  }
  function drawOurBird(f, dt, bubble, showArrows, t) {
    const { s, dpr } = view, F = f.b[f.follow], k = f === toy ? .8 : 1, sc = state.scene;
    { // the key steps aside when the ringed bird flies behind it
      const kr = keyEl.getBoundingClientRect(), cr = cv.getBoundingClientRect(), m = 22 * s * k;
      const bx = F.x + cr.left, by = F.y + cr.top;
      keyEl.classList.toggle('under', kr.width > 0 && bx > kr.left - m && bx < kr.right + m && by > kr.top - m && by < kr.bottom + m);
      const y0 = kr.top - cr.top;
      f.avoid = kr.width > 0 && y0 < f.h - 20 ? { x0: kr.left - cr.left, y0, x1: kr.right - cr.left, y1: kr.bottom - cr.top, m: 46 * s } : null;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.setLineDash([]);
    ctx.strokeStyle = 'rgba(' + INK + ',.7)'; ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.arc(F.x, F.y, 17 * s * k, 0, 6.2832); ctx.stroke();
    const R0 = 20 * s * k, Lmax = 44 * s * k, rings = ringsOn(), ka = Math.min(1, dt / .32), kn = Math.min(1, dt / 1.1);
    for (const A of arrows) {
      const r = A.rule, a = f.ask[r], al = Math.hypot(a[0], a[1]), l = Math.min(1, al / (2.2 * f.rms[r] + 1e-6));
      if (l > .07) { A.dx += (a[0] / al - A.dx) * ka; A.dy += (a[1] / al - A.dy) * ka; }
      A.len += ((l < .1 ? .2 : .42 + .58 * l) - A.len) * ka;
      A.e[0] += -A.e[0] * kn + gauss() * Math.sqrt(2 * kn) * .7; A.e[1] += -A.e[1] * kn + gauss() * Math.sqrt(2 * kn) * .7;
      if (!showArrows || (sc === 8 && r !== R7[state.rule7])) continue;
      const dl = Math.hypot(A.dx, A.dy) || 1, dx = A.dx / dl, dy = A.dy / dl, ul = A.len;
      const done = doneShare(f, r), same = Math.abs(done - 1) < .06, fade = clamp((ul - .2) / .2, 0, 1);
      const tipX = F.x + dx * (R0 + Lmax * ul), tipY = F.y + dy * (R0 + Lmax * ul);
      ctx.lineCap = 'round';
      if (!same && f !== toy) {   // what the rule asks, faint (in the toy an off rule asks nothing)
        ctx.strokeStyle = ctx.fillStyle = 'rgba(254,254,254,' + (.3 + .45 * fade).toFixed(2) + ')'; ctx.lineWidth = 3.6 * s; ctx.setLineDash([3 * s, 3.5 * s]);
        arrow(ctx, F.x + dx * R0, F.y + dy * R0, tipX, tipY, 6.8 * s);
        ctx.strokeStyle = ctx.fillStyle = 'rgba(' + HUE[r] + ',' + (.34 + .5 * fade).toFixed(2) + ')'; ctx.lineWidth = 1.8 * s; ctx.setLineDash([3 * s, 3.5 * s]);
        arrow(ctx, F.x + dx * R0, F.y + dy * R0, tipX, tipY, 5.5 * s);
      }
      const L = Lmax * ul * done;
      if (Math.abs(L) > 2.5 * s) {   // what the flock actually does about it
        const sgn = L < 0 ? -1 : 1, ax0 = F.x + dx * R0 * sgn, ay0 = F.y + dy * R0 * sgn, ax1 = F.x + dx * (R0 * sgn + L), ay1 = F.y + dy * (R0 * sgn + L);
        ctx.setLineDash([]); ctx.lineJoin = 'round';
        ctx.strokeStyle = ctx.fillStyle = 'rgba(254,254,254,.85)'; ctx.lineWidth = (2.5 * k + 2.6) * s; arrow(ctx, ax0, ay0, ax1, ay1, 6.5 * s * k); ctx.lineWidth = 2.6 * s; ctx.stroke();
        ctx.strokeStyle = ctx.fillStyle = 'rgb(' + HUE[r] + ')'; ctx.lineWidth = 2.5 * s * k;
        arrow(ctx, ax0, ay0, ax1, ay1, 6.5 * s * k);
      }
      if (rings) {   // the reader's guess: a ring thrown from the notes, which should land on the arrow's tip
        const G = A.g; if (!G) continue;
        const age = reduce ? FLY + .6 : t - G.t0;
        if (G.lead && !G.hopped && age >= -.1) { G.hopped = true; if (!reduce) hopReader(); }   // the reader hops as it throws
        if (age < 0 || age > FLY + HOLD + FADE) continue;
        const lx = clamp(tipX + G.ox * s, 10, view.w - 10), ly = clamp(tipY + G.oy * s, geo.top + 10, Math.min(geo.low, view.h) - 10);   // a guess never lands outside the frame or under a label
        const hit = Math.hypot(lx - tipX, ly - tipY) < 8.5 * s, rr = 6.4 * s, al = .3 + .7 * fade;
        ctx.setLineDash([]);
        if (age < FLY) {   // in the air, with a few dots trailing behind it
          const at = (v) => { const e = 1 - (1 - v) * (1 - v); return [geo.gx + (lx - geo.gx) * e, geo.gy + (ly - geo.gy) * e - Math.sin(Math.PI * v) * 38 * s]; };
          const u = age / FLY;
          for (let j = 4; j >= 1; j--) { const uj = u - j * .07; if (uj <= 0) continue; const p = at(uj); ctx.fillStyle = 'rgba(' + TEAL + ',' + (.55 * (1 - j / 5)).toFixed(2) + ')'; ctx.beginPath(); ctx.arc(p[0], p[1], (2.9 - .45 * j) * s, 0, 6.2832); ctx.fill(); }
          const p = at(u), r0 = rr * (.5 + .5 * u);
          ctx.strokeStyle = 'rgba(254,254,254,.85)'; ctx.lineWidth = 4.8 * s; ctx.beginPath(); ctx.arc(p[0], p[1], r0, 0, 6.2832); ctx.stroke();
          ctx.strokeStyle = 'rgb(' + TEAL + ')'; ctx.lineWidth = 2.3 * s; ctx.beginPath(); ctx.arc(p[0], p[1], r0, 0, 6.2832); ctx.stroke();
          continue;
        }
        const la = age - FLY, out = (1 - clamp((la - HOLD) / FADE, 0, 1)) * al;
        if (la < .5 && !reduce) {   // it lands: a ripple, bigger when it lands on the tip
          const v = la / .5; ctx.strokeStyle = 'rgba(' + TEAL + ',' + ((1 - v) * (hit ? .85 : .45)).toFixed(2) + ')'; ctx.lineWidth = (hit ? 2.4 : 1.4) * s;
          ctx.beginPath(); ctx.arc(lx, ly, rr + v * (hit ? 15 : 8) * s, 0, 6.2832); ctx.stroke();
        }
        if (!hit) { ctx.strokeStyle = 'rgba(' + TEAL + ',' + out.toFixed(2) + ')'; ctx.lineWidth = 1.3 * s; ctx.setLineDash([2 * s, 3 * s]); ctx.beginPath(); ctx.moveTo(tipX, tipY); ctx.lineTo(lx, ly); ctx.stroke(); ctx.setLineDash([]); }   // how far off it landed
        ctx.strokeStyle = 'rgba(254,254,254,' + (.85 * out).toFixed(2) + ')'; ctx.lineWidth = 4.8 * s; ctx.beginPath(); ctx.arc(lx, ly, rr, 0, 6.2832); ctx.stroke();
        ctx.strokeStyle = 'rgba(' + TEAL + ',' + out.toFixed(2) + ')'; ctx.lineWidth = 2.3 * s; ctx.beginPath(); ctx.arc(lx, ly, rr, 0, 6.2832); ctx.stroke();
        if (hit) { ctx.fillStyle = 'rgba(' + TEAL + ',' + (out * (.32 + .68 * Math.max(0, 1 - la / .5))).toFixed(2) + ')'; ctx.beginPath(); ctx.arc(lx, ly, rr - 2.4 * s, 0, 6.2832); ctx.fill(); }   // a hit fills in
      }
    }
    if (bubble) {   // the thought bubbles that tie our bird to its notes
      const ax = geo.padX, ay = geo.padY, d = Math.hypot(F.x - ax, F.y - ay);
      if (d > 50 * s) for (let i = 0; i < 4; i++) {
        const u = .16 + i * .21, x = F.x + (ax - F.x) * u, y = F.y + (ay - F.y) * u - Math.sin(u * 3.1) * 10 * s, r = (2 + i * 1.5) * s;
        const hot = Math.max(pulseAt(beat.t1, t, i), pulseAt(beat.t2, t, i));
        ctx.fillStyle = hot > 0 ? 'rgba(' + INK + ',' + (.15 + .7 * hot).toFixed(2) + ')' : 'rgba(254,254,254,.92)';
        ctx.strokeStyle = 'rgb(' + INK + ')'; ctx.lineWidth = 1.3;
        ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill(); ctx.stroke();
      }
    }
  }
  /* part 3, the two rounds. Round one: every bird hears where the others are (a faint web between all of them, the lines
     into the ringed bird drawn stronger, and a dot sliding in along each). After it, every bird is holding a first page.
     Round two: the lines come back dashed and this time it is those pages that slide in, so you can see the ringed bird
     hearing what the others already heard */
  const roundU = (n, t) => (n === 1 ? beat.on1 : beat.on2) ? (t - (n === 1 ? beat.t1 : beat.t2)) / (beat.len * .27) : -1;
  function drawRounds(f, F, t) {
    const { s, dpr } = view, u1 = roundU(1, t), u2 = roundU(2, t), B = f.b;
    const toOurs = () => { ctx.beginPath(); B.forEach((b, i) => { if (i !== f.follow) { ctx.moveTo(b.x, b.y); ctx.lineTo(F.x, F.y); } }); ctx.stroke(); };
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.lineCap = 'round'; ctx.setLineDash([]);
    if (u1 >= 0 && u1 <= 1) {
      const a = Math.min(1, Math.sin(Math.PI * u1) * 2.2), e = u1 * u1 * (3 - 2 * u1);
      ctx.strokeStyle = 'rgba(' + INK + ',' + (a * .13).toFixed(3) + ')'; ctx.lineWidth = Math.max(.7, .7 * s); ctx.beginPath();
      for (let i = 0; i < B.length; i++) if (i !== f.follow) for (let j = i + 1; j < B.length; j++) if (j !== f.follow) { ctx.moveTo(B[i].x, B[i].y); ctx.lineTo(B[j].x, B[j].y); }
      ctx.stroke();
      ctx.strokeStyle = 'rgba(254,254,254,' + (a * .7).toFixed(3) + ')'; ctx.lineWidth = 3.4 * s; toOurs();
      ctx.strokeStyle = 'rgba(' + INK + ',' + (a * .8).toFixed(3) + ')'; ctx.lineWidth = Math.max(1.3, 1.5 * s); toOurs();
      ctx.fillStyle = 'rgba(' + INK + ',' + Math.min(1, a).toFixed(3) + ')'; ctx.strokeStyle = 'rgba(254,254,254,' + (a * .9).toFixed(3) + ')'; ctx.lineWidth = 1.6 * s;
      B.forEach((b, i) => { if (i !== f.follow) { ctx.beginPath(); ctx.arc(b.x + (F.x - b.x) * e, b.y + (F.y - b.y) * e, 3.6 * s, 0, 6.2832); ctx.fill(); ctx.stroke(); } });
    }
    if (u2 >= 0 && u2 <= 1) {
      const a2 = Math.min(1, Math.sin(Math.PI * u2) * 2.2);
      ctx.strokeStyle = 'rgba(254,254,254,' + (a2 * .6).toFixed(3) + ')'; ctx.lineWidth = 3.4 * s; toOurs();
      ctx.strokeStyle = 'rgba(' + INK + ',' + (a2 * .8).toFixed(3) + ')'; ctx.lineWidth = Math.max(1.3, 1.5 * s);
      ctx.setLineDash([5 * s, 5 * s]); toOurs(); ctx.setLineDash([]);
    }
  }
  function sheet(x, y, k, a) {   // one small page
    const w = 10 * k, h = 12.5 * k;
    ctx.fillStyle = 'rgba(' + INK + ',' + (.22 * a).toFixed(3) + ')'; ctx.beginPath(); if (ctx.roundRect) ctx.roundRect(x - w / 2 + 1.6 * k, y - h / 2 + 2 * k, w, h, 1.8 * k); else ctx.rect(x - w / 2 + 1.6 * k, y - h / 2 + 2 * k, w, h); ctx.fill();   // a little shadow lifts the page off the sky
    ctx.fillStyle = 'rgba(254,254,254,' + a.toFixed(3) + ')'; ctx.strokeStyle = 'rgba(' + INK + ',' + a.toFixed(3) + ')'; ctx.lineWidth = 1.7;
    ctx.beginPath(); if (ctx.roundRect) ctx.roundRect(x - w / 2, y - h / 2, w, h, 1.6 * k); else ctx.rect(x - w / 2, y - h / 2, w, h); ctx.fill(); ctx.stroke();
    ctx.lineWidth = 1.3; ctx.beginPath(); for (const d of [-.22, 0, .22]) { ctx.moveTo(x - w * .25, y + h * d); ctx.lineTo(x + w * .25, y + h * d); } ctx.stroke();
  }
  function drawPages(f, F, t) {
    if (!beat.f1 || !beat.on1 || !beat.len) return;
    const { s, dpr } = view, u1 = roundU(1, t), u2 = beat.f2 ? roundU(2, t) : -1, k = Math.max(1, s), ph = beat.ph;
    const held = clamp((u1 - .8) / .2, 0, 1) * (state.hold4 === 3
      ? clamp((ph - .74) / .04, 0, 1) * (1 - clamp((ph - .94) / .05, 0, 1))   // "then it moves" held: the pages stay through the move it makes from them
      : 1 - clamp((ph - .72) / .06, 0, 1));   // from the end of round one until the move
    if (held <= 0) return;
    const ox = 14 * s, oy = -13 * s, e = u2 >= 0 && u2 <= 1 ? u2 * u2 * (3 - 2 * u2) : -1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.setLineDash([]); ctx.lineCap = 'round';
    f.b.forEach((b, i) => {
      if (i === f.follow) return;
      sheet(b.x + ox, b.y + oy, k, held);                                                                                   // the page it keeps
      if (e >= 0) sheet(b.x + ox + (F.x - b.x - ox) * e, b.y + oy + (F.y - b.y - oy) * e, k, held * Math.min(1, (1 - e) * 5));   // and what the ringed bird hears of it
    });
  }
  const pulseAt = (t0, t, i) => { const u = (t - t0 - i * .07) / .22; return u > 0 && u < 1 ? Math.sin(Math.PI * u) : 0; };
  function drawSingle(t, dt) {
    const { s, dpr } = view, f = flock(), sc = state.scene;
    drawScenery(t);
    const F = f.b[f.follow];
    if (!reduce && sc === 4) drawRounds(f, F, t);
    const size = f === toy ? Math.max(4, 5 * f.s) : f === fifty ? Math.max(5.4, 7.6 * f.s) : Math.max(6.6, 9.2 * f.s);
    if (sc !== 3) { drawFlock(f, 0, 0, size); if (!reduce && sc === 4) drawPages(f, F, t); drawOurBird(f, dt, sc >= 4, sc !== 4, t); return; }
    f.avoid = state.its ? { x0: 0, y0: 0, x1: 250, y1: 92, m: 34 * f.s } : null;   // keep the flock out from under the legend in the machine's pane
    const A = drawRecorded(f, t, size);
    if (A && state.split < .995) {   // the same bird is ringed in the sky, among the dots and in the table
      const o = f.follow * 5;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.setLineDash([]); ctx.strokeStyle = 'rgba(' + INK + ',' + (.75 * clamp((1 - state.split) * 4, 0, 1)).toFixed(2) + ')'; ctx.lineWidth = 1.3;
      ctx.beginPath(); ctx.arc(A[o], A[o + 1], size * 1.75, 0, 6.2832); ctx.stroke();
      drawMachineSide(f, t, A);
    }
  }
  /* scene 3 shows the flock a moment late, from its recording. Nobody can tell, and it means the machine's guesses
     can be drawn where the birds really are about to be */
  const LAG = .8, CYCLE = 2.1, guess = { t0: -9, f: null, pts: new Float32Array(0) };
  function drawRecorded(f, t, size) {
    const R = f.rec, dpr = view.dpr, n = f.n;
    if (!R.length) { drawFlock(f, 0, 0, size); return null; }
    const k = f.shown(t), keep = n > 30 ? 28 : 60, k0 = Math.max(0, k - keep), m = k - k0;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!reduce && m > 1) {
      ctx.lineCap = 'round';
      for (let j = 1; j <= m; j++) {
        const u = j / m, a = R[k0 + j - 1].a, b = R[k0 + j].a;
        ctx.strokeStyle = 'rgba(' + INK + ',' + (Math.pow(u, 1.6) * .26).toFixed(3) + ')'; ctx.lineWidth = f.s * (.3 + 1.5 * u);
        ctx.beginPath(); for (let o = 0; o < n * 5; o += 5) { ctx.moveTo(a[o], a[o + 1]); ctx.lineTo(b[o], b[o + 1]); } ctx.stroke();
      }
    }
    const A = R[k].a; ctx.fillStyle = 'rgb(' + INK + ')';
    for (let o = 0; o < n * 5; o += 5) { const w = wingsOf(f.b[o / 5], A[o + 2], A[o + 3]); starling(ctx, dpr, A[o], A[o + 1], w.hx, w.hy, size, w.sp, w.bank, 0, 0); }
    return A;
  }
  /* scene 3: the right pane is the very same flock the way the machine gets it. No sky, no rules: a dot per bird and a
     table of numbers. Every couple of seconds it guesses (a ring per bird), and then the birds fly into the rings */
  let wall = { at: -9, rows: [], more: 0 };
  function drawMachineSide(f, t, A) {
    const { w, h, dpr } = view, X = Math.round(state.split * w), k = Math.max(.8, f.s * .8), n = f.n, pw = w - X, tight = pw < 215;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.save(); ctx.beginPath(); ctx.rect(X, 0, pw, h); ctx.clip();
    ctx.fillStyle = '#fefefe'; ctx.fillRect(X, 0, pw, h);
    ctx.strokeStyle = '#e4e8ec'; ctx.lineWidth = 1; ctx.beginPath();
    for (let x = X + 36; x < w; x += 36) { ctx.moveTo(x + .5, 0); ctx.lineTo(x + .5, h); }
    for (let y = 36; y < h; y += 36) { ctx.moveTo(X, y + .5); ctx.lineTo(w, y + .5); }
    ctx.stroke();
    /* a new set of guesses: where each bird is right now is where the late picture will show it LAG seconds from now */
    const ready = f.lag >= LAG - .001 && state.split < .8 && f !== fifty;   // guesses were only scored on twelve birds, so fifty gets no rings
    if (ready && (t - guess.t0 > CYCLE || guess.f !== f)) {   // kept as fractions of the pane, so they survive the pane changing size
      if (guess.pts.length !== n * 2) guess.pts = new Float32Array(n * 2);
      for (let i = 0; i < n; i++) { guess.pts[2 * i] = (f.b[i].x + gauss() * 1.1 * k) / f.w; guess.pts[2 * i + 1] = (f.b[i].y + gauss() * 1.1 * k) / f.h; }
      guess.t0 = t; guess.f = f;
    }
    const u = ready && guess.f === f ? t - guess.t0 : -1, live = u >= 0 && u < LAG + .5, landed = u >= LAG;   // once a bird lands in its ring, the ring goes with it and fades
    const al = live ? clamp(u / .28, 0, 1) * (1 - clamp((u - LAG - .12) / .38, 0, 1)) : 0, pop = landed ? Math.max(0, 1 - (u - LAG) / .3) : 0;
    const P = guess.pts;
    for (let i = 0; i < n; i++) {
      const x = X + A[i * 5], y = A[i * 5 + 1];
      if (al > 0) {
        const gx = landed ? x : X + P[2 * i] * f.w, gy = landed ? y : P[2 * i + 1] * f.h;
        if (!landed) { ctx.strokeStyle = 'rgba(' + INK + ',' + (.34 * al).toFixed(2) + ')'; ctx.lineWidth = 1; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(gx, gy); ctx.stroke(); ctx.setLineDash([]); }
        ctx.strokeStyle = 'rgba(' + INK + ',' + al.toFixed(2) + ')'; ctx.lineWidth = 1.5 + 1.3 * pop;
        ctx.beginPath(); ctx.arc(gx, gy, (5.4 * (.5 + .5 * clamp(u / .28, 0, 1)) + 2.6 * pop) * k, 0, 6.2832); ctx.stroke();
      }
      ctx.fillStyle = 'rgb(' + INK + ')'; ctx.beginPath(); ctx.arc(x, y, 3.3 * k, 0, 6.2832); ctx.fill();
    }
    const fo = f.follow * 5, ringR = 11 * k;   // the ringed bird, as the machine gets it
    ctx.strokeStyle = 'rgba(' + INK + ',.75)'; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.arc(X + A[fo], A[fo + 1], ringR, 0, 6.2832); ctx.stroke();
    /* what the two marks mean, said in the picture */
    ctx.textBaseline = 'middle'; ctx.textAlign = 'left'; ctx.font = '600 12px "Bricolage Grotesque", Arial, sans-serif';
    const lx = X + (tight ? 6 : 12), ly = tight ? 54 : 60, room = pw - (tight ? 38 : 50);
    const guessLabel = ['where the machine guesses it will be a moment later', 'the machine\u2019s guess for a moment later', 'its guess for a moment later'].find((s) => ctx.measureText(s).width <= room) || 'its guess';
    const two = f !== fifty, lw = Math.max(two ? ctx.measureText(guessLabel).width : 0, ctx.measureText('a bird, right now').width) + 34;
    ctx.fillStyle = 'rgba(254,254,254,.93)'; ctx.beginPath(); if (ctx.roundRect) ctx.roundRect(lx, ly - 13, lw, two ? 45 : 26, 10); else ctx.rect(lx, ly - 13, lw, two ? 45 : 26); ctx.fill();
    ctx.fillStyle = 'rgb(' + INK + ')'; ctx.beginPath(); ctx.arc(lx + 14, ly, 3.6, 0, 6.2832); ctx.fill(); ctx.fillText('a bird, right now', lx + 26, ly + .5);
    if (two) { ctx.strokeStyle = 'rgb(' + INK + ')'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(lx + 14, ly + 19, 4.6, 0, 6.2832); ctx.stroke(); ctx.fillText(guessLabel, lx + 26, ly + 19.5); }
    /* the table of numbers: four per bird, the ringed bird first, refreshed a few times a second so it can be read */
    const fs = tight ? 9 : 11, cw = fs * (tight ? 3.7 : 3.9), rh = fs * 1.55, y0 = view.H + 4, x1 = w - (tight ? 9 : 16), x0 = x1 - 4 * cw + 6, rows = Math.max(1, Math.floor((h - y0 - 48) / rh) - (n > 6 ? 1 : 0));
    if (t - wall.at > .22 || wall.n !== n) {
      const fmt = (v) => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(2), out = [], order = [f.follow];
      for (let i = 0; i < n && order.length < rows; i++) if (i !== f.follow) order.push(i);
      for (const i of order) { const o = i * 5; out.push([(A[o] / f.w).toFixed(2), (A[o + 1] / f.h).toFixed(2), fmt(A[o + 2]), fmt(A[o + 3])]); }
      wall = { at: t, rows: out, more: Math.max(0, n - out.length), n };
    }
    ctx.textBaseline = 'alphabetic';
    ctx.font = '13.5px Newsreader, Georgia, serif'; ctx.fillStyle = 'rgb(' + INK + ')';
    ctx.textAlign = 'right'; ctx.fillText(tight ? 'one row per bird' : 'everything it gets, one row per bird', x1, y0 + 12);
    ctx.font = '600 ' + (tight ? 10 : 11.5) + 'px "Bricolage Grotesque", Arial, sans-serif';
    ctx.fillText('where it is', x1 - 2 * cw, y0 + 28); ctx.fillText('its heading', x1, y0 + 28);
    const ry = y0 + 44;   // the ringed bird's own row
    ctx.fillStyle = 'rgb(' + INK + ')'; ctx.beginPath(); if (ctx.roundRect) ctx.roundRect(x0 - (tight ? 5 : 12), ry - fs - 2.5, x1 - x0 + (tight ? 9 : 18), rh + 1, 6); else ctx.rect(x0 - 12, ry - fs - 2.5, x1 - x0 + 18, rh + 1); ctx.fill();
    if (!tight) {
      ctx.strokeStyle = 'rgba(' + INK + ',.75)'; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.arc(x0 - 27, ry - fs * .36, 7.5, 0, 6.2832); ctx.stroke();
      ctx.beginPath(); ctx.arc(x0 - 27, ry - fs * .36, 2.8, 0, 6.2832); ctx.fill();
    }
    ctx.font = '500 ' + fs + 'px "DM Mono", ui-monospace, Menlo, monospace';
    wall.rows.forEach((row, r) => { ctx.fillStyle = r ? 'rgb(' + INK + ')' : '#fefefe'; row.forEach((v, j) => ctx.fillText(v, x1 - (3 - j) * cw, ry + r * rh)); });
    if (wall.more) { ctx.font = '13px Newsreader, Georgia, serif'; ctx.fillStyle = 'rgb(' + INK + ')'; ctx.fillText('and ' + wall.more + ' more rows', x1, ry + wall.rows.length * rh + 2); }
    ctx.textAlign = 'left';
    ctx.restore();
    ctx.strokeStyle = 'rgb(' + INK + ')'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X, 0); ctx.lineTo(X, h); ctx.stroke();
  }
  function drawQuad(t) {
    const { w, h, dpr } = view, pw = w / 2, ph = h / 2;
    drawSkyBack(t, true);
    quad.forEach((f, m) => {
      const ox = (m % 2) * pw, oy = (m > 1 ? 1 : 0) * ph;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.save(); ctx.beginPath(); ctx.rect(ox, oy, pw, ph); ctx.clip();
      drawFlock(f, ox, oy, Math.max(6.4, 8.6 * f.s)); ctx.restore();
    });
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.strokeStyle = 'rgb(' + INK + ')'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(pw, 0); ctx.lineTo(pw, h); ctx.moveTo(0, ph); ctx.lineTo(w, ph); ctx.stroke();
  }
  const panes = $('#panes'), paneEls = [];
  quad.forEach((_, m) => {
    const d = document.createElement('div'); d.className = 'pane-tag'; d.style.left = (m % 2) * 50 + '%'; d.style.top = (m > 1 ? 50 : 0) + '%';
    d.innerHTML = '<b>Run ' + (m + 1) + '</b><span>heading together</span><em></em>';
    const bars = document.createElement('div'); bars.className = 'pane-bars'; bars.style.left = (m % 2) * 50 + '%'; bars.style.bottom = (m > 1 ? 0 : 50) + '%';
    bars.innerHTML = RULES.map((r) => '<div class="pb" data-rule="' + r + '"><svg viewBox="0 0 22 10" aria-hidden="true"><path d="M1 5h14" stroke="var(--c)" stroke-width="2.4" stroke-linecap="round"/><path d="M21 5l-7-4v8Z" fill="var(--c)"/></svg><span><i></i></span><b></b></div>').join('');
    bars.setAttribute('aria-label', 'Run ' + (m + 1) + ': how much of each rule is left');
    panes.appendChild(d); panes.appendChild(bars);
    paneEls.push({ pol: d.querySelector('em'), rows: RULES.map((r) => { const row = bars.querySelector('[data-rule="' + r + '"]'); return { row, fill: row.querySelector('i'), val: row.querySelector('b') }; }) });
  });
  function paintPanes() {
    paneEls.forEach((p, m) => {
      p.pol.textContent = DATA.together[m][state.deaf9].toFixed(2);   // measured, not the stand-in's own number
      const K = left(m, state.deaf9);
      RULES.forEach((r, j) => { const v = clamp(K[r], 0, 1), R = p.rows[j]; R.fill.style.width = (v * 100).toFixed(1) + '%'; R.val.textContent = pct(v); R.row.classList.toggle('gone', v < .1); });
    });
    grid8.dataset.deaf = state.deaf9;
  }
  /* the part 8 grid: rules across, runs down, and in every cell how much of the rule is left with round one deaf (1) and round two deaf (2) */
  const grid8 = $('#grid8'), tip8 = document.createElement('div'), IDLE8 = 'Hover over or tap a cell for its exact numbers.';
  (function buildGrid() {
    const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
    const lg = el('p', 'g8-legend'); lg.append(el('b', '', 'How much of each rule is left'));
    for (const [n, t] of [[1, 'with round one deaf'], [2, 'with round two deaf']]) { const sp = el('span'); sp.append(el('i', 'g8-n', n), ' ' + t); lg.append(sp); }
    grid8.append(lg, el('span'));
    for (const r of RULES) { const h = el('span', 'g8-head', NAME[r]); h.dataset.rule = r; grid8.append(h); }
    DATA.keeps.forEach((K, m) => {
      grid8.append(el('span', 'g8-run', 'run ' + (m + 1)));
      for (const r of RULES) {
        const cell = el('div', 'g8-cell'); cell.dataset.rule = r; cell.tabIndex = 0;
        const said = [];
        for (const n of [1, 2]) {
          const raw = K[n][r], v = clamp(raw, 0, 1), row = el('div', 'g8-b b' + n), track = el('span', 'g8-track'), fill = el('i');
          fill.style.width = (v * 100).toFixed(1) + '%'; track.append(fill); row.append(el('i', 'g8-n', n), track, el('b', '', pct(v))); cell.append(row);
          said.push(pct(v) + ' left with round ' + (n === 1 ? 'one' : 'two') + ' deaf' + (raw > 1.005 ? ' (measured ' + pct(raw) + ', shown as 100%)' : raw < -.005 ? ' (measured ' + pct(raw) + ', shown as 0%)' : ''));
        }
        cell.dataset.tip = 'Run ' + (m + 1) + ', ' + NAME[r] + ': ' + said.join(', ') + '.'; cell.setAttribute('aria-label', cell.dataset.tip);
        grid8.append(cell);
      }
    });
    tip8.className = 'g8-tip idle'; tip8.textContent = IDLE8; grid8.append(tip8);
    const say = (text) => { tip8.textContent = text || IDLE8; tip8.classList.toggle('idle', !text); };
    const show = (e) => { const c = e.target.closest('.g8-cell'); say(c ? c.dataset.tip : ''); };
    grid8.addEventListener('pointerover', show); grid8.addEventListener('focusin', show);
    grid8.addEventListener('pointerleave', () => say('')); grid8.addEventListener('focusout', () => say(''));
  })();

  /* ---------------- the notes: listening rewrites them ---------------- */
  const COLS = 8, ROWS = 3, CELLS = COLS * ROWS;
  const mats = (seed) => { const r = mulberry(seed); return Array.from({ length: CELLS }, () => Array.from({ length: 10 }, () => (r() * 2 - 1) * 1.5)); };
  const MAT = { 1: mats(21), 2: mats(22), raw1: mats(31), raw2: mats(32) };
  const pad = [null, { v: new Float32Array(CELLS), old: new Float32Array(CELLS), at: -9, heard: true }, { v: new Float32Array(CELLS), old: new Float32Array(CELLS), at: -9, heard: true }];
  pages.forEach((p, n) => { if (p) pad[n].ctx = p.querySelector('canvas').getContext('2d'); });
  function feats(hearing) {   // what our bird has to go on: itself, and (if it can hear) the others
    const f = main.b[main.follow], s = main.s; let cx = 0, cy = 0, hx = 0, hy = 0, nd = 1e9, nx = 0, ny = 0;
    for (const b of main.b) { if (b === f) continue; cx += b.x - f.x; cy += b.y - f.y; hx += b.hx; hy += b.hy; const d = Math.hypot(b.x - f.x, b.y - f.y); if (d < nd) { nd = d; nx = b.x - f.x; ny = b.y - f.y; } }
    const k = hearing ? 1 : 0, n = main.n - 1;
    return [f.hx, f.hy, k * cx / n / (70 * s), k * cy / n / (70 * s), k * hx / n, k * hy / n, k * nx / (45 * s), k * ny / (45 * s), k * (nd / (60 * s) - 1), .4];
  }
  function writePad(n, t) {
    const P = pad[n], deaf = deafNow() === n, M = MAT[(rawNow() ? 'raw' : '') + n];
    P.old.set(P.v); P.heard = !deaf; P.at = t + .3;   // the writing lands once the thought bubbles have carried it over
    if (deaf && n === 2) { P.v.set(pad[1].v); return; }   // a deaf round adds nothing: the page is just a copy of what was already there
    const F = feats(!deaf);
    for (let i = 0; i < CELLS; i++) { let a = 0; for (let j = 0; j < 10; j++) a += M[i][j] * F[j]; P.v[i] = Math.tanh(n === 2 ? .8 * pad[1].v[i] + .75 * a : a); }
  }
  function drawPads(t) {
    for (const n of [1, 2]) {
      const P = pad[n], c = P.ctx; c.setTransform(3, 0, 0, 3, 0, 0); c.clearRect(0, 0, 88, 32);
      for (let i = 0; i < CELLS; i++) {
        const col = i % COLS, row = (i / COLS) | 0, age = t - P.at - col * .04, shown = age > 0, v = shown ? P.v[i] : P.old[i];
        const fresh = shown && P.heard ? clamp(1 - age / .55, 0, 1) : 0;
        c.fillStyle = 'rgba(' + INK + ',' + (.06 + .84 * Math.pow((v + 1) / 2, 1.4)).toFixed(3) + ')';
        const g = fresh * 1.6, x = col * 11 + .5 - g / 2, y = row * 10.6 + .4 - g / 2, wd = 9.6 + g, ht = 9.2 + g;
        c.beginPath(); if (c.roundRect) c.roundRect(x, y, wd, ht, 2.4); else c.rect(x, y, wd, ht); c.fill();
      }
    }
  }
  const HOLD4 = [null, [.06, .4], [.4, .74], [.74, 1]];   // where each step of part 3 sits in the beat
  function runBeat(t) {
    if (state.scene < 4) return;
    if (reduce) { if (beat.n < 0) { beat.n = 0; writePad(1, t - 1); writePad(2, t - 1); } return; }
    const len = state.scene === 4 ? 5.6 : 3.2;   // one listen, listen, move per beat
    let n, ph;
    if (state.scene === 4 && state.hold4) {   // a held step plays on its own, over and over, with a breath between
      const [a, b] = HOLD4[state.hold4], seg = (b - a) * len, loop = seg + .7, lt = Math.max(0, t - beat.holdT0), k = Math.floor(lt / loop), u = lt - k * loop;
      n = 100000 + k; ph = a + Math.min(u, seg - .001) / len;
    } else { const tt = state.scene === 4 ? t - beat.origin : t; n = Math.floor(tt / len); ph = (tt - n * len) / len; }
    beat.ph = ph;
    if (n !== beat.n || len !== beat.len) { beat.n = n; beat.len = len; beat.f1 = beat.f2 = false; }
    /* a round that should already have happened (because a later step is held) is dated back to when it would have,
       so its lines are long gone and only its pages are left */
    if (!beat.f1 && ph >= .06) { beat.f1 = true; beat.on1 = deafNow() !== 1; beat.t1 = beat.on1 ? t - (ph - .06) * len : -9; writePad(1, t); if (state.page === 1) throwGuesses(t); }
    if (!beat.f2 && ph >= .4) { beat.f2 = true; beat.on2 = deafNow() !== 2; beat.t2 = beat.on2 ? t - (ph - .4) * len : -9; writePad(2, t); if (state.page === 2) throwGuesses(t); }
    const phase = ph < .06 ? 0 : ph < .4 ? 1 : ph < .74 ? 2 : 3;
    if (phase !== beat.phase) {
      beat.phase = phase;
      if (state.scene === 4) { paintCaption(); for (const k of [1, 2]) pages[k].classList.toggle('writing', phase === k); }
    }
  }

  /* ---------------- the big flocks: one to land on, one to leave on ----------------
     Neither has a sky of its own: both fly on the white of the page. The bottom one runs at full width,
     and is only started when the visitor is getting near it, so the page does not pay for both at once */
  const skyOn = { a: true, b: false }, murmN = $('#murmN');
  const phone = innerWidth < 900, flockSpeed = phone ? 2.6 : 2, light = phone ? { maxFps: 60, maxDpr: 1.5 } : {};
  const heroFlock = mountMurmuration($('#murm'), { sky: 'none', ground: false, autoFalcon: false, birdSize: 1.3, speed: flockSpeed, ...light, onCount: (n) => { murmN.textContent = n.toLocaleString('en-US'); } });
  { const fb = $('#falconBtn'), used = () => fb.classList.add('used');
    fb.addEventListener('click', () => { heroFlock.send(); used(); scatter(); }); $('#murm').addEventListener('pointerdown', () => { used(); scatter(); }); }
  let finaleFlock = null;
  $('#falconBtn2').addEventListener('click', () => { mountFinale(); finaleFlock.send(); });
  const mountFinale = () => { if (!finaleFlock) finaleFlock = mountMurmuration($('#murm2'), { sky: 'none', ground: false, autoFalcon: false, birdSize: 1.45, speed: flockSpeed, ...light, birds: phone ? 2600 : 'auto' }); };
  if ('IntersectionObserver' in window) {
    new IntersectionObserver((es) => { skyOn.a = es[es.length - 1].isIntersecting; }).observe($('#hero'));
    new IntersectionObserver((es) => { skyOn.b = es[es.length - 1].isIntersecting; }).observe($('#sky2host'));
    const nearEnd = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { mountFinale(); nearEnd.disconnect(); } }, { rootMargin: '1200px 0px' });
    nearEnd.observe($('#sky2host'));
  } else mountFinale();

  /* ---------------- strays: now and then a bird or two cuts across the edge of the page ---------------- */
  const mg = $('#margin'), gctx = mg.getContext('2d'), strays = [], feathers = [];
  let nextStray = 6, mgDirty = false;
  function sizeMargin() { const d = Math.min(2, devicePixelRatio || 1); mg.width = Math.round(innerWidth * d); mg.height = Math.round(innerHeight * d); mg._d = d; }
  function spawnStray(t) {
    const W = innerWidth, H = innerHeight, side = Math.random() < .5 ? -1 : 1, n = 1 + (Math.random() < .5) + (Math.random() < .25);
    const y0 = H * (.12 + .7 * Math.random()), dy = (Math.random() < .5 ? -1 : 1) * (160 + 220 * Math.random()), deep = 46 + 80 * Math.random();
    const xe = side < 0 ? -24 : W + 24, dur = 2.8 + 1.4 * Math.random();
    for (let i = 0; i < n; i++) strays.push({
      t0: t + i * .28, dur, p0: [xe, y0 + i * 14], p1: [xe - side * deep * 2.2, y0 + dy / 2 + i * 10], p2: [xe, y0 + dy + i * 14],
      sz: 6 + 2.5 * Math.random(), fp: Math.random() * 6.28
    });
  }
  function drawStrays(t) {
    if (!strays.length && !feathers.length && !mgDirty) return;
    const d = mg._d || 1; gctx.setTransform(1, 0, 0, 1, 0, 0); gctx.clearRect(0, 0, mg.width, mg.height); mgDirty = strays.length > 0 || feathers.length > 0;
    gctx.save();
    { const a = skywin.getBoundingClientRect(), c = capEl.getBoundingClientRect();   // the sky box (and its caption) stands in front of them: they pass behind it
      if (a.width && a.bottom > 0 && a.top < innerHeight) { const x0 = Math.min(a.left, c.left) - 6, y0 = a.top - 6, x1 = Math.max(a.right, c.right) + 6, y1 = Math.max(a.bottom, c.bottom) + 6;
        gctx.beginPath(); gctx.rect(0, 0, mg.width, mg.height); gctx.rect(x0 * d, y0 * d, (x1 - x0) * d, (y1 - y0) * d); gctx.clip('evenodd'); } }
    gctx.fillStyle = 'rgba(' + INK + ',.78)';
    for (let i = strays.length - 1; i >= 0; i--) {
      const s = strays[i], u = (t - s.t0) / s.dur; if (u >= 1) { strays.splice(i, 1); if (s.done) s.done(); continue; } if (u < 0) continue;
      const a = 1 - u, x = a * a * s.p0[0] + 2 * a * u * s.p1[0] + u * u * s.p2[0], y = a * a * s.p0[1] + 2 * a * u * s.p1[1] + u * u * s.p2[1];
      const vx = 2 * a * (s.p1[0] - s.p0[0]) + 2 * u * (s.p2[0] - s.p1[0]), vy = 2 * a * (s.p1[1] - s.p0[1]) + 2 * u * (s.p2[1] - s.p1[1]), vl = Math.hypot(vx, vy) || 1;
      flyer(gctx, d, x, y - (s.sy == null ? 0 : scrollY - s.sy), vx / vl, vy / vl, s.sz, .5 + .5 * Math.sin(t * 12 + s.fp));   // the edge strays stay put on screen; wire birds move with the page
    }
    for (let i = feathers.length - 1; i >= 0; i--) {   // a bird that leaves in a hurry sometimes drops a feather
      const f = feathers[i], a = t - f.t0; if (a > 3.4) { feathers.splice(i, 1); continue; }
      const sw = a * 2.2 + f.ph;
      gctx.setTransform(d, 0, 0, d, d * (f.x + Math.sin(sw) * 14), d * (f.y + a * 22 - (f.sy == null ? 0 : scrollY - f.sy))); gctx.rotate(Math.cos(sw) * .8);
      gctx.globalAlpha = Math.min(1, a * 5) * (1 - clamp(a - 2.4, 0, 1)) * .75;
      gctx.beginPath(); gctx.moveTo(0, -6.5); gctx.quadraticCurveTo(3.6, -1, 0, 6.5); gctx.quadraticCurveTo(-3.6, -1, 0, -6.5); gctx.fill();
      gctx.globalAlpha = 1;
    }
    gctx.restore();
  }

  /* ---------------- birds on wires: one strung under the top flock, and a few on the line above the notes ----------------
     Each is a small DOM bird that sits and now and then hops. Bring the pointer to one and it leaves: the sitting bird is
     hidden and a flying one (the same kind as the strays) carries on from where it sat, then another flies it home. */
  const perched = [];
  let sleepers = 0;
  const SLEEPER = "<svg class=\"sleeper\" viewBox=\"0 0 40 33\" aria-hidden=\"true\"> <path class=\"sl-ink\" d=\"M3.4 24.2 9.6 17.8l3.2 3.6-8.6 5.2c-1 .6-1.7-.5-.8-2.4Z\"/> <path class=\"sl-ink\" d=\"M8.2 18.6C9 12.4 14 8.2 20.6 8.2c2.3 0 4.3.5 6 1.5 1.7-2.6 4.7-3.5 7.3-2.4 2.9 1.2 4.2 4.5 3 7.4-.6 1.3-1.6 2.3-2.8 2.9C33.3 23 28.3 27 21.8 27 14.3 27 8.2 23.8 8.2 18.6Z\"/> <path class=\"sl-ink\" d=\"m36.3 9.3 3.5 1.9-3.8 1.2Z\"/> <path d=\"M18.5 26.5v5.3M23.5 26.5v5.3\" fill=\"none\" stroke=\"#0d1117\" stroke-width=\"1.5\" stroke-linecap=\"round\"/> <g class=\"sl-breath\"><use href=\"#blanket\"/></g> <g class=\"sl-cap\"> <clipPath id=\"sl-cap-clip\"><path d=\"M29.6 8.8C30.3 5.6 31 2.9 31 .5c2.4 2.4 4.6 5.8 5.7 9.4-2.2-2.2-4.8-2.6-7.1-1.1Z\"/></clipPath> <g clip-path=\"url(#sl-cap-clip)\"> <rect x=\"27\" y=\"-4\" width=\"12\" height=\"16\" fill=\"#b9a5e6\"/> <path d=\"M27 6.6 39 3.8M27 3.2 39 .4\" stroke=\"#efe9fb\" stroke-width=\"1.5\"/> </g> <path d=\"M29.6 8.8C30.3 5.6 31 2.9 31 .5c2.4 2.4 4.6 5.8 5.7 9.4-2.2-2.2-4.8-2.6-7.1-1.1Z\" fill=\"none\" stroke=\"#0d1117\" stroke-width=\".9\" stroke-linejoin=\"round\"/> <path d=\"M29.4 9c2.3-1.8 5.2-1.4 7.4 1\" fill=\"none\" stroke=\"#0d1117\" stroke-width=\"2.3\" stroke-linecap=\"round\"/> <path d=\"M29.4 9c2.3-1.8 5.2-1.4 7.4 1\" fill=\"none\" stroke=\"#efe9fb\" stroke-width=\"1.05\" stroke-linecap=\"round\"/> <circle cx=\"31\" cy=\"-.6\" r=\"1.75\" fill=\"#fefefe\" stroke=\"#0d1117\" stroke-width=\".9\"/> </g> <path class=\"sl-shut\" d=\"M31.4 10.6q1.4-1.5 2.8 0\" fill=\"none\" stroke=\"#fefefe\" stroke-width=\"1.1\" stroke-linecap=\"round\"/> <g class=\"sl-open\"><circle cx=\"32.8\" cy=\"10.2\" r=\"1.7\" fill=\"#fefefe\"/><circle cx=\"33.1\" cy=\"10.3\" r=\".8\" fill=\"#0d1117\"/></g> <ellipse class=\"sl-blush\" cx=\"34.9\" cy=\"12.6\" rx=\"1.3\" ry=\".75\" fill=\"#f19bb3\"/> <g class=\"sl-bub\"><circle cx=\"41.4\" cy=\"12.4\" r=\"2.5\" fill=\"rgba(196,226,250,.6)\" stroke=\"#6f97ba\" stroke-width=\".5\"/><circle cx=\"40.6\" cy=\"13.2\" r=\".55\" fill=\"#fefefe\"/></g> </svg>";
  const sleeperSVG = () => SLEEPER.replace(/sl-cap-clip/g, 'sl-cap-clip' + (++sleepers));   // each one needs its own clip id
  function perchBirds(host, spots, yAt) {
    for (const [f, off, w, mode] of spots) {
      const el = document.createElement('span'), h = w * .825, y = yAt(f);
      el.className = 'wb' + (mode ? ' ' + mode : '');
      el.style.setProperty('--w', w + 'px'); el.style.left = 'calc(' + (f * 100).toFixed(1) + '% + ' + off + 'px)';
      el.style.top = (mode === 'hang' ? y - h * .036 - 1 : y - h * .964 + 1).toFixed(1) + 'px';
      el.innerHTML = (mode === 'hang' ? sleeperSVG() : '<svg aria-hidden="true"><use href="#perch"/></svg>') + (mode === 'hang' ? '<span class="zz"><i>z</i><i>z</i><i>z</i></span>' : '');   // the one hanging upside down is asleep, and someone tucked it in
      el.firstChild.style.animationDelay = (-7 * Math.random()).toFixed(2) + 's';
      const b = { el, w, host, away: false, sleeper: mode === 'hang' };
      if (b.sleeper) wakeable(b);
      el.addEventListener('pointerenter', () => { if (b.sleeper) { el.classList.add('snore'); return; } flee(b, 0); for (const o of perched) if (o !== b && o.host === host && !o.away && Math.abs(o.el.offsetLeft - el.offsetLeft) < 90) kick(o.el, 'hop'); });
      host.appendChild(el); perched.push(b);
    }
  }
  function flee(b, delay, stay) {
    if (b.away || reduce || b.sleeper) return; b.away = true;
    setTimeout(() => {
      const r = b.el.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2, dir = Math.random() < .5 ? -1 : 1, t = performance.now() / 1000;
      b.el.classList.add('away');
      strays.push({ t0: t, dur: 1.5 + .6 * Math.random(), p0: [x, y], p1: [x + dir * 90, y - 150 - 80 * Math.random()], p2: [x + dir * (innerWidth * .35 + 200 * Math.random()), -40], sz: b.w * .3, fp: Math.random() * 6.28, sy: scrollY });
      if (Math.random() < .45) feathers.push({ x, y, t0: t, ph: Math.random() * 6.28, sy: scrollY });
      setTimeout(() => comeBack(b), (stay || 3000) + 2600 * Math.random());
    }, delay);
  }
  function comeBack(b) {
    const r = b.el.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
    const land = () => { b.el.classList.remove('away'); b.away = false; kick(b.el, 'hop'); };
    if (r.bottom < -40 || r.top > innerHeight + 40 || document.hidden) { land(); return; }   // nobody is looking: it is simply back
    const dir = Math.random() < .5 ? -1 : 1;
    strays.push({ t0: performance.now() / 1000, dur: 1.4 + .5 * Math.random(), p0: [x + dir * (innerWidth * .3 + 160 * Math.random()), -40], p1: [x + dir * 120, y - 180], p2: [x, y - 3], sz: b.w * .3, fp: Math.random() * 6.28, sy: scrollY, done: land });   // its path is fixed to the page, so a scroll carries it along to the wire
  }
  /* poke the sleeping bird and it wakes up just long enough to complain */
  const GRUMPS = ['five more minutes', 'I was dreaming about round two', 'shh', 'is the falcon gone', 'line up... zzz', 'do not deafen me', 'it is warm in here', 'go read part 3'];
  let grumpAt = 0;
  function wakeable(b) {
    const g = document.createElement('span'); g.className = 'grump'; g.setAttribute('aria-hidden', 'true'); b.el.appendChild(g);
    let timer = 0;
    b.el.addEventListener('pointerleave', () => b.el.classList.remove('snore'));   // it snores louder while you hover over it
    b.el.addEventListener('click', () => {
      clearTimeout(timer); g.textContent = GRUMPS[grumpAt++ % GRUMPS.length];
      b.el.classList.remove('woke'); void b.el.offsetWidth; b.el.classList.add('woke');
      g.classList.remove('left'); if (g.getBoundingClientRect().right > document.documentElement.clientWidth - 6) g.classList.add('left');   // no room on the right: it complains to the left
      g.classList.add('on');
      timer = setTimeout(() => { g.classList.remove('on'); b.el.classList.remove('woke'); }, 2400);
    });
  }
  const wire1 = $('#wire1'), hoodBirds = $('#hoodBirds');
  function scatter() { perched.filter((b) => b.host === wire1).forEach((b, i) => flee(b, 500 + 110 * i + 240 * Math.random(), 5200)); }   // a falcon is about
  wire1.innerHTML = '<svg class="wire-l" viewBox="0 0 100 56" preserveAspectRatio="none"><path d="M0 10Q50 46 100 10"/></svg>';
  { const heroT = $('.hero-t'), home = wire1.nextElementSibling, place = () => { if (isWide()) home.before(wire1); else heroT.before(wire1); };
    place(); matchMedia('(min-width:900px)').addEventListener('change', place); }
  const roomy = innerWidth >= 700;
  perchBirds(wire1, roomy ? [[.13, 0, 38], [.47, -46, 42], [.47, -3, 31, 'flip'], [.6, 0, 46, 'hang'], [.72, -44, 40, 'flip'], [.72, -3, 37], [.72, 31, 27], [.89, 0, 39, 'flip']] : [[.1, 0, 30], [.33, 0, 32], [.42, 0, 24, 'flip'], [.6, 0, 38, 'hang'], [.8, 0, 31, 'flip'], [.91, 0, 27]], (f) => 10 + 72 * f * (1 - f));
  perchBirds(hoodBirds, roomy ? [[.05, 0, 37], [.5, 0, 42, 'hang'], [.84, -20, 39], [.84, 18, 30, 'flip'], [.95, 0, 35, 'flip']] : [[.08, 0, 31], [.42, 0, 35, 'hang'], [.72, -16, 32], [.72, 14, 25, 'flip']], () => -1.5);
  $$('.qa > div').forEach((el, i) => {   // and one or two on the lines over the three answers
    const host = document.createElement('div'); host.className = 'perches'; host.setAttribute('aria-hidden', 'true'); el.prepend(host);
    perchBirds(host, [[[.88, 0, 36]], [], [[.62, -19, 37, 'flip'], [.62, 19, 28]]][i] || [], () => -1.5);
  });
  /* the reader dozes off in its own parts if nobody touches anything for a while, and wakes with a start */
  { const rz = document.createElement('span'); rz.className = 'rz'; rz.setAttribute('aria-hidden', 'true'); rz.innerHTML = '<i>z</i><i>z</i><i>z</i>'; readerEl.appendChild(rz);
    let lastAct = performance.now();
    const act = () => { lastAct = performance.now(); if (readerEl.classList.contains('dozing')) { readerEl.classList.remove('dozing'); hopReader(); } };
    for (const ev of ['scroll', 'pointerdown', 'pointermove', 'keydown', 'wheel', 'touchstart']) addEventListener(ev, act, { passive: true });
    if (!reduce) setInterval(() => {
      if (document.hidden) return;
      if (state.scene >= 5 && state.scene <= 7 && performance.now() - lastAct > 35000) readerEl.classList.add('dozing');
    }, 1000);
  }
  /* type "falcon" anywhere: it goes for whichever big flock is on screen, and if neither is, every bird in sight takes off */
  { let typed = '';
    addEventListener('keydown', (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey || e.key.length !== 1) return;
      typed = (typed + e.key.toLowerCase()).slice(-6);
      if (typed !== 'falcon') return;
      typed = '';
      const inView = (el) => { const r = el.getBoundingClientRect(); return r.bottom > innerHeight * .15 && r.top < innerHeight * .85; };
      if (inView($('#murm'))) { heroFlock.send(); scatter(); $('#falconBtn').classList.add('used'); }
      else if (inView($('#sky2host'))) { mountFinale(); finaleFlock.send(); scatterPalm(); }
      else for (const b of perched) { const r = b.el.getBoundingClientRect(); if (r.bottom > 0 && r.top < innerHeight) flee(b, 300 * Math.random(), 4000); }
    });
  }
  /* the palm at the end, full of starlings */
  const palmEl = $('#palm'), palmBirds = $('#palmBirds'), PALM_SPOTS = [[118.1, 76.7], [94.7, 91.7], [138.5, 52.1], [187.7, 54.2], [202.5, 77.9], [229.1, 95.1], [121.4, 100.7], [205.7, 106.2], [128.3, 58.7]];
  { const bw = Math.round(palmEl.getBoundingClientRect().width * .066) || 16, before = perched.length;
    perchBirds(palmBirds, PALM_SPOTS.map(([x], i) => [x / 280, 0, bw + (i % 3) - 1, i % 2 ? 'flip' : '']), () => 0);
    perched.slice(before).forEach((b, i) => { const h = b.w * .825; b.el.style.top = 'calc(' + (PALM_SPOTS[i][1] / 300 * 100).toFixed(2) + '% - ' + (h * .964 - 1).toFixed(1) + 'px)'; b.palm = true; });
  }
  function scatterPalm() { perched.filter((b) => b.palm).forEach((b, i) => flee(b, 250 + 90 * i + 200 * Math.random(), 4200)); }
  $('#falconBtn2').addEventListener('click', scatterPalm);
  $('#murm2').addEventListener('pointerdown', scatterPalm);
  /* and they yell: a little burst of lines over one of them, every so often, while the tree is on screen */
  if (!reduce) setInterval(() => {
    if (document.hidden || !skyOn.b) return;
    const here = perched.filter((b) => b.palm && !b.away); if (!here.length) return;
    for (let n = Math.random() < .3 ? 2 : 1; n > 0; n--) {
      const b = here[Math.floor(Math.random() * here.length)], flip = b.el.classList.contains('flip');
      const c = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); c.setAttribute('class', 'chirp'); c.setAttribute('viewBox', '0 0 16 12');
      c.innerHTML = flip ? '<path d="M9 9 4 7M9 6 5 2.5M11 5 10.5 .5"/>' : '<path d="M7 9l5-2M7 6l4-3.5M5 5l.5-4.5"/>';
      c.style.left = (b.el.offsetLeft + (flip ? -b.w * .55 : b.w * .35)).toFixed(1) + 'px'; c.style.top = (b.el.offsetTop - 9).toFixed(1) + 'px';
      palmBirds.appendChild(c); setTimeout(() => c.remove(), 1000);
      if (Math.random() < .35) kick(b.el, 'hop');
    }
  }, 650);
  if (!reduce) setInterval(() => {   // now and then one of them hops
    if (document.hidden) return;
    const b = perched[Math.floor(Math.random() * perched.length)], r = b.el.getBoundingClientRect();
    if (!b.away && r.bottom > 0 && r.top < innerHeight) kick(b.el, 'hop');
  }, 1700);

  /* ---------------- the loop ---------------- */
  let last = 0, gaugeAt = 0;
  function frame(now) {
    if (innerWidth < 900 && last && now - last < 14.5) { requestAnimationFrame(frame); return; }   // phones: at most 60 frames a second here too
    /* a frame's timestamp can be earlier than the previous one's clock reading; never step time backwards */
    const dt = last ? clamp((now - last) / 1000, 0, .05) : 0; last = now;
    const t = now / 1000;
    clock.dt = reduce ? 0 : t - clock.t > .1 ? .016 : t - clock.t; clock.t = t;
    tod = reduce ? todTo : tod + (todTo - tod) * Math.min(1, dt * 6);
    gust.now += (gust.v - gust.now) * Math.min(1, dt * 5); gust.v *= Math.max(0, 1 - dt * 4);
    gust.slow += (gust.now - gust.slow) * Math.min(1, dt * 1.3);   // only a change of pace leans on the birds, so a long steady scroll doesn't pin them to the top
    { const g = reduce ? 0 : -(gust.now - gust.slow) * 2.2; main.gust = fifty.gust = toy.gust = g; for (const q of quad) q.gust = g; }
    if (view.w && skyShown) {
      const quadOn = state.scene === 9;
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, cv.width, cv.height);
      if (quadOn) { if (dt > 0) quad.forEach((f) => f.step(dt, t)); drawQuad(t); }
      else {
        const f = flock();
        state.split += ((state.scene === 3 && state.its ? .5 : 1) - state.split) * Math.min(1, (dt || .016) * 4.5);
        if (state.scene === 3) for (const g of [main, fifty]) {   // the flock keeps to the left pane, a little smaller, so the right pane can show the very same birds
          const ww = view.w * state.split, sc = view.s * (g === fifty ? .6 : 1) * (.72 + .56 * (state.split - .5));
          if (Math.abs(g.w - ww) > .5 || Math.abs(g.s - sc) > .004) g.fit(ww, view.H, sc);
          if (g !== f) { if (dt > 0) g.step(dt, t); g.record(t, dt, LAG); }
        }
        if (state.scene === 2) { if (dt > 0) main.step(dt, t); main.record(t, dt, LAG); }   // warming up for the next part
        setSafe(f);
        if (dt > 0) f.step(dt, t);
        if (state.scene === 3) f.record(t, dt, LAG);
        runBeat(t); drawSingle(t, dt || .016); if (state.scene >= 4) drawPads(t);
      }
      if (replay7.on) applyT();
      if (now - gaugeAt > 110) {
        gaugeAt = now;
        if (!quadOn) {
          const p = clamp(flock().pol || 0, 0, 1), g = $('.gauge'), low = p < .6;
          $('#polBar').style.width = (p * 100).toFixed(0) + '%'; $('#polV').textContent = p.toFixed(2);
          if (low !== g.classList.contains('low')) { g.classList.toggle('low', low); if (low) kick(g, 'pop'); }   // the flock just fell apart: say so
        }
      }
    }
    if (!reduce) {
      if (t > nextStray && strays.length === 0 && !document.hidden && !skyOn.a && !skyOn.b && innerWidth >= 900) { spawnStray(t); nextStray = t + 2.4 + Math.random() * 3.4; }
      drawStrays(t);
    }
    requestAnimationFrame(frame);
  }

  if ('IntersectionObserver' in window && !reduce) {
    body.classList.add('mk');
    const io = new IntersectionObserver((es) => { for (const e of es) if (e.isIntersecting) { e.target.classList.add('lit'); io.unobserve(e.target); } }, { rootMargin: '0px 0px -16% 0px', threshold: .5 });
    $$('.reel strong').forEach((el) => io.observe(el));
  }
  sizeSky(); sizeMargin(); paint(); pickScene();
  addEventListener('resize', () => { onScroll(); measure(); sizeMargin(); });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { measure(); measureSafe(); onScroll(); });
  requestAnimationFrame(frame);
})();
