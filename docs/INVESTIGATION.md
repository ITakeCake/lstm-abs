# LSTM-ABS investigation log (2026-09-01 to 2026-09-02)

Working notes kept during the investigation, lightly edited. One specific project: **learning an ABS controller from my existing 25GB telemetry
corpus.** It is NOT about DynamicABS development, the PPO/SAC training runs, or the
brake-test tooling except where those intersect.

---

## 1. THE GOAL (do not drift from this)

Build a working, machine-learning-based ABS **using the 25GB of telemetry that
already exists on disk.**

Two reasons this framing is non-negotiable:

1. **The point is a concrete, genuinely working ML result.** The story is
   *"an ABS trained from 25GB of my own logged 2kHz vehicle telemetry."* A working result
   matters more than algorithmic sophistication.
2. **Do not collect new data.** BeamNG.tech is reserved for the separate PPO/SAC
   work. The whole exercise is a showcase of *reusing existing data*. If you
   find yourself planning a data-collection run, you have drifted off-goal. Stop and
   re-read this section.

Corollaries adopted for the project:
- **No teacher / no expert demonstrations required.** Weight each logged action by how
  good its measured outcome was; clone the good ones, down-weight the bad. The
  "teacher" is just "whatever worked."
- **Straight-line testing only, in BeamNG.drive.** Report straight-line stopping
  results only. Do not claim cornering performance.
- **`.tech` is off-limits.** Never kill `BeamNG.tech.exe` / `BeamNG.tech.x64.exe`.
  They belong to the RL side. `.drive` is yours to launch and close.

---

## 2. CURRENT STATUS IN ONE LINE

Actuation is verified correct (hardcoded 1.0 output = 1.00g locked-wheel stop); the
0.22g result was the model, caused by a corpus trap, and filtered retrains now stop at
**1.13g / 32.3m at 60mph (`baseline_dyn`, DynamicABS cloned from the 11GB telemetry
logs) vs DynamicABS 1.21g / 30.3m**; every net still lets wheels lock late in the stop.
See section 7.

## 3. THE DATA

### Master corpus (read-only, never modify)
`Desktop\Beamng_AI\ml\SpeedLSTM\recorded_data_2khz\`
- Folders `S1 S2 S3 S4 S5 SAC` — 2,955 CSV episodes, 99 columns, true 2kHz
  (`dt=0.0005`).
- `DriverBoy\` is **100Hz — do not mix it in.**
- 4 vehicle models: `etk800`, `etkc`, `sbr`, `vivace`. (An older note claiming "~20
  car models" is wrong; it was measured as 4.)
- `safety_variant` values: `stock_abs`, `stock_full`, `abs_1f`, `no_safety` (S1-S5),
  and `no_abs` (all 1,158 SAC files).
- **No DynamicABS episodes exist in this corpus.** That is why cloning DynamicABS
  specifically is not an option here.

### Cleaned clone (what training reads)
`Desktop\Beamng_AI\ml\SpeedLSTM\recorded_data_2khz_clean\`
- 2,950 parquet files, 37.3M rows, 25GB → 2.3GB, float32.
- Built by `clone_corpus_clean.py`. Masters are only ever read.

### Column contract (agreed with I — respect it)
**17 legal model inputs** (a real ECU could have these):
```
ws_fr ws_fl ws_rr ws_rl sensor_x sensor_y sensor_z yaw_rate
steer_input steer_angle front_wheel_angle brake_input throttle_input
roll_angle pitch_angle rpm gear
```
**4 targets** (the action being learned): `brake_torque_fr/fl/rr/rl`

**Grade-only columns** — legal for computing training weights offline, **illegal as
model inputs**: `speed_velo`, `wheelbase`, `damage`.

Everything else in the 99 columns (slip, contact, downforce, wheel_velo, brake_temp,
velo_*, airspeed, friction/diff columns) is "cheat" data and is excluded.

### Known data-quality traps (verified, do not re-discover)
- **`grip_pattern` labels are largely fictional.** Measured over 300 episodes:
  `black_ice` averages 0.908g, `sand` 1.009g, `dry_asphalt` 0.936g. The friction
  overrides mostly never applied. Genuine low-mu runs exist but are scattered *inside*
  every label (ice ranges 0.045–1.152g). **Never group or rank by `grip_pattern`.**
- **`sensor_y` is POSITIVE during braking** in this corpus (+6 to +13, matching the
  speed-derivative magnitude). Deceleration must be derived from the speed derivative,
  not from a sign assumption about the IMU. Getting this backwards silently zeroes
  every g-based training weight — it already happened once and invalidated a full
  experiment.
- Within braking segments, `brake_input` is near-constant ~1.0 (std 0.0148) and
  `steer_input`/`steer_angle` are ~constant zero. Their normalization std gets floored
  to 1.0. Treat these as low-information channels.

---

## 4. THE METHOD

Reward-weighted offline learning (behaviour cloning with per-tick weights). No RL
rollouts, no simulator in the training loop.

For each tick, the outcome is what happened in the **~200ms after it**:
- deceleration achieved (from the speed derivative), and
- whether yaw matched what the steering angle asked for (bicycle model, deadzoned).

That outcome is **percentile-ranked within its own episode** (one episode = one
car/surface/speed), which asks "was this action good *for this situation*" without
trusting any label. Rank maps to a weight in `[0.1, 1.0]` — never 0, so nothing is
deleted.

Targets are normalized to a **0–1 fraction of that car's max brake torque**, so four
different cars share one model. Per-car maxima measured empirically:
```
etk800  [3100, 3100, 1487.5, 1487.5]   etkc   [2800, 2800, 1200, 1200]
sbr     [1344, 1344, 3000, 3000]       vivace [3000, 3000,  980,  980]
```

### The six lenses (the lens experiment)
Identical architecture, differing only in how ticks are weighted. Any lens may combine
multiple signals; `avg_g_window` is the one the experiment required.
| lens | weighting |
|---|---|
| `baseline` | uniform (control) |
| `avg_g_window` | forward-window mean decel, episode-ranked **[required]** |
| `peak_g` | forward-window peak decel, episode-ranked |
| `yaw_intent` | yaw-vs-steering-intent match, episode-ranked |
| `combined` | avg_g_window × yaw_intent |
| `curation` | top-quartile episodes only, uniform weight (data selection) |

Plus ablation variants `peak_g_norpm` / `baseline_norpm` (rpm+gear zeroed).

**Important caveat about the offline metric:** val MAE measures agreement with the
*logged* action. A lens that correctly down-weights bad actions should score *worse*
on it while being a *better* controller. Val MAE therefore **cannot rank the lenses** —
only in-game testing can. Do not declare a winner from MAE.

---

## 5. FILES

**Since 2026-09-02 evening the working folder is `Desktop\lstm-abs\` (git, private GitHub
ITakeCake/lstm-abs).** Scripts below live there; the data folders stay in
`Desktop\Beamng_AI\ml\SpeedLSTM\` and are junctioned into the repo. Old script copies in
SpeedLSTM are parked with `.MOVED_TO_REPO_2026-09-02`. `deploy_mod.py` / `pull_mod.py`
move the mod between repo and game userpath.

**Restructure (2026-09-02 night):** one jbeam part `etk_DSE_LSTM_ABS` + one `LSTMABS.pc`; the
active net is chosen at runtime (`lstmabs_lens.txt` in the userpath at init, or
`controller.getController('lstm_abs').setLens(name)` live). Models live in
`experiments/<name>/{model.pt,meta.json,runs.csv}`; Lua weight modules are generated into
the mod by `export_lstm_abs.py` and are not in git. Per-lens jbeam parts and .pc files are
parked (`configs/_parked_per_lens_2026-09-02`, userpath `_parked_LSTMABS_per_lens_2026-09-02`).

| file | purpose |
|---|---|
| `clone_corpus_clean.py` | 99-col masters → 21-col parquet clone (12 workers) |
| `train_lenses_corpus.py` | builds cache, trains the lenses. `--rebuild-cache --epochs N --lenses a,b --zero-cols rpm,gear --tag _x` |
| `export_lstm_abs.py` | torch `.pt` → Lua string-blob weights + parity gate. `--lens all --install` |
| `test_lenses_drive.py` | in-game comparison harness (file-channel, no beamngpy) |
| `lens_models_corpus\` | 8 trained `.pt` + emitted `.lua` weight modules |
| `recorded_data_2khz_clean\_cache_corpus.npz` | preprocessed episode cache |
| `lens_drive_results_*.csv` | in-game results |

Deployed mod (BeamNG.drive userpath):
`...\BeamNG.drive\current\mods\Unpacked\lstm_abs\`
- `lua\vehicle\controller\lstm_abs.lua` — the controller
- `lua\vehicle\controller\lstmabs_weights_<lens>.lua` — 8 weight modules
- `vehicles\etk800\lstm_abs.jbeam` — 8 parts in the `etk_DSE_ABS` slot
- Configs `...\vehicles\etk800\LSTMABS_<lens>.pc`

---

## 6. HOW THE DEPLOYMENT WORKS

Architecture: 2-layer LSTM, hidden 64, 17 inputs → 4 outputs, sigmoid → 0–1 fraction
of car max torque. ~54.8k numbers, which is why weights use the **string-blob format**
(LuaJIT has a 65,536 constants-per-prototype cap; inline tables are unsafe at this
size).

- Runs at **200Hz** via a self-subdividing accumulator (matches the 10× training
  downsample). State is carried between ticks and zeroed at each brake engage.
- Engages at driver brake > 0.9 and speed > 8 m/s; disengages on release or < 0.5 m/s.
- Actuation: sets `wheelRotators[i].brakeTorque = origCapacity[i] * frac` and pins
  `input.brake = 1.0`, re-asserted every physics step. This is the capacity-scaling
  contract copied from MDABS (`desiredBrakingTorque = capacity * pedal`).
- Safety: no brake writes at all if weights fail to load, NaN→0, obs clipped ±10σ,
  2-tick warmup at 5%, floor 0.01, full capacity restore on disengage.
- Telemetry: `electrics.values.lstmabs_active / _fr / _fl / _rr / _rl /
  _engagements / _lens`.
- Debug: prints `LSTM_ABS DBG raw=... norm=... cmd=...` at tick 15 of each engagement,
  and a `SELFTEST` line at load.

**Wheel index mapping** (verified in-game): `wr[0]=RR wr[1]=RL wr[2]=FR wr[3]=FL`.

---

## 7. WHERE THINGS STAND (2026-09-02) — START HERE

### The fork in the old section 7 is resolved
`FORCE_CMD = 1.0` in `lstm_abs.lua` gave **1.0017g / 36.6m** at 60mph with the fronts
locked by tick 60 and ~2900/1450 Nm reaching the wheels (`LSTM_TRACE` lines in
beamng.log). `hasABS=0`, `absActive=nil`: stock ABS is not fighting the net.
**Actuation works. Every earlier 0.22g was the model.**

### Root cause of the 0.22g (measured, not guessed)
Braking ticks with low decel and no wheel-speed spread, which is exactly what the net
sees at engage (5% warmup), carry a mean logged front torque of **0.07** in the corpus,
and 140k of those 155k ticks come from the `abs_1f` variant (early controller that sat
near zero torque with the pedal down). The net learned "pedal on, no decel, no slip ->
release", which is self-fulfilling in closed loop: cmd goes 0.30 -> 0.003 in 20 ticks
and decel never builds. Bimodal per engagement (some runs escaped to 0.78g).
The offline probe `onset_probe.py` reproduces the in-game numbers exactly, so it is a
faithful pre-game check for the trap.

### What was changed
- `train_lenses_corpus.py`: new episode filters `--variants`, `--exclude-variants`,
  `--models`, `--min-peak-g` (measured forward-avg decel, never `grip_pattern`).
- `lstm_abs.lua`: `FORCE_CMD` (nil normally) and `TRACE_EVERY` diagnostics; rear
  pedal-split fix (rear capacity 1700 * split 0.875 = the 1487.5 the corpus logs, so
  rear cmd is divided by the split gain). Backup: `.BAK_pre-controltest_2026-09-02`.
- New lenses installed (jbeam parts + `LSTMABS_<lens>.pc`), all parity PASS.
- Probes: `onset_probe.py` (trap), `lock_reflex_probe.py` (release on simulated
  lock), `replay_ablate.py` (replay captured `LSTM_OBS` lines, per-channel ablation),
  `control_test_drive.py` (one-config runner with trace capture).

### Results, 60mph straight line, .drive smallgrid, standard 2kHz metric
| config | filter | avg g | dist | behaviour |
|---|---|---|---|---|
| old lenses (baseline, avg_g_window, ...) | none | 0.22 (bimodal, 0.78 once) | 160m | trap, no brakes |
| old `curation` | top quartile | 1.07 | 34m | modulates then locks |
| `baseline_hg` | minus abs_1f, peak >= 0.9g | 1.02 | 36m | ramps then locks |
| `avg_g_window_hg` | same | 1.03 | 35.5m | locks; 80mph 1.01g / 64m |
| `baseline_stock` | stock_abs+stock_full, 4 cars | 0.89 | 41m | modulates, gives up < 12 m/s |
| `avg_g_window_stock` | same | 1.03 | 35.5m | locks |
| `baseline_stocknorpm` | stock, rpm+gear zeroed | 0.99 | 37m | between |
| **`baseline_stock800`** | **stock, etk800 only** | **1.07, 1.07, 1.06** | **34m** | modulates ~1s then locks; 80mph 0.99g / 66m |
| `baseline_stock16` | stock, 16 epochs | 0.75 | 49m | overfit, low onset |
| FORCE_CMD 1.0 (control) | n/a | 1.00 | 36.6m | locked wheels |
| DynamicABS reference (same session) | n/a | 1.20, 1.21 | 30.3m | real ABS |

### Second data source: DynamicABS telemetry logs (added 2026-09-02, later the same day)
`.drive	elemetry\DynamicABS_Run_*.csv` = 2,237 runs, 11GB, true 2kHz, 1,693 with the
`*_DesiredBrake` target (same `desiredBrakingTorque` field the corpus logs). Cloned by
`clone_telemetry_dynabs.py` into `recorded_data_2khz_clean_tel\TEL_<variant>\`
(variants `dyn_abs`, `stock_tel`, `abs_1f_tel`, ...). Sensors there are `gx2/gy2/gz2`
(smoothed), so the ckpt carries `sensor_src="gy2"` and the controller reads those
when set; roll/pitch/gear/sensor_z are zeroed. Train with
`--roots <corpus>,<tel> --cache recorded_data_2khz_clean_tel\_cache_all.npz`.
GUI `BrakeTestResults_Straight.csv` joins to these logs by `BrakeTestRunID` (263 runs)
and is a grade only, never an input.

| config (etk800 only, peak >= 0.9g) | data | avg g @60 | dist | note |
|---|---|---|---|---|
| **`baseline_dyn`** | 1,391 DynamicABS runs | **1.135, 1.126, 1.135, 1.115** | **32.3m** | modulates ~1s then locks < 10 m/s |
| `baseline_dynnoimu` | same, IMU zeroed | 0.37 | 100m | decel channel is essential |
| `baseline_dynstock2` | + 116 stock-ABS telemetry runs | 0.25 | 147m | onset trap again |
| `baseline_dynstocknoabs2` | + corpus no_abs/no_safety | 0.80, 0.84 | 44m | blend |

Mixing controllers with no controller-identity input is fragile: 116 stock runs on
top of 1,391 DynamicABS runs flipped the onset behaviour back into the release trap.

### Speed-band grader (`band_g`, `band_gy` lenses, 2026-09-02 evening)
The idea: grade every run per speed band (`BAND_MPH`, `BAND_OVERLAP` constants at
the top of `train_lenses_corpus.py`; first run 5 mph, no overlap), rank each band
against all other runs of the same car, weight the ticks by their band's rank
(x yaw score for `band_gy`). Grades use true speed only: telemetry `Airspeed`, never
`CarSpeed` (= wheelspeed). `test_band_lens.py` covers the math.

| net | pool | avg g @60 | dist |
|---|---|---|---|
| `baseline_dyn` | DynamicABS only, uniform | 1.13, 1.11 | 32.4m |
| `band_g_dyn` | DynamicABS only, band-ranked | 1.06, 1.06 | 34.4m |
| **`band_g_mix`** | **dyn + stock + no-ABS, band-ranked** | **1.16, 1.13** | **31.6m** |
| `band_gy_mix` | same, x yaw | 1.12, 1.12 | 32.6m |
| (uniform mix, for reference) | dyn + stock + no-ABS | 0.80, 0.84 | 44m |

Reading: on one controller the grader only removes data (worse); on a mixed pool it
picks the best controller per band and beats plain DynamicABS cloning. Best single run
so far: `band_g_mix` 1.1585g / 31.6m vs DynamicABS 1.21g / 30.3m.

### Two lessons that shape the next step
1. **`avg_g_window` (the required lens) cannot learn to release.** Releasing is what
   lowers forward decel for the next 200ms, so within-episode ranking down-weights
   exactly the ticks where ABS lets go. Every avg_g lens locks the wheels. Report it as
   the experiment's finding; do not tune it into an ABS.
2. **Pure behaviour cloning breaks at lock.** Stock ABS almost never lets a wheel fully
   lock, so the corpus has ~no slip=1.0 ticks; once a cloned net lets one lock, its
   output there is extrapolation and it holds high. The shared 4-car clone also decays
   with speed (no car-identity input, regresses to the across-car mean); etk800-only
   fixed that. Three data-side fixes each helped; the remaining gap to DynamicABS is
   structural, not a bug.

### Next action (design decision to make first)
Offline relabeling: stock ABS's control law is public (`wheels.lua updateABSCoef`, a
slip PID on `2/v + slipRatioTarget`). Every corpus tick can be relabeled with "what
stock ABS would have commanded here" using the grade-only `speed_velo`, giving a
consistent teacher for every state including locked wheels, with no new data. That is
still "trained from the 25GB corpus". Alternative: a disclosed 2-line slip guard on top
of `baseline_stock800`. Pick one before building.

## 8. ENVIRONMENT GOTCHAS

- **beamngpy version wall.** `.drive` 0.39.4 speaks protocol v1.26 → needs beamngpy
  1.35.1 (isolated venv at `ml\SpeedLSTM\.driveenv`). Global beamngpy is 1.34.1
  (v1.24) and is **required by `.tech` 0.37.6 — do not upgrade it globally.**
- `test_lenses_drive.py` deliberately uses the **file channel**
  (`brake_cmd.txt` / `brake_done.txt` in the `.drive` userpath), not beamngpy, because
  the beamngpy connection proved fragile. Prefer this.
- `.drive` can only be connected with `launch=True`, or driven purely by the file
  channel.
- `.tech` **must** launch headless: `bng.open(None,'-headless','-gfx','null','-no-sound',launch=True)`.
  `-nographics` is the wrong flag and silently half-works. Windowed `.tech` 0.37.6
  hangs on `scenario.load`.
- **vlua can only write relative paths** — pass a bare filename to any recorder and
  move the file from the userpath afterwards.
- `math.tanh` is missing in some LuaJIT builds; use the `fastTanh` helper.
- Brake-test metric: `setTestParams(brakeMph, recordMph, coastMph)` — **brake first,
  record second**. Correct call is `(mph+2, mph, 2)`.
- The parity harness compares emitted-text→numpy vs torch and **does not execute
  Lua**. The in-game `SELFTEST` is what actually validates the Lua; keep using it.

---

## 9. STYLE

- Code comments: minimum necessary, ≤15 words, one line, never third person, never
  reference the conversation or a person. Applies to any subagent briefings too.
- camelCase for variables/functions where the language allows.
- No em dashes in prose. No "it's not X, it's Y" constructions.
- Never delete source files — park with a `.BAK_<reason>_<date>` suffix and ask.
- GitHub-published work carries no AI attribution; author is `ITakeCake`.
