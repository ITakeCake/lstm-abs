-- lstm_abs.lua - runs a lens-trained LSTM as the vehicle's ABS.
-- Net outputs per-wheel brake torque as a 0-1 fraction of that car's max, which
-- maps onto the stock pipeline's capacity scaling (desiredBrakingTorque =
-- capacity * pedal).
--
-- Runs at 200Hz (matches the 10x training downsample), state carried between
-- ticks and zeroed at each brake engage so the net starts near the regime its
-- fixed-length training windows came from.

local M = {}
M.type = "auxiliary"
M.relevantDevice = nil

local W = nil
local weightsOk = false
local lensName = "baseline"

local TICK_STEP = 1 / 200
local timeAccum = 0

local ENGAGE_BRAKE = 0.9
local ENGAGE_SPEED = 8.0    -- m/s
local RELEASE_SPEED = 0.5
local WARMUP_TICKS = 2
local WARMUP_FRAC = 0.05
local CMD_FLOOR = 0.01
local FORCE_CMD = nil      -- nil = run the net; constant = actuation control test
local TRACE_EVERY = 10     -- 200Hz ticks per trace line, 0 disables

local engaged = false
local engageCount = 0
local warmup = 0
local lastWrittenBrake = -1
local origCapacity = {0, 0, 0, 0}
local invSplit = {1, 1, 1, 1}
local wheelCount = 0

local mean, std = nil, nil
local layers = nil
local headW, headB = nil, nil
local hBuf, cBuf, xBuf, gBuf = nil, nil, nil, nil
local cmd = {0, 0, 0, 0}
local obs = nil

local mabs, mexp, mmin, mmax = math.abs, math.exp, math.min, math.max
local macos = math.acos

local fwNode1, fwNode2 = nil, nil

-- same derivation as the recorder, so the channel matches training
local function frontWheelAngle()
  if not fwNode1 then return 0 end
  local cosA = obj:nodeVecPlanarCosRightForward(fwNode1, fwNode2)
  if not cosA then return 0 end
  local a = macos(mmin(1, mmax(-1, cosA)))
  if (electrics.values.steering_input or 0) < 0 then a = -a end
  return a
end

-- LuaJIT builds do not all ship math.tanh
local function fastTanh(z)
  if z > 20 then return 1 elseif z < -20 then return -1 end
  local e = mexp(2 * z)
  return (e - 1) / (e + 1)
end

local function sigmoid(z)
  if z > 20 then return 1 elseif z < -20 then return 0 end
  return 1 / (1 + mexp(-z))
end

local function clampNum(v, lo, hi)
  if v ~= v then return 0 end                    -- NaN
  if v > hi then return hi elseif v < lo then return lo end
  return v
end

local function parseBlob(s, n)
  local t = {}
  local i = 0
  for num in s:gmatch("%S+") do
    i = i + 1
    t[i] = tonumber(num)
  end
  if n and i ~= n then return nil end
  return t
end

local function loadWeights(name)
  package.loaded["controller/lstmabs_weights_" .. name] = nil   -- allow runtime reload
  local ok, mod = pcall(require, "controller/lstmabs_weights_" .. name)
  if not ok or type(mod) ~= "table" then
    print("=== LSTM_ABS: weights missing for lens '" .. tostring(name) .. "' ===")
    return false
  end
  W = mod
  mean = parseBlob(W.mean_s, W.input_dim)
  std = parseBlob(W.std_s, W.input_dim)
  if not mean or not std then print("=== LSTM_ABS: bad norm blobs ===") return false end

  layers = {}
  for li, L in ipairs(W.layers) do
    local H, D = L.hidden, L.in_dim
    local wih = parseBlob(L.w_ih_s, 4 * H * D)
    local whh = parseBlob(L.w_hh_s, 4 * H * H)
    local bih = parseBlob(L.b_ih_s, 4 * H)
    local bhh = parseBlob(L.b_hh_s, 4 * H)
    if not (wih and whh and bih and bhh) then
      print("=== LSTM_ABS: bad layer " .. li .. " blob ===")
      return false
    end
    layers[li] = {H = H, D = D, wih = wih, whh = whh, bih = bih, bhh = bhh}
  end

  headW = parseBlob(W.head.w_s, W.head.rows * W.head.cols)
  headB = parseBlob(W.head.b_s, W.head.rows)
  if not headW or not headB then print("=== LSTM_ABS: bad head blob ===") return false end

  -- preallocate every buffer; zero allocation per tick
  hBuf, cBuf, gBuf = {}, {}, {}
  for li, L in ipairs(layers) do
    hBuf[li], cBuf[li], gBuf[li] = {}, {}, {}
    for i = 1, L.H do hBuf[li][i] = 0; cBuf[li][i] = 0 end
    for i = 1, 4 * L.H do gBuf[li][i] = 0 end
  end
  xBuf = {}
  for i = 1, W.input_dim do xBuf[i] = 0 end
  obs = {}
  for i = 1, W.input_dim do obs[i] = 0 end

  print(("=== LSTM_ABS: lens '%s' loaded (%d in, %d hidden, %d layers) ==="):format(
    W.lens, W.input_dim, W.hidden, #layers))
  return true
end

local function resetState()
  if not layers then return end
  for li, L in ipairs(layers) do
    local h, c = hBuf[li], cBuf[li]
    for i = 1, L.H do h[i] = 0; c[i] = 0 end
  end
end

-- one LSTM timestep, mirrors export_lstm_abs.py's lua_forward exactly
local function stepNet()
  local x, xn = xBuf, W.input_dim
  for li = 1, #layers do
    local L = layers[li]
    local H, D = L.H, xn
    local g, wih, whh, bih, bhh = gBuf[li], L.wih, L.whh, L.bih, L.bhh
    local h, c = hBuf[li], cBuf[li]

    for k = 1, 4 * H do
      local acc = bih[k] + bhh[k]
      local base = (k - 1) * D
      for j = 1, D do acc = acc + wih[base + j] * x[j] end
      local hbase = (k - 1) * H
      for j = 1, H do acc = acc + whh[hbase + j] * h[j] end
      g[k] = acc
    end

    for i = 1, H do
      local ig = sigmoid(g[i])
      local fg = sigmoid(g[H + i])
      local gg = fastTanh(g[2 * H + i])
      local og = sigmoid(g[3 * H + i])
      local cn = fg * c[i] + ig * gg
      c[i] = cn
      h[i] = og * fastTanh(cn)
    end
    x, xn = h, H
  end

  local rows, cols = W.head.rows, W.head.cols
  for k = 1, rows do
    local acc = headB[k]
    local base = (k - 1) * cols
    for j = 1, cols do acc = acc + headW[base + j] * x[j] end
    cmd[k] = sigmoid(acc)
  end
end


-- deterministic Lua-vs-numpy forward check; feeds a fixed pre-normalized
-- sequence so normalization is excluded and only the net math is tested
local function selfTest()
  resetState()
  for t = 1, 30 do
    for i = 1, W.input_dim do
      xBuf[i] = 0.1 * i - 0.5 + 0.01 * t
    end
    stepNet()
  end
  print(("=== LSTM_ABS SELFTEST lens=%s cmd=%.6f,%.6f,%.6f,%.6f ==="):format(
    tostring(W.lens), cmd[1], cmd[2], cmd[3], cmd[4]))
  resetState()
end

-- obs order must match W.input_cols exactly
local function buildObs()
  local wr = wheels.wheelRotators
  local ev = electrics.values
  local sx, sy, sz = 0, 0, 0
  if W.sensor_src == "gy2" then
    sx, sy, sz = sensors.gx2 or 0, sensors.gy2 or 0, sensors.gz2 or 0   -- matches the telemetry logger
  elseif sensors and sensors.ffiSensors then
    sx = sensors.ffiSensors.sensorX or 0
    sy = sensors.ffiSensors.sensorY or 0
    sz = sensors.ffiSensors.sensorZ or 0
  end
  local yawRate = 0
  pcall(function() yawRate = obj:getYawAngularVelocity() or 0 end)
  local roll, pitch = 0, 0
  pcall(function() local r, p = obj:getRollPitchYaw(); roll = r or 0; pitch = p or 0 end)

  local o = obs
  o[1] = mabs(wr[2].wheelSpeed or 0)   -- ws_fr
  o[2] = mabs(wr[3].wheelSpeed or 0)   -- ws_fl
  o[3] = mabs(wr[0].wheelSpeed or 0)   -- ws_rr
  o[4] = mabs(wr[1].wheelSpeed or 0)   -- ws_rl
  o[5] = sx
  o[6] = sy
  o[7] = sz
  o[8] = yawRate
  o[9] = ev.steering_input or 0
  o[10] = ev.steering or 0
  o[11] = frontWheelAngle()
  o[12] = lastWrittenBrake >= 0 and lastWrittenBrake or (input.brake or 0)
  o[13] = input.throttle or 0
  o[14] = roll
  o[15] = pitch
  o[16] = ev.rpm or 0
  o[17] = ev.gear_A or ev.gearIndex or 0

  if W.zero_idx then
    for _, zi in ipairs(W.zero_idx) do o[zi] = 0 end
  end
  for i = 1, W.input_dim do
    xBuf[i] = clampNum((o[i] - mean[i]) / std[i], -W.clip_obs, W.clip_obs)
  end
end

-- pedal-split gain at full pedal; logged torques already include it
local function pedalSplitGain(w)
  local split = w.brakeInputSplit or 1
  local coef = w.brakeSplitCoef or 1
  return mmin(1, split) + mmax(1 - split, 0) * coef
end

local function captureCapacity()
  local wr = wheels.wheelRotators
  wheelCount = wheels.wheelRotatorCount or 0
  for i = 0, mmin(wheelCount, 4) - 1 do
    origCapacity[i + 1] = wr[i].brakeTorque or 0
    invSplit[i + 1] = 1 / mmax(pedalSplitGain(wr[i]), 0.05)
  end
  if not fwNode1 and wheelCount >= 4 then
    fwNode1, fwNode2 = wr[2].node1, wr[2].node2
  end
end

local function restoreBrakes()
  local wr = wheels.wheelRotators
  for i = 0, mmin(wheelCount, 4) - 1 do
    wr[i].brakeTorque = origCapacity[i + 1]
  end
  lastWrittenBrake = -1
end

-- cmd is fraction of car max; scale capacity and pin the pedal so the stock
-- pipeline's capacity*pedal lands on exactly that fraction
local function applyBrakes()
  local wr = wheels.wheelRotators
  local f = clampNum(cmd[1], CMD_FLOOR, 1)
  local fl = clampNum(cmd[2], CMD_FLOOR, 1)
  local rr = clampNum(cmd[3], CMD_FLOOR, 1)
  local rl = clampNum(cmd[4], CMD_FLOOR, 1)
  if warmup > 0 then f, fl, rr, rl = WARMUP_FRAC, WARMUP_FRAC, WARMUP_FRAC, WARMUP_FRAC end

  wr[2].brakeTorque = origCapacity[3] * mmin(1, f * invSplit[3])
  wr[3].brakeTorque = origCapacity[4] * mmin(1, fl * invSplit[4])
  wr[0].brakeTorque = origCapacity[1] * mmin(1, rr * invSplit[1])
  wr[1].brakeTorque = origCapacity[2] * mmin(1, rl * invSplit[2])
  input.brake = 1.0
  lastWrittenBrake = 1.0

  electrics.values.lstmabs_active = 1
  electrics.values.lstmabs_fr = f
  electrics.values.lstmabs_fl = fl
  electrics.values.lstmabs_rr = rr
  electrics.values.lstmabs_rl = rl
end

local dbgTicks = 0
local function traceLine()
  local wr = wheels.wheelRotators
  local ev = electrics.values
  local v = 0
  pcall(function() v = obj:getVelocity():length() end)
  print(string.format(
    "=== LSTM_TRACE t=%d v=%.2f ws=%.2f,%.2f,%.2f,%.2f cmd=%.3f,%.3f,%.3f,%.3f cap=%.0f,%.0f,%.0f,%.0f des=%.0f,%.0f,%.0f,%.0f app=%.0f,%.0f,%.0f,%.0f abs=%s,%s,%s,%s ebrake=%.2f ibrake=%.2f hasABS=%s sy=%.2f split=%.3f,%.3f ===",
    dbgTicks, v,
    obs[1], obs[2], obs[3], obs[4],
    cmd[1], cmd[2], cmd[3], cmd[4],
    wr[2].brakeTorque, wr[3].brakeTorque, wr[0].brakeTorque, wr[1].brakeTorque,
    wr[2].desiredBrakingTorque or 0, wr[3].desiredBrakingTorque or 0, wr[0].desiredBrakingTorque or 0, wr[1].desiredBrakingTorque or 0,
    wr[2].brakingTorque or 0, wr[3].brakingTorque or 0, wr[0].brakingTorque or 0, wr[1].brakingTorque or 0,
    tostring(wr[2].absActive), tostring(wr[3].absActive), tostring(wr[0].absActive), tostring(wr[1].absActive),
    ev.brake or -1, input.brake or -1, tostring(ev.hasABS), obs[6], 1 / invSplit[3], 1 / invSplit[1]))
  local ro = {}
  for i = 1, W.input_dim do ro[i] = string.format("%.5f", obs[i]) end
  print("=== LSTM_OBS t=" .. dbgTicks .. " " .. table.concat(ro, ",") .. " ===")
end

local function runTick()
  buildObs()
  stepNet()
  if FORCE_CMD then cmd[1], cmd[2], cmd[3], cmd[4] = FORCE_CMD, FORCE_CMD, FORCE_CMD, FORCE_CMD end
  if TRACE_EVERY > 0 and dbgTicks % TRACE_EVERY == 0 then traceLine() end
  if warmup > 0 then warmup = warmup - 1 end
  dbgTicks = dbgTicks + 1
  if dbgTicks == 15 then
    local ro, nz = {}, {}
    for i = 1, W.input_dim do
      ro[i] = string.format("%.4f", obs[i])
      nz[i] = string.format("%.2f", xBuf[i])
    end
    print("=== LSTM_ABS DBG raw=" .. table.concat(ro, ",") .. " ===")
    print("=== LSTM_ABS DBG norm=" .. table.concat(nz, ",") .. " ===")
    print(string.format("=== LSTM_ABS DBG cmd=%.4f,%.4f,%.4f,%.4f ===",
      cmd[1], cmd[2], cmd[3], cmd[4]))
  end
end

local function updateEngage(driverBrake, speed)
  if not engaged then
    if driverBrake > ENGAGE_BRAKE and speed > ENGAGE_SPEED then
      engaged = true
      warmup = WARMUP_TICKS
      resetState()
      captureCapacity()
      dbgTicks = 0
      engageCount = engageCount + 1
      electrics.values.lstmabs_engagements = engageCount
      electrics.values.lstmabs_lens = lensName
    end
  else
    if speed < RELEASE_SPEED or driverBrake < 0.05 then
      engaged = false
      restoreBrakes()
      electrics.values.lstmabs_active = 0
    end
  end
end

local function update(dtPhys)
  if not weightsOk then return end

  -- lastWrittenBrake recovers true driver intent, since we overwrite input.brake
  local driverBrake = input.brake or 0
  if lastWrittenBrake >= 0 and mabs(driverBrake - lastWrittenBrake) < 1e-6 then
    driverBrake = 1.0
  end
  local speed = 0
  pcall(function() local v = obj:getVelocity(); speed = v and v:length() or 0 end)

  updateEngage(driverBrake, speed)
  if not engaged then return end

  timeAccum = timeAccum + dtPhys
  while timeAccum >= TICK_STEP do
    runTick()
    timeAccum = timeAccum - TICK_STEP
  end
  applyBrakes()  -- re-assert every physics step or stock reclaims full torque
end

-- lstmabs_lens.txt in the userpath overrides the jbeam default
local function lensFromFile()
  local name = nil
  pcall(function()
    local txt = readFile("lstmabs_lens.txt")
    if type(txt) == "string" then name = txt:match("^%s*(%S+)") end
  end)
  return name
end

local function activate(name)
  if engaged then restoreBrakes() end
  engaged = false
  timeAccum = 0
  lastWrittenBrake = -1
  weightsOk = loadWeights(name)
  if weightsOk then
    lensName = name
    captureCapacity()
    selfTest()
  end
  electrics.values.lstmabs_active = 0
  electrics.values.lstmabs_engagements = engageCount
  electrics.values.lstmabs_lens = weightsOk and lensName or ("MISSING:" .. tostring(name))
  return weightsOk
end

-- runtime switch: controller.getController("lstm_abs").setLens("band_g_mix")
local function setLens(name)
  return activate(name)
end

local function init(jbeamData)
  activate(lensFromFile() or (jbeamData and jbeamData.lens) or "baseline_dyn")
end

local function reset()
  engaged = false
  timeAccum = 0
  lastWrittenBrake = -1
  resetState()
  if weightsOk then captureCapacity() end
  electrics.values.lstmabs_active = 0
end

M.init = init
M.setLens = setLens
M.reset = reset
M.update = update

return M
