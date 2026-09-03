-- lstm2khz.lua — Raw 2kHz sensor collector for LSTM training data
-- Captures every sensor at physics rate. No aggregation, no averaging.
-- Python sends startRecording(path) / stopRecording() via queue_lua_command.

local M = {}

local recording = false
local file = nil
local buffer = {}
local bufferCount = 0
local FLUSH_SIZE = 2000  -- ~1 second at 2kHz
local simTime = 0
local prevSpeed = -1
local wasBraking = false
local frameCount = 0

-- Vehicle geometry (read at init)
local wheelbase = 2.6       -- meters, overwritten from vehicle data
local trackWidthFront = 1.5 -- meters, overwritten from vehicle data
local trackWidthRear = 1.5  -- meters, overwritten from vehicle data

-- Front wheel node IDs for direct angle measurement
local frontWheelNode1 = nil
local frontWheelNode2 = nil
local frontWheelInited = false

-- Vehicle metadata (read once at init/reset)
local drivetrain = "unknown"  -- AWD, FWD, RWD, 4WD
local vehicleModel = "unknown"
local gripPattern = "unknown"

-- Differential state (read at init, constant per config)
-- Up to 3 diffs: front, rear, center. Missing diffs get "none"/0.
local diffDevices = {}  -- populated at init
local diffFrontType, diffRearType, diffCenterType = "none", "none", "none"
local diffFrontRatio, diffRearRatio, diffCenterRatio = 0, 0, 0
local diffFrontLockCoef, diffRearLockCoef, diffCenterLockCoef = 0, 0, 0
local diffFrontRevLockCoef, diffRearRevLockCoef, diffCenterRevLockCoef = 0, 0, 0
local diffFrontPreload, diffRearPreload, diffCenterPreload = 0, 0, 0

-- Per-wheel friction labels (set before recording via startRecording config)
local frictionStaticFR, frictionStaticFL, frictionStaticRR, frictionStaticRL = -1, -1, -1, -1
local frictionSlidingFR, frictionSlidingFL, frictionSlidingRR, frictionSlidingRL = -1, -1, -1, -1

-- Debug: safety variant label (set via startRecording config)
local safetyVariant = "unknown"

-- Column header
-- REAL SENSORS (features): ws_*, sensor_*, yaw_rate, steer_*, front_wheel_angle,
--   rpm, gear, brake_input, throttle_input, brake_torque_*
-- CHEAT SENSORS (labels/grading only): speed_velo, airspeed,
--   slip_*, contact_*, downforce_*, damage, drivetrain
local COLUMNS = table.concat({
  "timestamp", "dt",
  -- Real sensors
  "ws_fr", "ws_fl", "ws_rr", "ws_rl",
  "sensor_x", "sensor_y", "sensor_z",
  "yaw_rate",
  "steer_input", "steer_angle", "front_wheel_angle",
  "rpm", "gear",
  "brake_input", "throttle_input",
  -- Real: per-wheel brake torque (Nm) — ECU knows this
  "brake_torque_fr", "brake_torque_fl", "brake_torque_rr", "brake_torque_rl",
  -- Real: per-wheel drive/propulsion torque (Nm) — ECU knows this in modern cars
  "drive_torque_fr", "drive_torque_fl", "drive_torque_rr", "drive_torque_rl",
  -- Cheat: per-wheel slip ratio (needs true speed)
  "slip_fr", "slip_fl", "slip_rr", "slip_rl",
  -- Cheat: per-wheel ground contact (1=ground, 0=air)
  "contact_fr", "contact_fl", "contact_rr", "contact_rl",
  -- Cheat: per-wheel downforce / normal load (N)
  "downforce_fr", "downforce_fl", "downforce_rr", "downforce_rl",
  -- Real: per-wheel angular velocity (rad/s) — from encoder, different from wheelSpeed
  "wheel_angvel_fr", "wheel_angvel_fl", "wheel_angvel_rr", "wheel_angvel_rl",
  -- Cheat: per-wheel translational speed (true hub velocity through space, m/s)
  "wheel_velo_fr", "wheel_velo_fl", "wheel_velo_rr", "wheel_velo_rl",
  -- Cheat: per-wheel brake temperature (Kelvin)
  "brake_temp_fr", "brake_temp_fl", "brake_temp_rr", "brake_temp_rl",
  -- Cheat: vehicle orientation (from physics engine)
  "roll_angle", "pitch_angle", "yaw_angle",
  -- Cheat: velocity components (for lateral/longitudinal speed decomposition)
  "velo_x", "velo_y", "velo_z",
  -- Cheat: labels
  "speed_velo", "airspeed", "virtual_airspeed",
  -- Cheat: stock ESC/ABS virtual speed (electrics.values.wheelspeed, m/s)
  "esc_wheelspeed",
  -- Cheat: per-wheel friction labels (constant per session)
  "friction_static_fr", "friction_static_fl", "friction_static_rr", "friction_static_rl",
  "friction_sliding_fr", "friction_sliding_fl", "friction_sliding_rr", "friction_sliding_rl",
  -- Cheat: differential dynamic state (torque in/out per diff, 2kHz)
  "diff_front_input_torque", "diff_front_output_torque_l", "diff_front_output_torque_r",
  "diff_rear_input_torque", "diff_rear_output_torque_l", "diff_rear_output_torque_r",
  "diff_center_input_torque", "diff_center_output_torque_f", "diff_center_output_torque_r",
  -- Meta
  "damage", "drivetrain", "vehicle_model", "grip_pattern",
  -- Meta: differential constants (from jbeam, read at init)
  "diff_front_type", "diff_front_ratio", "diff_front_lock_coef",
  "diff_rear_type", "diff_rear_ratio", "diff_rear_lock_coef",
  "diff_center_type", "diff_center_ratio", "diff_center_lock_coef",
  -- Real config: vehicle geometry constants (read at init)
  "wheelbase", "track_width_front", "track_width_rear",
  -- Debug: safety system activity (per-frame, LSTM never sees these)
  "abs_active", "esc_active", "tcs_active",
  "has_abs", "has_esc", "has_tcs",
  -- Debug: safety variant label (constant per session)
  "safety_variant",
}, ",") .. "\n"

local math_abs = math.abs
local math_sqrt = math.sqrt
local math_acos = math.acos
local math_atan2 = math.atan2
local math_pi = math.pi
local fmt = string.format


local function flushBuffer()
  if not file or bufferCount == 0 then return end
  file:write(table.concat(buffer, "", 1, bufferCount))
  file:flush()
  bufferCount = 0
end


local function startRecording(path, config)
  if recording then
    flushBuffer()
    if file then file:close() end
  end

  config = config or {}

  -- Per-wheel friction labels (must be set before recording)
  frictionStaticFR  = config.friction_static_fr  or -1
  frictionStaticFL  = config.friction_static_fl  or -1
  frictionStaticRR  = config.friction_static_rr  or -1
  frictionStaticRL  = config.friction_static_rl  or -1
  frictionSlidingFR = config.friction_sliding_fr or -1
  frictionSlidingFL = config.friction_sliding_fl or -1
  frictionSlidingRR = config.friction_sliding_rr or -1
  frictionSlidingRL = config.friction_sliding_rl or -1

  -- Session metadata
  vehicleModel = config.vehicle_model or "unknown"
  gripPattern  = config.grip_pattern  or "unknown"
  safetyVariant = config.safety_variant or "unknown"

  file = io.open(path, "w")
  if not file then
    print("=== LSTM2KHZ: FAILED to open " .. path .. " ===")
    return
  end

  file:write(COLUMNS)
  buffer = {}
  bufferCount = 0
  simTime = 0
  prevSpeed = -1
  frameCount = 0
  recording = true
  print(fmt("=== LSTM2KHZ: Recording to %s | %s | %s | grip=[%.2f,%.2f,%.2f,%.2f] ===",
    path, vehicleModel, gripPattern,
    frictionStaticFR, frictionStaticFL, frictionStaticRR, frictionStaticRL))
end


local function updateFriction(config)
  config = config or {}
  frictionStaticFR  = config.friction_static_fr  or frictionStaticFR
  frictionStaticFL  = config.friction_static_fl  or frictionStaticFL
  frictionStaticRR  = config.friction_static_rr  or frictionStaticRR
  frictionStaticRL  = config.friction_static_rl  or frictionStaticRL
  frictionSlidingFR = config.friction_sliding_fr or frictionSlidingFR
  frictionSlidingFL = config.friction_sliding_fl or frictionSlidingFL
  frictionSlidingRR = config.friction_sliding_rr or frictionSlidingRR
  frictionSlidingRL = config.friction_sliding_rl or frictionSlidingRL
  if config.grip_pattern then gripPattern = config.grip_pattern end
end


local function stopRecording()
  if not recording then return end
  flushBuffer()
  if file then
    file:close()
    file = nil
  end
  recording = false
  print("=== LSTM2KHZ: Stopped. " .. frameCount .. " frames written ===")
end


local function getFrontWheelAngle()
  if not frontWheelInited then return 0 end
  local cosAngle = obj:nodeVecPlanarCosRightForward(frontWheelNode1, frontWheelNode2)
  if not cosAngle then return 0 end
  local angle = math_acos(math.min(1, math.max(-1, cosAngle)))
  local steerIn = electrics.values.steering_input or 0
  if steerIn < 0 then angle = -angle end
  return angle
end


local function onPhysicsStep(dtPhys)
  if not recording then return end
  if dtPhys <= 0 then return end

  simTime = simTime + dtPhys

  -- Wheel speeds (raw, absolute)
  local wsFR, wsFL, wsRR, wsRL = 0, 0, 0, 0
  if wheels and wheels.wheelRotators and wheels.wheelRotatorCount and wheels.wheelRotatorCount >= 4 then
    wsRR = math_abs(wheels.wheelRotators[0].wheelSpeed or 0)
    wsRL = math_abs(wheels.wheelRotators[1].wheelSpeed or 0)
    wsFR = math_abs(wheels.wheelRotators[2].wheelSpeed or 0)
    wsFL = math_abs(wheels.wheelRotators[3].wheelSpeed or 0)
  end

  -- Raw IMU
  local sx = (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorX) or 0
  local sy = (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorY) or 0
  local sz = (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorZ) or 0

  -- Yaw rate
  local yawRate = 0
  pcall(function() yawRate = obj:getYawAngularVelocity() or 0 end)

  -- Velocity magnitude (label)
  local vel = obj:getVelocity()
  local speedVelo = 0
  if vel then
    speedVelo = math_sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
  end

  -- Steering: driver input (normalized -1..1) and physical column angle (rad)
  local steerInput = electrics.values.steering_input or 0
  local steerAngle = electrics.values.steering or 0  -- actual column angle in rad

  -- Front wheel angle from node geometry (2kHz via obj:nodeVecPlanarCosRightForward)
  local frontWheelAngle = getFrontWheelAngle()

  -- 60Hz sensors (stale between GFX updates, that's fine)
  local rpm = electrics.values.rpm or 0
  local gear = electrics.values.gear_A or electrics.values.gearIndex or 0

  -- Pedals (2kHz)
  local brakeIn = input.brake or 0
  local throttleIn = input.throttle or 0

  -- ═══════════════════════════════════════════════════════════════════
  -- CHEAT SENSORS (labels / grading only — never used as LSTM features)
  -- ═══════════════════════════════════════════════════════════════════

  -- Per-wheel slip ratio: (carSpeed - wheelSpeed) / carSpeed
  local slipFR, slipFL, slipRR, slipRL = 0, 0, 0, 0
  if speedVelo > 0.5 then
    slipFR = math.max(0, math.min(1, (speedVelo - wsFR) / speedVelo))
    slipFL = math.max(0, math.min(1, (speedVelo - wsFL) / speedVelo))
    slipRR = math.max(0, math.min(1, (speedVelo - wsRR) / speedVelo))
    slipRL = math.max(0, math.min(1, (speedVelo - wsRL) / speedVelo))
  end

  -- Per-wheel ground contact (1=on ground, 0=airborne)
  local contFR, contFL, contRR, contRL = 0, 0, 0, 0
  -- Per-wheel brake torque (Nm) and downforce (N)
  local btFR, btFL, btRR, btRL = 0, 0, 0, 0
  local dfFR, dfFL, dfRR, dfRL = 0, 0, 0, 0
  if wheels and wheels.wheelRotators and wheels.wheelRotatorCount and wheels.wheelRotatorCount >= 4 then
    local wr = wheels.wheelRotators
    -- Contact: contactDepth > 0 means wheel is touching ground
    contRR = (wr[0].contactDepth or 0) > 0 and 1 or 0
    contRL = (wr[1].contactDepth or 0) > 0 and 1 or 0
    contFR = (wr[2].contactDepth or 0) > 0 and 1 or 0
    contFL = (wr[3].contactDepth or 0) > 0 and 1 or 0
    -- Brake torque
    btRR = wr[0].desiredBrakingTorque or 0
    btRL = wr[1].desiredBrakingTorque or 0
    btFR = wr[2].desiredBrakingTorque or 0
    btFL = wr[3].desiredBrakingTorque or 0
    -- Downforce / normal load
    dfRR = wr[0].downForce or 0
    dfRL = wr[1].downForce or 0
    dfFR = wr[2].downForce or 0
    dfFL = wr[3].downForce or 0
  end

  -- Per-wheel drive torque (Nm) — propulsion from engine/motor
  local dtqFR, dtqFL, dtqRR, dtqRL = 0, 0, 0, 0
  if wheels and wheels.wheelRotators and wheels.wheelRotatorCount and wheels.wheelRotatorCount >= 4 then
    dtqRR = wheels.wheelRotators[0].propulsionTorque or 0
    dtqRL = wheels.wheelRotators[1].propulsionTorque or 0
    dtqFR = wheels.wheelRotators[2].propulsionTorque or 0
    dtqFL = wheels.wheelRotators[3].propulsionTorque or 0
  end

  -- Per-wheel brake temperature (Kelvin) — cheat label for fade detection
  local btempFR, btempFL, btempRR, btempRL = 0, 0, 0, 0
  if wheels and wheels.wheels then
    for _, wd in pairs(wheels.wheels) do
      local temp = wd.brakeCoreTemperature or 0
      local name = wd.name or ""
      if name == "FR" then btempFR = temp
      elseif name == "FL" then btempFL = temp
      elseif name == "RR" then btempRR = temp
      elseif name == "RL" then btempRL = temp end
    end
  end

  -- Per-wheel angular velocity (rad/s) — real sensor, different from wheelSpeed (m/s)
  local avFR, avFL, avRR, avRL = 0, 0, 0, 0
  if wheels and wheels.wheelRotators and wheels.wheelRotatorCount and wheels.wheelRotatorCount >= 4 then
    avRR = math_abs(wheels.wheelRotators[0].angularVelocity or 0)
    avRL = math_abs(wheels.wheelRotators[1].angularVelocity or 0)
    avFR = math_abs(wheels.wheelRotators[2].angularVelocity or 0)
    avFL = math_abs(wheels.wheelRotators[3].angularVelocity or 0)
  end

  -- Per-wheel translational speed (true hub velocity through space)
  local wvFR, wvFL, wvRR, wvRL = 0, 0, 0, 0
  pcall(function()
    if wheels and wheels.wheelRotators and wheels.wheelRotatorCount and wheels.wheelRotatorCount >= 4 then
      local wr = wheels.wheelRotators
      for idx = 0, 3 do
        local n1 = wr[idx].node1
        local n2 = wr[idx].node2
        if n1 and n2 then
          local nv = obj:getNodeVelocity(n1, n2)
          if nv then
            local spd = math_sqrt(nv.x * nv.x + nv.y * nv.y + nv.z * nv.z)
            if idx == 0 then wvRR = spd
            elseif idx == 1 then wvRL = spd
            elseif idx == 2 then wvFR = spd
            elseif idx == 3 then wvFL = spd
            end
          end
        end
      end
    end
  end)

  -- Vehicle orientation (2kHz from physics)
  local rollAngle, pitchAngle, yawAngle = 0, 0, 0
  pcall(function()
    local r, p, y = obj:getRollPitchYaw()
    rollAngle = r or 0
    pitchAngle = p or 0
    yawAngle = y or 0
  end)

  -- Velocity components (for lateral vs longitudinal speed decomposition)
  local veloX, veloY, veloZ = 0, 0, 0
  if vel then
    veloX = vel.x or 0
    veloY = vel.y or 0
    veloZ = vel.z or 0
  end

  -- Differential dynamic torques (2kHz)
  local dfIT, dfOTL, dfOTR = 0, 0, 0  -- front diff
  local drIT, drOTL, drOTR = 0, 0, 0  -- rear diff
  local dcIT, dcOTF, dcOTR = 0, 0, 0  -- center diff
  pcall(function()
    if diffDevices.front then
      local d = diffDevices.front
      dfIT = d.inputTorque or 0
      dfOTL = d.outputTorque1 or 0
      dfOTR = d.outputTorque2 or 0
    end
    if diffDevices.rear then
      local d = diffDevices.rear
      drIT = d.inputTorque or 0
      drOTL = d.outputTorque1 or 0
      drOTR = d.outputTorque2 or 0
    end
    if diffDevices.center then
      local d = diffDevices.center
      dcIT = d.inputTorque or 0
      dcOTF = d.outputTorque1 or 0
      dcOTR = d.outputTorque2 or 0
    end
  end)

  local airspeed = electrics.values.airspeed or 0
  local virtualAirspeed = electrics.values.virtualAirspeed or 0
  local escWheelspeed = electrics.values.wheelspeed or 0
  local damage = electrics.values.damage or 0

  -- Debug: safety system activity flags
  -- ABS: check per-wheel absActive directly (electrics flag is stale in onPhysicsStep)
  local absAct = 0
  if wheels and wheels.wheelRotators and wheels.wheelRotatorCount and wheels.wheelRotatorCount >= 4 then
    for i = 0, wheels.wheelRotatorCount - 1 do
      if wheels.wheelRotators[i].absActive then absAct = 1; break end
    end
  end
  -- ESC/TCS: read from electrics (updated in updateGFX, ~60Hz resolution)
  local escAct = 0
  local tcsAct = 0
  pcall(function()
    local esc = electrics.values.esc
    if esc and esc ~= 0 then escAct = 1 end
    local tcs = electrics.values.tcs
    if tcs and tcs ~= 0 then tcsAct = 1 end
  end)
  -- electrics.values.hasABS/ESC/TCS can be boolean or number depending on the
  -- vehicle; %d formatting needs a real number, not a boolean.
  local function toFlag(v)
    if v == true then return 1 end
    if v == false or v == nil then return 0 end
    return v
  end
  local hasAbs = toFlag(electrics.values.hasABS)
  local hasEsc = toFlag(electrics.values.hasESC)
  local hasTcs = toFlag(electrics.values.hasTCS)

  -- Buffer the row
  frameCount = frameCount + 1
  bufferCount = bufferCount + 1
  buffer[bufferCount] = fmt(
    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
    -- 1-2: timestamp, dt
    fmt("%.6f", simTime), fmt("%.6f", dtPhys),
    -- 3-6: wheel speeds
    fmt("%.4f", wsFR), fmt("%.4f", wsFL), fmt("%.4f", wsRR), fmt("%.4f", wsRL),
    -- 7-9: IMU
    fmt("%.4f", sx), fmt("%.4f", sy), fmt("%.4f", sz),
    -- 10: yaw
    fmt("%.6f", yawRate),
    -- 11-13: steering
    fmt("%.6f", steerInput), fmt("%.6f", steerAngle), fmt("%.6f", frontWheelAngle),
    -- 14-15: powertrain
    fmt("%.1f", rpm), fmt("%d", gear),
    -- 16-17: pedals
    fmt("%.4f", brakeIn), fmt("%.4f", throttleIn),
    -- 18-21: brake torque
    fmt("%.1f", btFR), fmt("%.1f", btFL), fmt("%.1f", btRR), fmt("%.1f", btRL),
    -- 22-25: drive torque
    fmt("%.1f", dtqFR), fmt("%.1f", dtqFL), fmt("%.1f", dtqRR), fmt("%.1f", dtqRL),
    -- 26-29: slip
    fmt("%.4f", slipFR), fmt("%.4f", slipFL), fmt("%.4f", slipRR), fmt("%.4f", slipRL),
    -- 26-29: contact
    fmt("%d", contFR), fmt("%d", contFL), fmt("%d", contRR), fmt("%d", contRL),
    -- 30-33: downforce
    fmt("%.1f", dfFR), fmt("%.1f", dfFL), fmt("%.1f", dfRR), fmt("%.1f", dfRL),
    -- 34-37: angular velocity
    fmt("%.4f", avFR), fmt("%.4f", avFL), fmt("%.4f", avRR), fmt("%.4f", avRL),
    -- 38-41: wheel translational velocity
    fmt("%.4f", wvFR), fmt("%.4f", wvFL), fmt("%.4f", wvRR), fmt("%.4f", wvRL),
    -- 42-45: brake temperature
    fmt("%.1f", btempFR), fmt("%.1f", btempFL), fmt("%.1f", btempRR), fmt("%.1f", btempRL),
    -- 46-48: orientation
    fmt("%.6f", rollAngle), fmt("%.6f", pitchAngle), fmt("%.6f", yawAngle),
    -- 49-51: velocity components
    fmt("%.4f", veloX), fmt("%.4f", veloY), fmt("%.4f", veloZ),
    -- 52-53: speed labels
    fmt("%.4f", speedVelo), fmt("%.4f", airspeed), fmt("%.4f", virtualAirspeed),
    fmt("%.4f", escWheelspeed),
    -- 54-61: friction labels
    fmt("%.4f", frictionStaticFR), fmt("%.4f", frictionStaticFL),
    fmt("%.4f", frictionStaticRR), fmt("%.4f", frictionStaticRL),
    fmt("%.4f", frictionSlidingFR), fmt("%.4f", frictionSlidingFL),
    fmt("%.4f", frictionSlidingRR), fmt("%.4f", frictionSlidingRL),
    -- diff dynamic torques (9 values)
    fmt("%.1f", dfIT), fmt("%.1f", dfOTL), fmt("%.1f", dfOTR),
    fmt("%.1f", drIT), fmt("%.1f", drOTL), fmt("%.1f", drOTR),
    fmt("%.1f", dcIT), fmt("%.1f", dcOTF), fmt("%.1f", dcOTR),
    -- meta
    fmt("%.2f", damage), drivetrain, vehicleModel, gripPattern,
    -- diff constants
    diffFrontType, fmt("%.4f", diffFrontRatio), fmt("%.4f", diffFrontLockCoef),
    diffRearType, fmt("%.4f", diffRearRatio), fmt("%.4f", diffRearLockCoef),
    diffCenterType, fmt("%.4f", diffCenterRatio), fmt("%.4f", diffCenterLockCoef),
    -- vehicle geometry constants
    fmt("%.4f", wheelbase), fmt("%.4f", trackWidthFront), fmt("%.4f", trackWidthRear),
    -- debug: safety system activity
    fmt("%d", absAct), fmt("%d", escAct), fmt("%d", tcsAct),
    fmt("%d", hasAbs), fmt("%d", hasEsc), fmt("%d", hasTcs),
    safetyVariant
  )

  -- Flush on buffer full
  if bufferCount >= FLUSH_SIZE then
    flushBuffer()
  end

  -- Flush on brake release
  local isBraking = brakeIn > 0.05
  if wasBraking and not isBraking then
    flushBuffer()
  end
  wasBraking = isBraking

  prevSpeed = speedVelo
end


local function initVehicleGeometry()
  -- Read wheelbase, track width, tire radii from wheel positions
  pcall(function()
    if wheels and wheels.wheelRotators and wheels.wheelRotatorCount and wheels.wheelRotatorCount >= 4 then
      local wr = wheels.wheelRotators
      local fr = wr[2]  -- front-right
      local fl = wr[3]  -- front-left
      local rr = wr[0]  -- rear-right
      local rl = wr[1]  -- rear-left

      -- Node positions for geometry
      if fr and rr and fr.node1 and rr.node1 then
        local fPos = obj:getNodePosition(fr.node1)
        local rPos = obj:getNodePosition(rr.node1)
        if fPos and rPos then
          local dx = fPos.x - rPos.x
          local dy = fPos.y - rPos.y
          local wb = math_sqrt(dx * dx + dy * dy)
          if wb > 1.0 and wb < 5.0 then
            wheelbase = wb
          end
        end
        frontWheelNode1 = fr.node1
        frontWheelNode2 = fr.node2
        frontWheelInited = true
      end

      -- Track width: lateral distance between left and right wheels
      if fr and fl and fr.node1 and fl.node1 then
        local frP = obj:getNodePosition(fr.node1)
        local flP = obj:getNodePosition(fl.node1)
        if frP and flP then
          local tw = math_abs(frP.x - flP.x)
          if tw > 0.5 and tw < 3.0 then trackWidthFront = tw end
        end
      end
      if rr and rl and rr.node1 and rl.node1 then
        local rrP = obj:getNodePosition(rr.node1)
        local rlP = obj:getNodePosition(rl.node1)
        if rrP and rlP then
          local tw = math_abs(rrP.x - rlP.x)
          if tw > 0.5 and tw < 3.0 then trackWidthRear = tw end
        end
      end

      -- Detect drivetrain from which wheels have propulsion
      local frontDriven, rearDriven = false, false
      for i = 0, wheels.wheelRotatorCount - 1 do
        local w = wr[i]
        if w and w.isPropulsed then
          if i >= 2 then frontDriven = true else rearDriven = true end
        end
      end
      if frontDriven and rearDriven then drivetrain = "AWD"
      elseif frontDriven then drivetrain = "FWD"
      elseif rearDriven then drivetrain = "RWD"
      else drivetrain = "unknown" end
    end
  end)

  -- Read differential configuration from powertrain
  diffDevices = {}
  diffFrontType = "none"; diffRearType = "none"; diffCenterType = "none"
  diffFrontRatio = 0; diffRearRatio = 0; diffCenterRatio = 0
  diffFrontLockCoef = 0; diffRearLockCoef = 0; diffCenterLockCoef = 0

  pcall(function()
    local diffs = powertrain.getDevicesByCategory("differential")
    if diffs then
      for _, d in pairs(diffs) do
        local name = (d.name or ""):lower()
        local dtype = d.diffType or d.type or "open"
        local ratio = d.gearRatio or 0
        local lockCoef = d.lsdLockCoef or 0

        -- Classify by name or position
        if name:find("front") or name:find("fwd") then
          diffFrontType = dtype
          diffFrontRatio = ratio
          diffFrontLockCoef = lockCoef
          diffDevices.front = d
        elseif name:find("rear") or name:find("rwd") then
          diffRearType = dtype
          diffRearRatio = ratio
          diffRearLockCoef = lockCoef
          diffDevices.rear = d
        elseif name:find("center") or name:find("transfer") then
          diffCenterType = dtype
          diffCenterRatio = ratio
          diffCenterLockCoef = lockCoef
          diffDevices.center = d
        else
          -- Unknown name — assign to first empty slot
          if diffRearType == "none" then
            diffRearType = dtype
            diffRearRatio = ratio
            diffRearLockCoef = lockCoef
            diffDevices.rear = d
          elseif diffFrontType == "none" then
            diffFrontType = dtype
            diffFrontRatio = ratio
            diffFrontLockCoef = lockCoef
            diffDevices.front = d
          end
        end
      end
    end
  end)

  print(fmt("=== LSTM2KHZ: WB=%.3fm TW_F=%.3fm TW_R=%.3fm DT=%s Diffs: F=%s(%.2f) R=%s(%.2f) C=%s(%.2f) ===",
    wheelbase, trackWidthFront, trackWidthRear, drivetrain,
    diffFrontType, diffFrontRatio,
    diffRearType, diffRearRatio,
    diffCenterType, diffCenterRatio))
end


local function updateGFX(dtSim)
  if not frontWheelInited then
    initVehicleGeometry()
  end
end

local function onExtensionLoaded()
  print("=== LSTM2KHZ: Extension loaded (2kHz raw sensor collector) ===")
  enablePhysicsStepHook()
end

local function onReset()
  if recording then
    flushBuffer()
  end
  prevSpeed = -1
  simTime = 0
  initVehicleGeometry()
end

local function getStatus()
  return {
    recording = recording,
    frameCount = frameCount,
    bufferCount = bufferCount,
    fileOpen = file ~= nil,
  }
end

M.onExtensionLoaded = onExtensionLoaded
M.updateGFX = updateGFX
M.onPhysicsStep = onPhysicsStep
M.onReset = onReset
M.startRecording = startRecording
M.stopRecording = stopRecording
M.updateFriction = updateFriction
M.getStatus = getStatus

return M
