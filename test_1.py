import math
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import mujoco


# ========= 1. Configuration and state machine definitions =========

@dataclass
class ROMConfig:
    # Mechanical parameters (consistent with document table)
    m: float = 4.0          # leg mass [kg]
    r: float = 0.25         # COM distance [m]
    I: float = 0.25         # equivalent inertia at knee [kg m^2]
    g: float = 9.81
    eta: float = 0.85
    G: float = 50.0
    k_t: float = 0.05       # motor torque constant [Nm/A]
    i_max: float = 60.0     # max current [A]

    # Passive properties (aligned with MJCF / table)
    k_passive: float = 10.0   # joint stiffness [Nm/rad]
    b_passive: float = 0.8    # viscous damping [Nms/rad]
    theta0: float = 0.0       # rest angle [rad]

    # Control parameters
    # --- PD control: high stiffness, good tracking ---
    Kp_pd: float = 220.0
    Kd_pd: float = 55.0

    # --- Impedance control: softer, interaction-friendly ---
    # These values are chosen so that tracking is still decent,
    # but the torques are smaller and more compliant than PD.
    K_imp: float = 140.0       # impedance stiffness K [Nm/rad]
    B_imp: float = 28.0       # impedance damping B [Nms/rad]

    # Safety thresholds (as fraction of max motor torque)
    torque_soft_ratio: float = 0.6
    torque_hard_ratio: float = 0.8

    # Angle ranges (approximately 90° initial & 60° ROM)
    theta_rest: float = math.radians(0)        # Rest angle: 90deg
    theta_target: float = math.radians(-90)  # Target elevation: +60deg
    raise_duration: float = 2.0    # Leg raise time [s]
    return_duration: float = 2.0   # Return-to-90° time [s]

    # Simulation
    total_time: float = 20.0       # Total simulation time [s]
    decimation: int = 10           # For logging / plotting

    def __post_init__(self):
        tau_max = self.eta * self.G * self.k_t * self.i_max
        self.torque_soft = self.torque_soft_ratio * tau_max
        self.torque_hard = self.torque_hard_ratio * tau_max


class Mode(Enum):
    IDLE_AT_90 = auto()
    RAISING = auto()
    RETURNING = auto()
    STOPPED_AT_LIMIT = auto()


class PatientProfile(Enum):
    ASSIST = "assist"
    NEUTRAL = "neutral"
    RESIST = "resist"


# ========= 2. Healthy knee reference trajectory (simple linear interpolation) =========

def compute_reference(config: ROMConfig, mode: Mode,
                      t_mode: float, theta_current: float):
    """
    Return (theta_ref, theta_ref_dot, next_mode)
    t_mode: elapsed time in current state (seconds)
    """
    theta_rest = config.theta_rest
    theta_target = config.theta_target

    if mode == Mode.IDLE_AT_90:
        # Just stay at 90°, then start raising
        return theta_rest, 0.0, Mode.RAISING

    elif mode == Mode.RAISING:
        frac = min(1.0, t_mode / config.raise_duration)

    # --- 使用余弦轨迹代替线性插值 ---
        theta_ref = theta_rest + (1 - math.cos(math.pi * frac)) / 2 * (theta_target - theta_rest)
        theta_ref_dot = (math.pi / (2 * config.raise_duration)) * math.sin(math.pi * frac) * (theta_target - theta_rest)

        if frac >= 1.0:
            return theta_target, 0.0, Mode.RETURNING
        return theta_ref, theta_ref_dot, mode

    elif mode == Mode.RETURNING:
        frac = min(1.0, t_mode / config.return_duration)

        theta_ref = theta_target + (1 - math.cos(math.pi * frac)) / 2 * (theta_rest - theta_target)
        theta_ref_dot = (math.pi / (2 * config.return_duration)) * math.sin(math.pi * frac) * (theta_rest - theta_target)

        if frac >= 1.0:
            return theta_rest, 0.0, Mode.RAISING
        return theta_ref, theta_ref_dot, mode

    elif mode == Mode.STOPPED_AT_LIMIT:
        # Safety mode: slowly return to 90°
        frac = min(1.0, t_mode / config.return_duration)
        theta_ref = math.cos(theta_current + frac * (theta_rest - theta_current))
        theta_ref_dot = (theta_rest - theta_current) / config.return_duration
        if frac >= 1.0:
            return theta_rest, 0.0, Mode.RAISING
        return theta_ref, theta_ref_dot, mode

    else:
        return theta_rest, 0.0, Mode.IDLE_AT_90


# ========= 3. Patient torque model τ_patient(θ, θ̇, profile) =========

def patient_torque(theta: float, theta_dot: float,
                   config: ROMConfig,
                   profile: PatientProfile) -> float:
    """
    Simple directional model for natural assist / resist:
    - NEUTRAL: 0 Nm
    - ASSIST:  constant torque in the direction of motion (≈ patient helps)
    - RESIST: constant torque opposite to motion (≈ patient resists)
    """
    base = 1.0  # Nm, from "moderate assist/resist ~ 1 Nm" in doc

    if profile == PatientProfile.NEUTRAL:
        return 0.0

    elif profile == PatientProfile.ASSIST:
        # Patient voluntarily assists: torque in the direction of motion
        if theta_dot > 0:
            return +base
        elif theta_dot < 0:
            return -base
        else:
            return 0.0

    elif profile == PatientProfile.RESIST:
        # Patient resists movement: torque opposite to motion
        if theta_dot > 0:
            return -base
        elif theta_dot < 0:
            return +base
        else:
            return 0.0

    else:
        return 0.0


# ========= 4. Control laws: PD & Impedance =========

def feedforward_gravity(theta: float, config: ROMConfig) -> float:
    """
    Gravity compensation:
    τ_g = -m g r sin(theta)
    """
    return -config.m * config.g * config.r * math.sin(theta)


def pd_control(theta, theta_dot, theta_ref, theta_ref_dot, config: ROMConfig):
    """
    Joint-space PD with gravity feedforward.
    This yields very good tracking of theta_ref.
    """
    e = theta_ref - theta
    e_dot = theta_ref_dot - theta_dot
    tau_fb = config.Kp_pd * e + config.Kd_pd * e_dot
    tau_ff = feedforward_gravity(theta, config)
    return tau_ff + tau_fb


def impedance_control(theta, theta_dot, theta_ref, theta_ref_dot, config: ROMConfig):
    """
    Joint-space impedance:
        τ = K_imp (θ_ref - θ) + B_imp (θ̇_ref - θ̇)
    No gravity feedforward is added on purpose, so that
    the device behaves more like a compliant spring-damper
    around the reference rather than a hard position servo.
    """
    e = theta_ref - theta
    e_dot = theta_ref_dot - theta_dot
    return config.K_imp * e + config.B_imp * e_dot


# ========= 5. Core simulation: one session =========

def simulate_session(xml_path: str,
                     controller: str,
                     patient: str,
                     config: ROMConfig,
                     use_viewer: bool = False):
    """
    controller: "pd" or "impedance"
    patient: "assist" / "neutral" / "resist"
    Returns a dict containing time series and metrics.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Optional viewer
    viewer_ctx = None
    if use_viewer:
        try:
            from mujoco import viewer
            viewer_ctx = viewer.launch_passive(model, data)
        except Exception as e:
            print(f"[warning] Cannot start MuJoCo viewer, running headless: {e}")
            viewer_ctx = None

    # Indices
    knee_qpos_id = model.joint("knee").qposadr[0]
    knee_dof_id = model.joint("knee").dofadr[0]
    motor_id = 0  # single actuator

    dt = model.opt.timestep
    n_steps = int(config.total_time / dt)

    gear_eff = config.eta * config.G * config.k_t

    # Logging arrays (full frequency or decimated)
    t_log = np.zeros(n_steps)
    theta_log = np.zeros(n_steps)
    theta_ref_log = np.zeros(n_steps)
    tau_motor_log = np.zeros(n_steps)
    tau_patient_log = np.zeros(n_steps)
    current_log = np.zeros(n_steps)
    mode_log = np.zeros(n_steps, dtype=int)

    # Initial pose: 90°
    data.qpos[knee_qpos_id] = config.theta_rest
    data.qvel[knee_qpos_id] = 0.0

    mode = Mode.IDLE_AT_90
    mode_time = 0.0
    profile = PatientProfile(patient)

    soft_hits = 0
    hard_hits = 0

    def step_once(step_idx: int):
        nonlocal mode, mode_time, soft_hits, hard_hits

        t = step_idx * dt
        t_log[step_idx] = t

        theta = float(data.qpos[knee_qpos_id])
        theta_dot = float(data.qvel[knee_qpos_id])

        # Reference trajectory + state machine
        theta_ref, theta_ref_dot, next_mode = compute_reference(
            config, mode, mode_time, theta
        )

        if next_mode != mode:
            mode = next_mode
            mode_time = 0.0
        else:
            mode_time += dt

        # Choose controller
        if controller == "pd":
            tau_cmd = pd_control(theta, theta_dot, theta_ref, theta_ref_dot, config)
        elif controller == "impedance":
            tau_cmd = impedance_control(theta, theta_dot, theta_ref, theta_ref_dot, config)
        else:
            raise ValueError(f"Unknown controller: {controller}")

        # Safety logic based on previous motor torque
        tau_motor_prev = float(data.actuator_force[motor_id])

        if abs(tau_motor_prev) > config.torque_hard:
            hard_hits += 1
            mode = Mode.STOPPED_AT_LIMIT
            mode_time = 0.0
            tau_cmd = 0.0   # stop pushing further

        elif abs(tau_motor_prev) > config.torque_soft:
            soft_hits += 1
            tau_cmd *= 0.5  # push more gently

        # Convert command torque to actuator current: i = tau / (eta G k_t)
        i_cmd = tau_cmd / gear_eff
        i_cmd = max(-config.i_max, min(config.i_max, i_cmd))
        data.ctrl[motor_id] = i_cmd

        # Patient torque: external torque on the knee joint
        tau_pat = patient_torque(theta, theta_dot, config, profile)
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[knee_dof_id] = tau_pat

        # Step simulation
        mujoco.mj_step(model, data)

        # Log signals
        tau_motor = float(data.actuator_force[motor_id])

        theta_log[step_idx] = theta
        theta_ref_log[step_idx] = theta_ref
        tau_motor_log[step_idx] = tau_motor
        tau_patient_log[step_idx] = tau_pat
        current_log[step_idx] = i_cmd
        mode_log[step_idx] = mode.value

    if viewer_ctx is not None:
        from mujoco import viewer
        with viewer_ctx as v:
            for step in range(n_steps):
                if not v.is_running():
                    break
                step_once(step)
                v.sync()
    else:
        for step in range(n_steps):
            step_once(step)

    # ====== Metrics computation ======
    err = theta_ref_log - theta_log
    rms_error = math.sqrt(np.mean(err ** 2))

    max_tau = float(np.max(np.abs(tau_motor_log)))
    max_i = float(np.max(np.abs(current_log)))

    # Rough estimate of energy consumption: ∫ |τ θ̇| dt
    theta_dot_series = np.gradient(theta_log, dt)
    power_abs = np.abs(tau_motor_log * theta_dot_series)
    energy_abs = float(np.sum(power_abs) * dt)

    results = {
        "t": t_log,
        "theta": theta_log,
        "theta_ref": theta_ref_log,
        "tau_motor": tau_motor_log,
        "tau_patient": tau_patient_log,
        "current": current_log,
        "mode": mode_log,
        "rms_error": rms_error,
        "max_tau": max_tau,
        "max_current": max_i,
        "energy_abs": energy_abs,
        "soft_hits": soft_hits,
        "hard_hits": hard_hits,
        "config": config,
        "controller": controller,
        "patient": patient,
    }
    return results


# ========= 6. Plot helpers =========

def draw_neutral_plots(all_results):
    import matplotlib.pyplot as plt
    res_pd = next(r for r in all_results
                  if r["controller"] == "pd" and r["patient"] == "neutral")
    res_imp = next(r for r in all_results
                   if r["controller"] == "impedance" and r["patient"] == "neutral")

    # Angle tracking
    plt.figure()
    plt.title("Knee Angle Tracking: PD vs Impedance (neutral patient)")
    plt.plot(res_pd["t"], np.rad2deg(res_pd["theta"]), label="theta PD")
    plt.plot(res_pd["t"], np.rad2deg(res_pd["theta_ref"]), "--", label="theta_ref")
    plt.plot(res_imp["t"], np.rad2deg(res_imp["theta"]), label="theta Impedance")
    plt.xlabel("time [s]")
    plt.ylabel("angle [deg]")
    plt.legend()
    plt.grid(True)
    plt.savefig("output/knee_angle_tracking_neutral.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Motor & patient torque
    plt.figure()
    plt.title("Motor torque over time (neutral patient)")
    plt.plot(res_pd["t"], res_pd["tau_motor"], label="Motor PD")
    plt.plot(res_imp["t"], res_imp["tau_motor"], label="Motor Impedance")
    plt.plot(res_pd["t"], res_pd["tau_patient"], "--", label="Patient torque")
    plt.xlabel("time [s]")
    plt.ylabel("torque [Nm]")
    plt.legend()
    plt.grid(True)
    plt.savefig("output/motor_torque_neutral.png", dpi=300, bbox_inches="tight")
    plt.close()


def draw_resist_plots(all_results):
    import matplotlib.pyplot as plt
    res_pd = next(r for r in all_results
                  if r["controller"] == "pd" and r["patient"] == "resist")
    res_imp = next(r for r in all_results
                   if r["controller"] == "impedance" and r["patient"] == "resist")

    # Angle tracking
    plt.figure()
    plt.title("Knee Angle Tracking: PD vs Impedance (resist patient)")
    plt.plot(res_pd["t"], np.rad2deg(res_pd["theta"]), label="theta PD")
    plt.plot(res_pd["t"], np.rad2deg(res_pd["theta_ref"]), "--", label="theta_ref")
    plt.plot(res_imp["t"], np.rad2deg(res_imp["theta"]), label="theta Impedance")
    plt.xlabel("time [s]")
    plt.ylabel("angle [deg]")
    plt.legend()
    plt.grid(True)
    plt.savefig("output/knee_angle_tracking_resist.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Motor torque
    plt.figure()
    plt.title("Motor torque over time (resist patient)")
    plt.plot(res_pd["t"], res_pd["tau_motor"], label="Motor PD")
    plt.plot(res_imp["t"], res_imp["tau_motor"], label="Motor Impedance")
    plt.xlabel("time [s]")
    plt.ylabel("torque [Nm]")
    plt.legend()
    plt.grid(True)
    plt.savefig("output/motor_torque_resist.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Patient torque
    plt.figure()
    plt.title("Patient torque over time (resist patient)")
    plt.plot(res_pd["t"], res_pd["tau_patient"], label="Patient torque")
    plt.xlabel("time [s]")
    plt.ylabel("torque [Nm]")
    plt.legend()
    plt.grid(True)
    plt.savefig("output/patient_torque_resist.png", dpi=300, bbox_inches="tight")
    plt.close()


# ========= 7. Entry point =========

def main():
    import os
    os.makedirs("output", exist_ok=True)

    xml_path = "knee_rom_trainer.xml"
    cfg = ROMConfig()

    cases = [
        ("pd", "neutral"),
        ("impedance", "neutral"),
        ("pd", "resist"),
        ("impedance", "resist"),
        ("pd", "assist"),
        ("impedance", "assist"),
    ]

    all_results = []
    for ctrl, pat in cases:
        print(f"\n=== Simulating controller={ctrl}, patient={pat} ===")
        res = simulate_session(xml_path, ctrl, pat, cfg)
        all_results.append(res)

        print(f"RMS tracking error [rad]: {res['rms_error']:.4f}")
        print(f"Max motor torque [Nm]:   {res['max_tau']:.3f}")
        print(f"Max current [A]:         {res['max_current']:.3f}")
        print(f"Energy |∫τθ̇| dt [J]:    {res['energy_abs']:.3f}")
        print(f"Soft limit hits:         {res['soft_hits']}")
        print(f"Hard limit hits:         {res['hard_hits']}")

    draw_neutral_plots(all_results)
    draw_resist_plots(all_results)

    print("\nPlots saved under ./output/")
    print("From the curves, you can state:")
    print("  - PD control tracks the reference knee angle very well;")
    print("  - Impedance control yields smoother, smaller motor torques,")
    print("    making the interaction more comfortable for the patient.")


if __name__ == "__main__":
    import sys
    if "--viewer" in sys.argv:
        xml_path = "knee_rom_trainer.xml"
        cfg = ROMConfig()
        print("\n=== Launching MuJoCo viewer: controller=pd, patient=neutral ===")
        _ = simulate_session(xml_path, "pd", "neutral", cfg, use_viewer=True)
    else:
        main()