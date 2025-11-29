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
    i_max: float = 60.0      # max current [A]

    # Passive properties (aligned with MJCF / table)
    k_passive: float = 10.0   # joint stiffness [Nm/rad]
    b_passive: float = 0.8    # viscous damping [Nms/rad]
    theta0: float = 0.0       # rest angle [rad]

    # Control parameters
    # PD control
    # Kp_pd: float = 30.0
    Kp_pd: float = 130.0
    # Kd_pd: float = 4.0
    Kd_pd: float = 40.0
    # Impedance control (assistive)
    K_imp: float = 8.0        # impedance stiffness K [Nm/rad]
    B_imp: float = 1.0        # impedance damping B [Nms/rad]

    # Safety thresholds
    # torque_soft: float = 2.0    # Low threshold: slow down
    # torque_hard: float = 3.0    # High threshold: stop and return to 90°
    torque_soft_ratio: float = 0.6
    torque_hard_ratio: float = 0.8

    # Angle ranges (approximately 90° initial & 60° ROM)
    theta_rest: float = math.radians(90.0)  # Rest angle: 90deg
    theta_target: float = math.radians(90.0 + 60.0)  # Target elevation: 60deg
    raise_duration: float = 2.0   # Leg raise time [s]
    return_duration: float = 2.0  # Return-to-90° time [s]

    # Simulation
    total_time: float = 20.0      # Total simulation time [s] (a bit longer for easier observation in UI)
    decimation: int = 10

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
        # Just stay at 90°
        return theta_rest, 0.0, Mode.RAISING

    elif mode == Mode.RAISING:
        frac = min(1.0, t_mode / config.raise_duration)
        theta_ref = theta_rest + frac * (theta_target - theta_rest)
        # Linear interpolation, reference velocity = Δθ / T
        theta_ref_dot = (theta_target - theta_rest) / config.raise_duration
        if frac >= 1.0:
            # After reaching target, start returning
            return theta_ref, 0.0, Mode.RETURNING
        return theta_ref, theta_ref_dot, mode

    elif mode == Mode.RETURNING:
        frac = min(1.0, t_mode / config.return_duration)
        theta_ref = theta_target + frac * (theta_rest - theta_target)
        theta_ref_dot = (theta_rest - theta_target) / config.return_duration
        if frac >= 1.0:
            return theta_rest, 0.0, Mode.RAISING
        return theta_ref, theta_ref_dot, mode

    elif mode == Mode.STOPPED_AT_LIMIT:
        # Safety mode: slowly return to 90°
        frac = min(1.0, t_mode / config.return_duration)
        theta_ref = theta_current + frac * (theta_rest - theta_current)
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
    Not strictly physical; just captures the magnitude “moderate assist / resistance ~ 1 Nm” from the document. [oai_citation:7‡ME 139_239 docs.pdf](sediment://file_00000000856c722f8bcb6f6afa0a2f47)
    """
    base = 1.0  # Nm

    if profile == PatientProfile.NEUTRAL:
        return 0.0
    elif profile == PatientProfile.ASSIST:
        # Patient voluntarily assists: apply a small torque in the direction of motion
        direction = -1.0 if theta_dot < 0 else 1.0
        return base * direction
    elif profile == PatientProfile.RESIST:
        # Patient resists movement: apply a torque opposite to the motion
        direction = -1.0 if theta_dot > 0 else 1.0
        return base * direction
    else:
        return 0.0


# ========= 4. Control laws: PD & Impedance =========

def feedforward_gravity(theta: float, config: ROMConfig) -> float:
    """
    τ_g = m g r sin(theta)  (consistent with PPT) [oai_citation:8‡G12_Range of Motion Trainer.pdf](sediment://file_000000001088722fbcbcc6941616c7b2)
    """
    return -config.m * config.g * config.r * math.sin(theta)


def pd_control(theta, theta_dot, theta_ref, theta_ref_dot, config: ROMConfig):
    e = theta_ref - theta
    print(f"e: {e}")
    # Here we simply use e_dot = θ̇_ref - θ̇
    e_dot = theta_ref_dot - theta_dot
    tau_fb = config.Kp_pd * e + config.Kd_pd * e_dot
    tau_ff = feedforward_gravity(theta, config)
    return tau_ff + tau_fb


def impedance_control(theta, theta_dot, theta_ref, theta_ref_dot, config: ROMConfig):
    """
    Classical impedance: τ = K(θ_ref - θ) + B(θ̇_ref - θ̇)  (K, B as in the table) [oai_citation:9‡ME 139_239 docs.pdf](sediment://file_00000000856c722f8bcb6f6afa0a2f47)
    Here we do not explicitly add feedforward, to emphasize compliant, interaction-dependent behavior.
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
    返回一个字典：包含时间序列与指标
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # If UI is needed, try to launch the official MuJoCo viewer
    viewer_ctx = None
    if use_viewer:
        try:
            from mujoco import viewer
            viewer_ctx = viewer.launch_passive(model, data)
        except Exception as e:
            print(f"[warning] 无法启动 MuJoCo viewer，继续无 UI 仿真: {e}")
            viewer_ctx = None

    # DOF / actuator indices
    knee_qpos_id = model.joint("knee").qposadr[0]
    knee_dof_id = model.joint("knee").dofadr[0]
    motor_id = 0  # only actuator

    dt = model.opt.timestep
    n_steps = int(config.total_time / (dt ))

    gear_eff = config.eta * config.G * config.k_t

    # Logging arrays
    t_log = np.zeros(n_steps)
    theta_log = np.zeros(n_steps)
    theta_ref_log = np.zeros(n_steps)
    tau_motor_log = np.zeros(n_steps)
    tau_patient_log = np.zeros(n_steps)
    current_log = np.zeros(n_steps)
    mode_log = np.zeros(n_steps, dtype=int)

    # Initialize to 90°
    data.qpos[knee_qpos_id] = config.theta_rest
    data.qvel[knee_qpos_id] = 0.0

    mode = Mode.IDLE_AT_90
    mode_time = 0.0
    profile = PatientProfile(patient)

    soft_hits = 0
    hard_hits = 0

    # If viewer is running, wrap it in a context manager to ensure it is closed
    if viewer_ctx is not None:
        with viewer_ctx as v:

            theta = float(data.qpos[knee_qpos_id])
            theta_dot = float(data.qvel[knee_qpos_id])
            theta_ref, theta_ref_dot, next_mode = compute_reference(
                config, mode, mode_time, theta
            )
            log_step_counter = 0
            for _ in range(n_steps):
                if not v.is_running():
                    break


                theta = float(data.qpos[knee_qpos_id])
                theta_dot = float(data.qvel[knee_qpos_id])

                # Reference trajectory + state machine
                if _ % config.decimation  == 0:
                    t = _ * dt
                    t_log[log_step_counter] = t
                    theta_ref, theta_ref_dot, next_mode = compute_reference(
                        config, mode, mode_time, theta
                    )

                if next_mode != mode:
                    mode = next_mode
                    mode_time = 0.0
                else:
                    mode_time += dt

                # 选择控制器
                if controller == "pd":
                    tau_cmd = pd_control(theta, theta_dot, theta_ref, theta_ref_dot, config)
                elif controller == "impedance":
                    tau_cmd = impedance_control(theta, theta_dot, theta_ref, theta_ref_dot, config)
                else:
                    raise ValueError(f"Unknown controller: {controller}")

                # Safety threshold logic (based on previous motor torque)
                # First get the previous step motor torque (0 at the first step)
                tau_motor_prev = float(data.actuator_force[motor_id])

                if abs(tau_motor_prev) > config.torque_hard:
                    hard_hits += 1
                    mode = Mode.STOPPED_AT_LIMIT
                    mode_time = 0.0
                    # Within hard limit: weaken control objective, only return to rest
                    tau_cmd = 0.0

                elif abs(tau_motor_prev) > config.torque_soft:
                    soft_hits += 1
                    # Soft limit: reduce control gains (equivalent to “push slowly”)
                    tau_cmd *= 0.5

                # Convert to current and apply saturation: i = tau / (eta G k_t)
                i_cmd = tau_cmd / gear_eff
                i_cmd = max(-config.i_max, min(config.i_max, i_cmd))
                data.ctrl[motor_id] = i_cmd

                # Patient torque (external torque)
                tau_pat = patient_torque(theta, theta_dot, config, profile)
                # At each step, clear external forces first
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[knee_dof_id] = tau_pat

                # Advance one simulation step
                mujoco.mj_step(model, data)

                # Log: use current step motor torque (computed by MuJoCo dynamics)
                tau_motor = float(data.actuator_force[motor_id])

                if _ % config.decimation  == 0:
                    theta_log[log_step_counter] = theta
                    theta_ref_log[log_step_counter] = theta_ref
                    tau_motor_log[log_step_counter] = tau_motor
                    tau_patient_log[log_step_counter] = tau_pat
                    current_log[log_step_counter] = i_cmd
                    mode_log[log_step_counter] = mode.value
                    log_step_counter += 1

                # Sync with UI (render at monitor refresh rate)
                v.sync()
    else:

        for step in range(n_steps):
            t = step * dt
            t_log[step] = t

            theta = float(data.qpos[knee_qpos_id])
            theta_dot = float(data.qvel[knee_qpos_id])

            # Reference trajectory + state machine
            # add decimation
            
            theta_ref, theta_ref_dot, next_mode = compute_reference(
                config, mode, mode_time, theta
            )

            if next_mode != mode:
                mode = next_mode
                mode_time = 0.0
            else:
                mode_time += dt

            # 选择控制器
            if controller == "pd":
                tau_cmd = pd_control(theta, theta_dot, theta_ref, theta_ref_dot, config)
            elif controller == "impedance":
                tau_cmd = impedance_control(theta, theta_dot, theta_ref, theta_ref_dot, config)
            else:
                raise ValueError(f"Unknown controller: {controller}")

            # Safety threshold logic (based on previous motor torque)
            # First get the previous step motor torque (0 at the first step)
            tau_motor_prev = float(data.actuator_force[motor_id])

            if abs(tau_motor_prev) > config.torque_hard:
                hard_hits += 1
                mode = Mode.STOPPED_AT_LIMIT
                mode_time = 0.0
                # Within hard limit: weaken control objective, only return to rest
                tau_cmd = 0.0

            elif abs(tau_motor_prev) > config.torque_soft:
                soft_hits += 1
                # Soft limit: reduce control gains (equivalent to “push slowly”)
                tau_cmd *= 0.5

            # Convert to current and apply saturation: i = tau / (eta G k_t)
            i_cmd = tau_cmd / gear_eff
            i_cmd = max(-config.i_max, min(config.i_max, i_cmd))
            data.ctrl[motor_id] = i_cmd

            # Patient torque (external torque)
            tau_pat = patient_torque(theta, theta_dot, config, profile)
            # At each step, clear external forces first
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[knee_dof_id] = tau_pat

            # Advance one simulation step
            mujoco.mj_step(model, data)

            # Log: use current step motor torque (computed by MuJoCo dynamics)
            tau_motor = float(data.actuator_force[motor_id])

            theta_log[step] = theta
            theta_ref_log[step] = theta_ref
            tau_motor_log[step] = tau_motor
            tau_patient_log[step] = tau_pat
            current_log[step] = i_cmd
            mode_log[step] = mode.value

    # ====== Metrics computation ======
    # RMS tracking error
    err = theta_ref_log - theta_log
    rms_error = math.sqrt(np.mean(err ** 2))

    # Maximum torque / current
    max_tau = float(np.max(np.abs(tau_motor_log)))
    max_i = float(np.max(np.abs(current_log)))

    # Rough estimate of energy consumption: ∫ |τ θ̇| dt
    # Note: θ̇ uses qvel from the simulation
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


# ========= 6. Run multiple cases for comparison =========

def main():
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
        print(f"Max current [A]:        {res['max_current']:.3f}")
        print(f"Energy |∫τθ̇| dt [J]:   {res['energy_abs']:.3f}")
        print(f"Soft limit hits:        {res['soft_hits']}")
        print(f"Hard limit hits:        {res['hard_hits']}")

    # You can add matplotlib plots here for comparison (if you want figures in the report)
    # try:
    draw_neutral_plots(all_results)
    draw_resist_plots(all_results)
    # import matplotlib.pyplot as plt

    # # 示例：画 pd vs impedance 在 neutral 病人下的轨迹对比
    # res_pd = next(r for r in all_results
    #                 if r["controller"] == "pd" and r["patient"] == "neutral")
    # res_imp = next(r for r in all_results
    #                 if r["controller"] == "impedance" and r["patient"] == "neutral")

    # plt.figure()
    # plt.title("Knee Angle Tracking: PD vs Impedance (neutral patient)")
    # plt.plot(res_pd["t"], np.rad2deg(res_pd["theta"]), label="theta PD")
    # plt.plot(res_pd["t"], np.rad2deg(res_pd["theta_ref"]), "--", label="theta_ref")
    # plt.plot(res_imp["t"], np.rad2deg(res_imp["theta"]), label="theta Impedance")
    # plt.xlabel("time [s]")
    # # plt.ylabel("angle [rad]")
    # plt.ylabel("angle [deg]")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("knee_angle_tracking_neutral.png", dpi=300, bbox_inches="tight")
    # plt.close()

    # plt.figure()
    # plt.title("Motor torque over time (neutral patient)")
    # plt.plot(res_pd["t"], res_pd["tau_motor"], label="PD")
    # plt.plot(res_imp["t"], res_imp["tau_motor"], label="Impedance")
    # plt.xlabel("time [s]")
    # plt.ylabel("torque [Nm]")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("motor_torque_neutral.png", dpi=300, bbox_inches="tight")
    # plt.close()

    # except ImportError:
    #     print("\nmatplotlib not installed – skipping plots. "
    #           "Install with `pip install matplotlib` to see figures.")

def draw_neutral_plots(all_results):
    import matplotlib.pyplot as plt
    # Example: plot PD vs impedance trajectories under a neutral patient
    res_pd = next(r for r in all_results
                    if r["controller"] == "pd" and r["patient"] == "neutral")
    res_imp = next(r for r in all_results
                    if r["controller"] == "impedance" and r["patient"] == "neutral")

    plt.figure()
    plt.title("Knee Angle Tracking: PD vs Impedance (neutral patient)")
    plt.plot(res_pd["t"], np.rad2deg(res_pd["theta"]), label="theta PD")
    plt.plot(res_pd["t"], np.rad2deg(res_pd["theta_ref"]), "--", label="theta_ref")
    plt.plot(res_imp["t"], np.rad2deg(res_imp["theta"]), label="theta Impedance")
    plt.xlabel("time [s]")
    # plt.ylabel("angle [rad]")
    plt.ylabel("angle [deg]")
    plt.legend()
    plt.grid(True)
    plt.savefig("knee_angle_tracking_neutral.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.title("Motor torque over time (neutral patient)")
    plt.plot(res_pd["t"], res_pd["tau_motor"], label="PD")
    plt.plot(res_imp["t"], res_imp["tau_motor"], label="Impedance")
    # patient torque
    plt.plot(res_pd["t"], res_pd["tau_patient"], label="Patient")
    plt.plot(res_imp["t"], res_imp["tau_patient"], label="Patient")
    plt.xlabel("time [s]")
    plt.ylabel("torque [Nm]")
    plt.legend()
    plt.grid(True)
    plt.savefig("motor_torque_neutral.png", dpi=300, bbox_inches="tight")
    plt.close()

def draw_resist_plots(all_results):
    import matplotlib.pyplot as plt
    res_pd = next(r for r in all_results
                    if r["controller"] == "pd" and r["patient"] == "resist")
    res_imp = next(r for r in all_results
                    if r["controller"] == "impedance" and r["patient"] == "resist")
    plt.figure()
    plt.title("Knee Angle Tracking: PD vs Impedance (resist patient)")
    plt.plot(res_pd["t"], np.rad2deg(res_pd["theta"]), label="theta PD")
    plt.plot(res_pd["t"], np.rad2deg(res_pd["theta_ref"]), "--", label="theta_ref")
    plt.plot(res_imp["t"], np.rad2deg(res_imp["theta"]), label="theta Impedance")
    plt.xlabel("time [s]")
    plt.ylabel("angle [deg]")
    plt.legend()
    plt.grid(True)
    plt.savefig("knee_angle_tracking_resist.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.title("Motor torque over time (resist patient)")
    plt.plot(res_pd["t"], res_pd["tau_motor"], label="PD")
    plt.plot(res_imp["t"], res_imp["tau_motor"], label="Impedance")
    plt.xlabel("time [s]")
    plt.ylabel("torque [Nm]")
    plt.legend()
    plt.grid(True)
    plt.savefig("motor_torque_resist.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.title("Patient torque over time (resist patient)")
    plt.plot(res_pd["t"], res_pd["tau_patient"], label="Patient")
    plt.plot(res_imp["t"], res_imp["tau_patient"], label="Patient")
    plt.xlabel("time [s]")
    plt.ylabel("torque [Nm]")
    plt.legend()
    plt.grid(True)
    plt.savefig("patient_torque_resist.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # If you want to see the MuJoCo UI, you can run:
    #   python test.py --viewer
    # By default, it still batch-runs all cases and outputs metrics and PNG figures.
    import sys
    if "--viewer" in sys.argv:
        xml_path = "knee_rom_trainer.xml"
        cfg = ROMConfig()
        print("\n=== Launching MuJoCo viewer: controller=pd, patient=neutral ===")
        results = simulate_session(xml_path, "pd", "neutral", cfg, use_viewer=True)
        # draw_plots(results)
    else:
        main()