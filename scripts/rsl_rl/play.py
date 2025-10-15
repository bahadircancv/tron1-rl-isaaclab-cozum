"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--debug", action="store_true", default=False, help="Whether to use keyboard.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg,DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, export_mlp_as_onnx, export_policy_as_jit
from isaaclab.devices import Se2Keyboard


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed

    # --- Keyboard input helpers (do NOT change policy obs space) ---
    controller = None
    _advance_with_aim = None  # set if --keyboard

    if args_cli.keyboard:
        # one env, disable timeouts & debug clutter
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False

        controller = Se2Keyboard(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1] / 2,
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )

        aim = {"target": None, "heading_tol": 0.10}  # (tx, ty) in WORLD meters
        # PID gains
        kp_ang, ki_ang, kd_ang = 10.0, 5.00, 0.05
        kp_lin, ki_lin, kd_lin = 0.8, 0.05, 0.05
        # PID state
        pid_state = {"i_lin": 0.0, "i_ang": 0.0, "prev_e_lin": 0.0, "prev_e_ang": 0.0, "initialized": False}
        goal_tol = 0.15  # m

        def _wrap_pi(a: float) -> float:
            import math
            return (a + math.pi) % (2 * math.pi) - math.pi

        def _quat_wxyz_to_yaw(q: torch.Tensor) -> torch.Tensor:
            w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
            return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        def _get_dt(env) -> float:
            default = 1.0 / 60.0
            unwrapped = getattr(env, "unwrapped", env)
            for attr in ("step_dt", "dt"):
                if hasattr(unwrapped, attr):
                    try:
                        return float(getattr(unwrapped, attr))
                    except Exception:
                        pass
            sim = getattr(unwrapped, "sim", None)
            if sim is not None and hasattr(sim, "dt"):
                try:
                    return float(sim.dt)
                except Exception:
                    pass
            return default

        def _clamp(x: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, x))

        def _limit_integral(i_val: float, ki: float, out_limit: float) -> float:
            # Avoid integral windup; critical with manual aim.
            if ki <= 0.0:
                return 0.0
            i_max = abs(out_limit) / ki
            return _clamp(i_val, -i_max, i_max)

        def _advance_with_aim(env) -> torch.Tensor:
            import math

            cmd = list(controller.advance())  # [vx, vy, ωz]
            if args_cli.debug:
                print(f"[DEBUG] cmd: {cmd}")
            if aim["target"] is None:
                cmd[0] = cmd[0] * 3
                # cmd[1] = cmd[1] * 5
                # cmd[2] = cmd[2] * 3
                return torch.tensor(cmd, dtype=torch.float32, device=env.device).unsqueeze(0)

            # Robot pose
            robot = env.unwrapped.scene.articulations["robot"]
            rs = robot.data.root_state_w[:1]
            pos = rs[0, 0:2]
            yaw = float(_quat_wxyz_to_yaw(rs[0, 3:7]).item())

            tx, ty = aim["target"]
            ex = tx - float(pos[0].item())
            ey = ty - float(pos[1].item())
            dist = math.hypot(ex, ey)
            target_yaw = math.atan2(ey, ex)
            e_yaw = _wrap_pi(target_yaw - yaw)

            # Limits
            ranges = env.unwrapped.cfg.commands.base_velocity.ranges
            v_x_max = float(ranges.lin_vel_x[1])
            w_max = float(ranges.ang_vel_z[1])

            dt = _get_dt(env)
            if not pid_state["initialized"]:
                pid_state["prev_e_lin"] = dist
                pid_state["prev_e_ang"] = e_yaw
                pid_state["initialized"] = True

            # Goal reached
            if dist < goal_tol:
                pid_state["i_lin"] = 0.0
                pid_state["i_ang"] = 0.0
                return torch.zeros(1, 3, dtype=torch.float32, device=env.device)

            # Angular PID
            e_ang = e_yaw
            de_ang = (e_ang - pid_state["prev_e_ang"]) / dt if dt > 0 else 0.0
            pid_state["i_ang"] = _limit_integral(pid_state["i_ang"] + e_ang * dt, ki_ang, w_max)
            w_cmd = _clamp(kp_ang * e_ang + ki_ang * pid_state["i_ang"] + kd_ang * de_ang, -w_max, w_max)

            if abs(e_yaw) < aim["heading_tol"]:
                aim["heading_tol"] = 0.20
                e_lin = dist
                de_lin = (e_lin - pid_state["prev_e_lin"]) / dt if dt > 0 else 0.0
                v_limit = (2.0 / 3.0) * v_x_max
                pid_state["i_lin"] = _limit_integral(pid_state["i_lin"] + e_lin * dt, ki_lin, v_limit)
                v_cmd = _clamp(kp_lin * e_lin + ki_lin * pid_state["i_lin"] + kd_lin * de_lin, 0.0, v_limit)
                out = torch.tensor([v_cmd, 0.0, 0.0], dtype=torch.float32, device=env.device).unsqueeze(0)
            else:
                aim["heading_tol"] = 0.10
                pid_state["i_lin"] = 0.0
                pid_state["prev_e_lin"] = dist
                out = torch.tensor([0.0, 0.0, w_cmd], dtype=torch.float32, device=env.device).unsqueeze(0)

            pid_state["prev_e_ang"] = e_ang
            pid_state["prev_e_lin"] = dist
            return out

        # IMPORTANT:
        # Do NOT set env_cfg.observations.policy.velocity_commands = ObsTerm(...)
        # That would change num_actor_obs (+3) and break checkpoint loading.

    # specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    # export policy to onnx
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, export_model_dir
        )
        print("Exported policy as jit script to: ", export_model_dir)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor, 
            export_model_dir, 
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )
    # reset environment
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands") 
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            est = encoder(obs_history)

            # If keyboard, override the command fed to the policy ONLY (keep env obs shape intact).
            if args_cli.keyboard:
                cmd_for_policy = _advance_with_aim(env)
            else:
                cmd_for_policy = commands

            actions = policy(torch.cat((est, obs, cmd_for_policy), dim=-1).detach())

            # env stepping
            obs, _, _, infos = env.step(actions)
            obs_history = infos["observations"].get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = infos["observations"].get("commands") 

    # close the simulator
    env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
