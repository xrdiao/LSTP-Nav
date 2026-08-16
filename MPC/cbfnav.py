from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    class _TqdmFallback:
        def __init__(self, iterable, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, *args, **kwargs):
            return None

    def tqdm(iterable, **kwargs):
        return _TqdmFallback(iterable, **kwargs)

try:
    import casadi as ca
except ImportError as exc:  # pragma: no cover
    ca = None
    _CASADI_IMPORT_ERROR = exc
else:  # pragma: no cover
    _CASADI_IMPORT_ERROR = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env_sim.argument import LASER_NUM, MAX_SPEED, ROBOT_WIDTH
from train import create_env


DEFAULT_ROBOT_NUMS = (1, 5, 10)
DEFAULT_OBSTACLE_NUMS = (5, 10, 15, 20, 25, 30, 35)
DEFAULT_EVAL_TIMES = 300
DEFAULT_ENV_NAME = "circle"
DEFAULT_ENV_RADIUS = 17.0
DEFAULT_MAX_EPISODE_STEPS = 6000
DEFAULT_PARALLEL_WORKERS = 4

DEFAULT_PREDICTION_HORIZON = 10
DEFAULT_OBS_PREDICTION_HORIZON = 10
DEFAULT_MAX_OBSTACLES = 8
DEFAULT_SAFETY_MARGIN = 0.5
DEFAULT_CBF_GAMMA = 0.7
DEFAULT_W_POS = 5.0
DEFAULT_W_VEL = 0.01
DEFAULT_W_U = 0.0
DEFAULT_TERMINAL_WEIGHT = 10.0
DEFAULT_MAX_ACCELERATION = 1.0


@dataclass
class RobotState:
    idx: int
    position: np.ndarray
    velocity: np.ndarray
    goal: np.ndarray
    radius: float
    max_speed: float
    max_acceleration: float


@dataclass
class ObstacleState:
    position: np.ndarray
    velocity: np.ndarray
    radius: float


def ensure_casadi_available():
    if ca is None:  # pragma: no cover
        raise ImportError(
            "casadi is required to run CBFNav. Install it in the active environment first."
        ) from _CASADI_IMPORT_ERROR


def ensure_repo_root_as_cwd():
    if Path.cwd().resolve() != REPO_ROOT:
        os.chdir(REPO_ROOT)


def smooth_brake(v: np.ndarray, a_max: float, k: float = 5.0, v_deadzone: float = 0.02) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    speed = float(np.linalg.norm(v))
    if speed < v_deadzone or a_max <= 0:
        return np.zeros(2, dtype=float)

    denom = max(speed, a_max / k)
    a_cmd = -(a_max / denom) * v
    a_norm = float(np.linalg.norm(a_cmd))
    if a_norm > a_max:
        a_cmd *= a_max / a_norm
    return a_cmd


def obstacle_record_to_radius(record: dict) -> float:
    shape_type = record.get("shape_type", "BOX")
    if shape_type == "BOX":
        size_x = float(record.get("size_x", 1.0))
        size_y = float(record.get("size_y", 1.0))
        return 0.5 * float(np.hypot(size_x, size_y))
    if shape_type in ("CYLINDER", "SPHERE"):
        return float(record.get("radius", 0.5))
    if shape_type == "CAPSULE":
        radius = float(record.get("radius", 0.5))
        length = float(record.get("length", 2.0))
        return radius + length / 2.0
    return 0.5


def extract_planar_velocity(robot) -> np.ndarray:
    robot.get_vel_and_pos()
    planar_velocity = np.asarray(getattr(robot, "cur_vel", np.zeros(2, dtype=float)), dtype=float).reshape(-1)
    if planar_velocity.shape[0] >= 2:
        return planar_velocity[:2]
    if planar_velocity.shape[0] == 1:
        return np.array(
            [
                float(planar_velocity[0] * np.cos(robot.theta)),
                float(planar_velocity[0] * np.sin(robot.theta)),
            ],
            dtype=float,
        )
    return np.zeros(2, dtype=float)


class EnvStateAdapter:
    def __init__(
        self,
        env,
        robot_radius: float = ROBOT_WIDTH,
        max_speed: float = MAX_SPEED,
        max_acceleration: float = DEFAULT_MAX_ACCELERATION,
    ):
        self.env = env
        self.robot_radius = float(robot_radius)
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)

    def get_robot_states(self) -> list[RobotState]:
        states = []
        for idx, robot in enumerate(self.env.robots):
            states.append(
                RobotState(
                    idx=idx,
                    position=np.asarray(robot.cur_pos, dtype=float),
                    velocity=extract_planar_velocity(robot),
                    goal=np.asarray(robot.target_pos, dtype=float),
                    radius=self.robot_radius,
                    max_speed=self.max_speed,
                    max_acceleration=self.max_acceleration,
                )
            )
        return states

    def get_static_obstacles(self) -> list[ObstacleState]:
        obstacles = []
        for record in self.env.get_obstacle_records():
            obstacles.append(
                ObstacleState(
                    position=np.asarray([record["x"], record["y"]], dtype=float),
                    velocity=np.zeros(2, dtype=float),
                    radius=obstacle_record_to_radius(record),
                )
            )
        return obstacles

    def snapshot(self) -> tuple[list[RobotState], list[ObstacleState]]:
        return self.get_robot_states(), self.get_static_obstacles()


class CBFNavPlanner:
    def __init__(
        self,
        dt: float = 0.1,
        prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
        obs_prediction_horizon: int = DEFAULT_OBS_PREDICTION_HORIZON,
        max_obstacles: int = DEFAULT_MAX_OBSTACLES,
        max_speed: float = MAX_SPEED,
        robot_radius: float = ROBOT_WIDTH,
        safety_margin: float = DEFAULT_SAFETY_MARGIN,
        cbf_gamma: float = DEFAULT_CBF_GAMMA,
        w_pos: float = DEFAULT_W_POS,
        w_vel: float = DEFAULT_W_VEL,
        w_u: float = DEFAULT_W_U,
        terminal_weight: float = DEFAULT_TERMINAL_WEIGHT,
    ):
        ensure_casadi_available()
        self.dt = float(dt)
        self.prediction_horizon = int(prediction_horizon)
        self.obs_prediction_horizon = int(obs_prediction_horizon)
        self.max_obstacles = int(max_obstacles)
        self.max_speed = float(max_speed)
        self.robot_radius = float(robot_radius)
        self.safety_margin = float(safety_margin)
        self.cbf_gamma = float(cbf_gamma)
        self.w_pos = float(w_pos)
        self.w_vel = float(w_vel)
        self.w_u = float(w_u)
        self.terminal_weight = float(terminal_weight)

        self.prev_solution: tuple[list[list[float]], list[list[float]]] | None = None
        self.predicted_trajectory: list[np.ndarray] = []
        self.solve_failures = 0

        self._build_solver()

    def _build_solver(self):
        n_states = 2
        n_controls = 2
        self.obs_params_count = 6  # x, y, vx, vy, radius, mask

        opt_vars = []
        lbx = []
        ubx = []
        g = []
        lbg = []
        ubg = []

        x0 = ca.SX.sym("X0", n_states)
        opt_vars.append(x0)
        lbx.extend([-ca.inf] * n_states)
        ubx.extend([ca.inf] * n_states)

        param_dim = n_states * 2 + self.max_obstacles * self.obs_params_count
        params = ca.SX.sym("P", param_dim)

        g.append(x0 - params[:2])
        lbg.extend([0.0] * n_states)
        ubg.extend([0.0] * n_states)

        x_seq = [x0]
        u_seq = []
        cost = 0
        goal_state = params[2:4]

        for step_idx in range(self.prediction_horizon):
            u_k = ca.SX.sym(f"U_{step_idx}", n_controls)
            opt_vars.append(u_k)
            u_seq.append(u_k)
            lbx.extend([-self.max_speed, -self.max_speed])
            ubx.extend([self.max_speed, self.max_speed])

            x_next_var = ca.SX.sym(f"X_{step_idx + 1}", n_states)
            opt_vars.append(x_next_var)
            x_seq.append(x_next_var)
            lbx.extend([-ca.inf] * n_states)
            ubx.extend([ca.inf] * n_states)

            x_next = x_seq[step_idx] + u_k * self.dt
            g.append(x_next_var - x_next)
            lbg.extend([0.0] * n_states)
            ubg.extend([0.0] * n_states)

            cost += self.w_pos * ca.sumsqr(x_seq[step_idx] - goal_state)
            cost += self.w_vel * ca.sumsqr(u_k)
            if step_idx > 0 and self.w_u > 0:
                cost += self.w_u * ca.sumsqr(u_k - u_seq[step_idx - 1])

            if step_idx < self.obs_prediction_horizon:
                for obs_idx in range(self.max_obstacles):
                    obs_start = n_states * 2 + obs_idx * self.obs_params_count
                    obs_x = params[obs_start]
                    obs_y = params[obs_start + 1]
                    obs_vx = params[obs_start + 2]
                    obs_vy = params[obs_start + 3]
                    obs_radius = params[obs_start + 4]
                    obs_mask = params[obs_start + 5]

                    pred_obs_x = obs_x + step_idx * self.dt * obs_vx
                    pred_obs_y = obs_y + step_idx * self.dt * obs_vy

                    dx = x_seq[step_idx][0] - pred_obs_x
                    dy = x_seq[step_idx][1] - pred_obs_y
                    safe_dist = self.robot_radius + obs_radius + self.safety_margin
                    h = dx * dx + dy * dy - safe_dist * safe_dist
                    rel_vx = u_k[0] - obs_vx
                    rel_vy = u_k[1] - obs_vy
                    cbf_constraint = 2 * dx * rel_vx + 2 * dy * rel_vy + self.cbf_gamma * h

                    g.append(obs_mask * cbf_constraint + (1.0 - obs_mask) * 1000.0)
                    lbg.append(0.0)
                    ubg.append(ca.inf)

        cost += self.terminal_weight * self.w_pos * ca.sumsqr(x_seq[-1] - goal_state)

        opt_vars = ca.vertcat(*opt_vars)
        nlp = {"f": cost, "x": opt_vars, "g": ca.vertcat(*g), "p": params}
        self.solver = ca.nlpsol(
            "solver",
            "ipopt",
            nlp,
            {
                "ipopt": {
                    "max_iter": 100,
                    "print_level": 0,
                    "acceptable_tol": 1e-3,
                },
                "print_time": 0,
            },
        )

        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg
        self.n_states = n_states
        self.n_controls = n_controls

    def reset_warm_start(self):
        self.prev_solution = None
        self.predicted_trajectory = []

    def plan(self, robot: RobotState, obstacles: list[ObstacleState]) -> tuple[np.ndarray, float]:
        current_state = np.asarray(robot.position, dtype=float)
        goal_state = np.asarray(robot.goal, dtype=float)

        nearest_obstacles = sorted(
            obstacles,
            key=lambda obs: float(np.linalg.norm(np.asarray(obs.position, dtype=float) - current_state)),
        )

        obs_data = np.zeros(self.max_obstacles * self.obs_params_count, dtype=float)
        for idx, obs in enumerate(nearest_obstacles[: self.max_obstacles]):
            start = idx * self.obs_params_count
            obs_data[start] = float(obs.position[0])
            obs_data[start + 1] = float(obs.position[1])
            obs_data[start + 2] = float(obs.velocity[0])
            obs_data[start + 3] = float(obs.velocity[1])
            obs_data[start + 4] = float(obs.radius)
            obs_data[start + 5] = 1.0

        param_vector = np.concatenate([current_state, goal_state, obs_data])
        x0_guess = self._build_initial_guess(current_state, goal_state)

        start_t = time.time()
        try:
            sol = self.solver(
                x0=x0_guess,
                p=param_vector,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
            )
            if not bool(self.solver.stats().get("success", False)):
                raise RuntimeError("CBFNav solver failed")

            opt_vars = sol["x"].full().flatten()
            desired_velocity = np.asarray(
                [opt_vars[self.n_states], opt_vars[self.n_states + 1]],
                dtype=float,
            )
            speed = float(np.linalg.norm(desired_velocity))
            if speed > robot.max_speed and speed > 0:
                desired_velocity = desired_velocity / speed * robot.max_speed

            self._update_solution_cache(opt_vars, current_state)
            return desired_velocity, (time.time() - start_t) * 1000.0
        except Exception:
            self.solve_failures += 1
            self.reset_warm_start()
            fallback_velocity = robot.velocity + smooth_brake(robot.velocity, robot.max_acceleration) * self.dt
            speed = float(np.linalg.norm(fallback_velocity))
            if speed > robot.max_speed and speed > 0:
                fallback_velocity = fallback_velocity / speed * robot.max_speed
            return fallback_velocity, (time.time() - start_t) * 1000.0

    def _build_initial_guess(self, current_state: np.ndarray, goal_state: np.ndarray) -> list[float]:
        if self.prev_solution is not None:
            prev_u_opt, prev_x_opt = self.prev_solution
            x0_guess = []
            x0_guess.extend(current_state.tolist())
            for step_idx in range(self.prediction_horizon):
                if step_idx + 1 < len(prev_u_opt):
                    u_guess = prev_u_opt[step_idx + 1]
                else:
                    u_guess = prev_u_opt[-1]

                x0_guess.extend(list(u_guess))
                state_guess = prev_x_opt[min(step_idx + 1, len(prev_x_opt) - 1)]
                x0_guess.extend(list(state_guess))
            return x0_guess

        direction = np.asarray(goal_state - current_state, dtype=float)
        distance = float(np.linalg.norm(direction))
        if distance > 1e-6:
            guide_velocity = direction / distance * min(self.max_speed, distance / max(self.dt, 1e-6))
        else:
            guide_velocity = np.zeros(2, dtype=float)

        x0_guess = []
        x0_guess.extend(current_state.tolist())
        state_rollout = np.asarray(current_state, dtype=float).copy()
        for _ in range(self.prediction_horizon):
            x0_guess.extend(guide_velocity.tolist())
            state_rollout = state_rollout + guide_velocity * self.dt
            x0_guess.extend(state_rollout.tolist())
        return x0_guess

    def _update_solution_cache(self, opt_vars: np.ndarray, current_state: np.ndarray):
        u_opt = []
        x_opt = [np.asarray(current_state, dtype=float).tolist()]

        idx = self.n_states
        for _ in range(self.prediction_horizon):
            u_k = [float(opt_vars[idx]), float(opt_vars[idx + 1])]
            u_opt.append(u_k)
            idx += self.n_controls

            x_k = [float(opt_vars[idx]), float(opt_vars[idx + 1])]
            x_opt.append(x_k)
            idx += self.n_states

        self.prev_solution = (u_opt, x_opt)
        self.predicted_trajectory = [np.asarray(state, dtype=float) for state in x_opt]


class ActionProjector:
    def __init__(self, env):
        self.env = env

    def project_velocity(self, robot_idx: int, desired_velocity: np.ndarray) -> list[float]:
        return self.env.robots[robot_idx].cal_effective_cmd(np.asarray(desired_velocity, dtype=float))


class CBFNavController:
    def __init__(self, env):
        self.env = env
        self.adapter = EnvStateAdapter(env)
        self.planner_kwargs = {
            "dt": getattr(env, "delta_time", 0.1),
            "max_speed": MAX_SPEED,
            "robot_radius": ROBOT_WIDTH,
        }
        self.planners: list[CBFNavPlanner] = []
        self._sync_planners()
        self.projector = ActionProjector(env)

    def _sync_planners(self):
        target_count = len(self.env.robots)
        current_count = len(self.planners)
        if current_count < target_count:
            self.planners.extend(CBFNavPlanner(**self.planner_kwargs) for _ in range(target_count - current_count))
        elif current_count > target_count:
            self.planners = self.planners[:target_count]

    def reset_warm_starts(self):
        self._sync_planners()
        for planner in self.planners:
            planner.reset_warm_start()

    def compute_actions(self) -> tuple[list[list[float]], list[float]]:
        self._sync_planners()
        robot_states, static_obstacles = self.adapter.snapshot()
        actions = []
        solve_times = []

        for robot_state in robot_states:
            dynamic_obstacles = [
                ObstacleState(position=other.position, velocity=other.velocity, radius=other.radius)
                for other in robot_states
                if other.idx != robot_state.idx
            ]
            combined_obstacles = dynamic_obstacles + static_obstacles
            desired_velocity, solve_time_ms = self.planners[robot_state.idx].plan(robot_state, combined_obstacles)
            actions.append(self.projector.project_velocity(robot_state.idx, desired_velocity))
            solve_times.append(solve_time_ms)

        return actions, solve_times

    @property
    def solve_failures(self) -> int:
        return sum(planner.solve_failures for planner in self.planners)


class CBFNavEvaluator:
    def __init__(self, env, agent_name: str = "CBFNav"):
        self.env = env
        self.agent_name = agent_name
        self.controller = CBFNavController(env)
        self.robots_num = 0

    def evaluate(self, times: int = DEFAULT_EVAL_TIMES, show_progress: bool = True) -> dict:
        next_obs, _ = self.env.reset()
        _ = next_obs

        collision_times = 0
        reach_times = 0
        trap_times = 0
        tot_time = 0.0
        tot_step = 0

        ep_avg_reward_list = []
        ep_reach_rate_list = []
        ep_trap_rate_list = []
        ep_collision_rate_list = []
        ep_avg_time_list = []
        ep_avg_step_list = []
        all_solve_times = []

        self.robots_num = len(self.env.robots)
        tq_bar = tqdm(range(1, times + 1), desc=self.agent_name) if show_progress else range(1, times + 1)

        for episode_idx in tq_bar:
            done = [False] * self.robots_num
            start_time = time.time()
            episode_rewards = []
            self.controller.reset_warm_starts()

            while True:
                actions, solve_times = self.controller.compute_actions()
                all_solve_times.extend(solve_times)

                for idx, robot in enumerate(self.env.robots):
                    if robot.reach_goal:
                        actions[idx] = [0.0, 0.0]

                next_obs, reward, te, tr, _ = self.env.step(actions)
                _ = next_obs
                episode_rewards.append(reward)

                for robot in self.env.robots:
                    if robot.collision_num == 0 and robot.reach_goal and not robot.end_test:
                        reach_times += 1
                        robot.end_test = True
                        tot_time += time.time() - start_time
                        tot_step += self.env.simulate_steps

                done = [i or j or d for i, j, d in zip(te, tr, done)]

                if all(done):
                    ep_return_vec = np.sum(np.asarray(episode_rewards), axis=0)

                    for robot in self.env.robots:
                        if not robot.end_test:
                            if robot.collision_num == 0:
                                trap_times += 1
                            else:
                                collision_times += 1

                    cur_reach_rate = reach_times / episode_idx / self.robots_num
                    cur_trap_rate = trap_times / episode_idx / self.robots_num
                    cur_collision_rate = collision_times / episode_idx / self.robots_num
                    cur_avg_time = tot_time / (reach_times + 1e-8)
                    cur_avg_step = tot_step / (reach_times + 1e-8)

                    if show_progress:
                        tq_bar.set_postfix(
                            {
                                "reach_rate": f"{cur_reach_rate:.2f}",
                                "trap_rate": f"{cur_trap_rate:.2f}",
                                "collision_rate": f"{cur_collision_rate:.2f}",
                                "avg_time": f"{cur_avg_time:.2f}",
                                "avg_step": f"{cur_avg_step:.2f}",
                            }
                        )

                    ep_avg_reward_list.append(float(np.mean(ep_return_vec)))
                    ep_reach_rate_list.append(float(cur_reach_rate))
                    ep_trap_rate_list.append(float(cur_trap_rate))
                    ep_collision_rate_list.append(float(cur_collision_rate))
                    ep_avg_time_list.append(float(cur_avg_time))
                    ep_avg_step_list.append(float(cur_avg_step))

                    next_obs = self.env.reset(tr=done, te=done)[0]
                    _ = next_obs
                    break

                next_obs = self.env.reset(tr=tr, te=te)[0] if all(item for item in te) else next_obs
                _ = next_obs

        tot_test_times = times * self.robots_num
        data_dict = {
            "avg_rewards": float(np.mean(ep_avg_reward_list)) if ep_avg_reward_list else 0.0,
            "collision_rate": collision_times / max(tot_test_times, 1),
            "reach_rate": reach_times / max(tot_test_times, 1),
            "trap_rate": trap_times / max(tot_test_times, 1),
            "avg_time": tot_time / (reach_times + 1e-8),
            "avg_step": tot_step / (reach_times + 1e-8),
            "SR": reach_times / max(tot_test_times, 1),
            "CR": collision_times / max(tot_test_times, 1),
            "TR": trap_times / max(tot_test_times, 1),
            "AT": tot_time / (reach_times + 1e-8),
            "AS": tot_step / (reach_times + 1e-8),
            "var_avg_rewards": float(np.var(ep_avg_reward_list, ddof=0)) if ep_avg_reward_list else 0.0,
            "var_reach_rate": float(np.var(ep_reach_rate_list, ddof=0)) if ep_reach_rate_list else 0.0,
            "var_trap_rate": float(np.var(ep_trap_rate_list, ddof=0)) if ep_trap_rate_list else 0.0,
            "var_collision_rate": float(np.var(ep_collision_rate_list, ddof=0)) if ep_collision_rate_list else 0.0,
            "var_avg_time": float(np.var(ep_avg_time_list, ddof=0)) if ep_avg_time_list else 0.0,
            "var_avg_step": float(np.var(ep_avg_step_list, ddof=0)) if ep_avg_step_list else 0.0,
            "avg_solve_time_ms": float(np.mean(all_solve_times)) if all_solve_times else 0.0,
            "var_solve_time_ms": float(np.var(all_solve_times, ddof=0)) if all_solve_times else 0.0,
            "solve_failures": int(self.controller.solve_failures),
            "agent_name": self.agent_name,
            "env_name": self.env.name,
            "test_times": int(times),
            "obstacles": self.env.random_obstacles,
            "laser_num": LASER_NUM,
            "robots_num": self.env.robots_num,
            "x_lim": self.env.x_lim,
            "y_lim": self.env.y_lim,
        }

        output_dir = REPO_ROOT / "data"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{self.agent_name}_{self.env.robots_num}_{self.env.random_obstacles}.json"
        output_path.write_text(json.dumps(data_dict, sort_keys=False, indent=4, separators=(",", ": ")))
        return data_dict


def build_env(
    robot_num: int = 1,
    obs_num: int = 15,
    *,
    render: bool = False,
    radius: float = DEFAULT_ENV_RADIUS,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
):
    ensure_repo_root_as_cwd()
    env, _ = create_env(
        render=render,
        name=DEFAULT_ENV_NAME,
        robot_num=robot_num,
        obstacle_num=obs_num,
        radius=radius,
        cli_args=[],
    )
    env.set_max_step(max_episode_steps)
    return env


def evaluate_single_setting(
    robot_num: int = 1,
    obs_num: int = 15,
    *,
    times: int = DEFAULT_EVAL_TIMES,
    render: bool = False,
    radius: float = DEFAULT_ENV_RADIUS,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    show_progress: bool = True,
    print_stats: bool = True,
) -> dict:
    env = build_env(
        robot_num=robot_num,
        obs_num=obs_num,
        render=render,
        radius=radius,
        max_episode_steps=max_episode_steps,
    )
    try:
        if print_stats:
            print("policy:", "CBFNav", "robot_nums:", env.robots_num, "obstacle_nums:", env.random_obstacles)
        evaluator = CBFNavEvaluator(env)
        stats = evaluator.evaluate(times=times, show_progress=show_progress)
        if print_stats:
            print(stats)
        return stats
    finally:
        env.close()


def _parallel_worker(task: tuple[int, int, int, bool, float, int]) -> tuple[tuple[int, int], dict]:
    robot_num, obs_num, times, render, radius, max_episode_steps = task
    stats = evaluate_single_setting(
        robot_num=robot_num,
        obs_num=obs_num,
        times=times,
        render=render,
        radius=radius,
        max_episode_steps=max_episode_steps,
        show_progress=False,
        print_stats=False,
    )
    return (robot_num, obs_num), stats


def run_experiments(
    robot_nums: Iterable[int] = DEFAULT_ROBOT_NUMS,
    obstacle_nums: Iterable[int] = DEFAULT_OBSTACLE_NUMS,
    *,
    times: int = DEFAULT_EVAL_TIMES,
    render: bool = False,
    radius: float = DEFAULT_ENV_RADIUS,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
) -> dict[tuple[int, int], dict]:
    all_stats = {}
    for robot_num in robot_nums:
        for obs_num in obstacle_nums:
            all_stats[(robot_num, obs_num)] = evaluate_single_setting(
                robot_num=robot_num,
                obs_num=obs_num,
                times=times,
                render=render,
                radius=radius,
                max_episode_steps=max_episode_steps,
            )
    return all_stats


def run_experiments_parallel(
    robot_nums: Iterable[int] = DEFAULT_ROBOT_NUMS,
    obstacle_nums: Iterable[int] = DEFAULT_OBSTACLE_NUMS,
    *,
    times: int = DEFAULT_EVAL_TIMES,
    render: bool = False,
    radius: float = DEFAULT_ENV_RADIUS,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    num_workers: int | None = None,
) -> dict[tuple[int, int], dict]:
    tasks = [
        (robot_num, obs_num, times, render, radius, max_episode_steps)
        for robot_num in robot_nums
        for obs_num in obstacle_nums
    ]
    if not tasks:
        return {}

    worker_count = num_workers or min(len(tasks), os.cpu_count() or 1)
    worker_count = max(1, min(worker_count, len(tasks)))

    all_stats = {}
    ctx = get_context("spawn")
    with ctx.Pool(processes=worker_count) as pool:
        for key, stats in tqdm(
            pool.imap_unordered(_parallel_worker, tasks),
            total=len(tasks),
            desc="CBFNav batch",
        ):
            all_stats[key] = stats
            print(
                f"finished robot={key[0]} obs={key[1]} "
                f"reach={stats['reach_rate']:.3f} collision={stats['collision_rate']:.3f}"
            )
    return all_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Run CBFNav evaluation on my_env circle.")
    parser.add_argument("--robot-nums", type=int, nargs="*", default=list(DEFAULT_ROBOT_NUMS))
    parser.add_argument("--obstacle-nums", type=int, nargs="*", default=list(DEFAULT_OBSTACLE_NUMS))
    parser.add_argument("--times", type=int, default=DEFAULT_EVAL_TIMES)
    parser.add_argument("--radius", type=float, default=DEFAULT_ENV_RADIUS)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=DEFAULT_PARALLEL_WORKERS,
        help="0 means sequential run; positive values use spawn-based multiprocessing.",
    )
    parser.add_argument("--single-robot-num", type=int, default=None)
    parser.add_argument("--single-obstacle-num", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_repo_root_as_cwd()

    if args.single_robot_num is not None and args.single_obstacle_num is not None:
        evaluate_single_setting(
            robot_num=args.single_robot_num,
            obs_num=args.single_obstacle_num,
            times=args.times,
            render=args.render,
            radius=args.radius,
            max_episode_steps=args.max_episode_steps,
        )
        return

    if args.parallel_workers and args.parallel_workers > 0:
        run_experiments_parallel(
            robot_nums=args.robot_nums,
            obstacle_nums=args.obstacle_nums,
            times=args.times,
            render=args.render,
            radius=args.radius,
            max_episode_steps=args.max_episode_steps,
            num_workers=args.parallel_workers,
        )
        return

    run_experiments(
        robot_nums=args.robot_nums,
        obstacle_nums=args.obstacle_nums,
        times=args.times,
        render=args.render,
        radius=args.radius,
        max_episode_steps=args.max_episode_steps,
    )


if __name__ == "__main__":
    main()
