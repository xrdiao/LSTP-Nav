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
from tqdm.auto import tqdm

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


DEFAULT_ROBOT_NUMS = (5, 10)
DEFAULT_OBSTACLE_NUMS = (5, 10, 15, 20, 25, 30, 35)
DEFAULT_EVAL_TIMES = 100
DEFAULT_ENV_NAME = "circle"
DEFAULT_ENV_RADIUS = 17.0
DEFAULT_MAX_EPISODE_STEPS = 6000
DEFAULT_PARALLEL_WORKERS = 0


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
class DynamicObstacleState:
    position: np.ndarray
    velocity: np.ndarray
    radius: float


@dataclass
class StaticObstacleState:
    position: np.ndarray
    radius: float


def ensure_casadi_available():
    if ca is None:  # pragma: no cover
        raise ImportError(
            "casadi is required to run SICNav. Install it in the active environment first."
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


def critically_damped_pd(
    position: np.ndarray,
    velocity: np.ndarray,
    target: np.ndarray,
    kp: float = 1.0,
    a_max: float | None = None,
) -> np.ndarray:
    acc = -kp * (position - target) - 2.0 * np.sqrt(kp) * velocity
    if a_max is not None and a_max > 0:
        norm = float(np.linalg.norm(acc))
        if norm > a_max and norm > 0:
            acc = acc / norm * a_max
    return acc


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


class EnvStateAdapter:
    def __init__(self, env, robot_radius: float = ROBOT_WIDTH, max_speed: float = MAX_SPEED, max_acceleration: float = 1.0):
        self.env = env
        self.robot_radius = float(robot_radius)
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)

    def get_robot_states(self) -> list[RobotState]:
        states = []
        for idx, robot in enumerate(self.env.robots):
            planar_velocity = np.asarray(robot.cur_vel, dtype=float).reshape(-1)
            if planar_velocity.shape[0] >= 2:
                planar_velocity = planar_velocity[:2]
            elif planar_velocity.shape[0] == 1:
                planar_velocity = np.array(
                    [float(planar_velocity[0] * np.cos(robot.theta)), float(planar_velocity[0] * np.sin(robot.theta))],
                    dtype=float,
                )
            else:
                planar_velocity = np.zeros(2, dtype=float)
            states.append(
                RobotState(
                    idx=idx,
                    position=np.asarray(robot.cur_pos, dtype=float),
                    velocity=planar_velocity,
                    goal=np.asarray(robot.target_pos, dtype=float),
                    radius=self.robot_radius,
                    max_speed=self.max_speed,
                    max_acceleration=self.max_acceleration,
                )
            )
        return states

    def get_static_obstacles(self) -> list[StaticObstacleState]:
        obstacles = []
        for record in self.env.get_obstacle_records():
            obstacles.append(
                StaticObstacleState(
                    position=np.asarray([record["x"], record["y"]], dtype=float),
                    radius=obstacle_record_to_radius(record),
                )
            )
        return obstacles

    def snapshot(self) -> tuple[list[RobotState], list[StaticObstacleState]]:
        return self.get_robot_states(), self.get_static_obstacles()


class SICNavPlanner:
    def __init__(
        self,
        dt: float = 0.1,
        tau: float = 2.0,
        lambda_upper: float = 0.8,
        slack_penalty: float = 5000.0,
        safety_margin: float = 0.25,
        goal_gain: float = 1.0,
    ):
        ensure_casadi_available()
        self.dt = float(dt)
        self.tau = float(tau)
        self.lambda_upper = float(lambda_upper)
        self.slack_penalty = float(slack_penalty)
        self.safety_margin = float(safety_margin)
        self.goal_gain = float(goal_gain)
        self.solve_failures = 0

    def plan(
        self,
        robot: RobotState,
        dynamic_obstacles: list[DynamicObstacleState],
        static_obstacles: list[StaticObstacleState],
    ) -> tuple[np.ndarray, float]:
        target_acc = critically_damped_pd(
            position=robot.position,
            velocity=robot.velocity,
            target=robot.goal,
            kp=self.goal_gain,
            a_max=robot.max_acceleration,
        )

        v_curr = robot.velocity
        v_pref = v_curr + target_acc * self.dt
        speed = float(np.linalg.norm(v_pref))
        if speed > robot.max_speed and speed > 0:
            v_pref = v_pref / speed * robot.max_speed

        if not dynamic_obstacles and not static_obstacles:
            return v_pref, 0.0

        start_t = time.time()
        try:
            v_safe = self._solve_velocity(robot, v_pref, dynamic_obstacles, static_obstacles)
        except Exception:
            self.solve_failures += 1
            v_safe = v_curr + smooth_brake(v_curr, robot.max_acceleration) * self.dt
            safe_speed = float(np.linalg.norm(v_safe))
            if safe_speed > robot.max_speed and safe_speed > 0:
                v_safe = v_safe / safe_speed * robot.max_speed
        return v_safe, (time.time() - start_t) * 1000.0

    def _solve_velocity(
        self,
        robot: RobotState,
        v_pref: np.ndarray,
        dynamic_obstacles: list[DynamicObstacleState],
        static_obstacles: list[StaticObstacleState],
    ) -> np.ndarray:
        opti = ca.Opti()
        v_opt = opti.variable(2)
        total_cost = ca.sumsqr(v_opt - v_pref)
        v_curr = robot.velocity

        dynamic_constraints = []
        for obs in dynamic_obstacles:
            rel_position = np.asarray(obs.position - robot.position, dtype=float)
            rel_velocity = np.asarray(v_curr - obs.velocity, dtype=float)
            combined_radius = robot.radius + obs.radius + self.safety_margin
            n_vec, u_vec = self._velocity_obstacle_update(rel_position, rel_velocity, combined_radius)
            dynamic_constraints.append((obs, n_vec, u_vec, rel_velocity))

        static_constraints = []
        for obs in static_obstacles:
            rel_position = np.asarray(obs.position - robot.position, dtype=float)
            rel_velocity = np.asarray(v_curr, dtype=float)
            combined_radius = robot.radius + obs.radius + self.safety_margin
            n_vec, u_vec = self._velocity_obstacle_update(rel_position, rel_velocity, combined_radius)
            static_constraints.append((n_vec, float(np.dot(n_vec, v_curr + u_vec))))

        if dynamic_constraints:
            obstacle_vel = opti.variable(2, len(dynamic_constraints))
            lambdas = opti.variable(len(dynamic_constraints))
            slack = opti.variable(len(dynamic_constraints))

            opti.subject_to(lambdas >= 0)
            opti.subject_to(lambdas <= self.lambda_upper)
            opti.subject_to(slack >= 0)
            total_cost += self.slack_penalty * ca.sumsqr(slack)

            for i, (obs, n_vec, u_vec, rel_velocity) in enumerate(dynamic_constraints):
                rhs = float(np.dot(n_vec, rel_velocity + u_vec))
                v_obs_pref = obs.velocity
                opti.subject_to(obstacle_vel[0, i] == v_obs_pref[0] - lambdas[i] * n_vec[0])
                opti.subject_to(obstacle_vel[1, i] == v_obs_pref[1] - lambdas[i] * n_vec[1])

                gap = (
                    n_vec[0] * (v_opt[0] - obstacle_vel[0, i]) +
                    n_vec[1] * (v_opt[1] - obstacle_vel[1, i]) +
                    slack[i] - rhs
                )
                opti.subject_to(gap >= 0)
                opti.subject_to(lambdas[i] * gap <= 1e-2)

                opti.set_initial(obstacle_vel[0, i], v_obs_pref[0])
                opti.set_initial(obstacle_vel[1, i], v_obs_pref[1])

            opti.set_initial(lambdas, np.zeros(len(dynamic_constraints)))
            opti.set_initial(slack, np.zeros(len(dynamic_constraints)))

        if static_constraints:
            slack_static = opti.variable(len(static_constraints))
            opti.subject_to(slack_static >= 0)
            total_cost += self.slack_penalty * ca.sumsqr(slack_static)

            for i, (n_vec, rhs) in enumerate(static_constraints):
                opti.subject_to(n_vec[0] * v_opt[0] + n_vec[1] * v_opt[1] + slack_static[i] >= rhs)

            opti.set_initial(slack_static, np.zeros(len(static_constraints)))

        opti.minimize(total_cost)
        opti.subject_to(ca.sumsqr(v_opt) <= robot.max_speed ** 2 + 0.05)
        opti.set_initial(v_opt, v_curr)
        opti.solver(
            "ipopt",
            {
                "ipopt.max_iter": 300,
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.acceptable_tol": 1e-2,
                "ipopt.acceptable_obj_change_tol": 1e-2,
            },
        )

        solution = opti.solve()
        return np.asarray(solution.value(v_opt), dtype=float).reshape(2)

    def _velocity_obstacle_update(
        self,
        rel_position: np.ndarray,
        rel_velocity: np.ndarray,
        combined_radius: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        noise = np.random.uniform(-1e-4, 1e-4, size=2)
        x_vec = np.asarray(rel_position, dtype=float) + noise
        dist_sq = float(np.dot(x_vec, x_vec))
        if dist_sq < 1e-8:
            x_vec = np.array([1e-4, 0.0], dtype=float)
            dist_sq = float(np.dot(x_vec, x_vec))

        if dist_sq < combined_radius ** 2:
            dist = max(np.sqrt(dist_sq), 1e-6)
            n_vec = -x_vec / dist
            repulsion_speed = (combined_radius - dist) / self.dt
            u_vec = n_vec * repulsion_speed - rel_velocity
            return n_vec, u_vec

        w_vec = rel_velocity - x_vec / self.tau
        w_len_sq = float(np.dot(w_vec, w_vec))
        dot_w_x = float(np.dot(w_vec, x_vec))

        if w_len_sq < 1e-8:
            n_vec = x_vec / max(np.linalg.norm(x_vec), 1e-6)
            u_vec = n_vec * (combined_radius / self.tau)
            return n_vec, u_vec

        if dot_w_x < 0 and dot_w_x ** 2 > combined_radius ** 2 * w_len_sq:
            w_len = max(np.sqrt(w_len_sq), 1e-6)
            n_vec = w_vec / w_len
            u_vec = n_vec * (combined_radius / self.tau - w_len)
            return n_vec, u_vec

        leg_len = np.sqrt(max(0.0, dist_sq - combined_radius ** 2))
        det = x_vec[0] * rel_velocity[1] - x_vec[1] * rel_velocity[0]
        if det > 0:
            n_vec = np.array(
                [
                    -(x_vec[1] * leg_len + x_vec[0] * combined_radius),
                    (x_vec[0] * leg_len - x_vec[1] * combined_radius),
                ],
                dtype=float,
            ) / dist_sq
        else:
            n_vec = np.array(
                [
                    (x_vec[1] * leg_len - x_vec[0] * combined_radius),
                    -(x_vec[0] * leg_len + x_vec[1] * combined_radius),
                ],
                dtype=float,
            ) / dist_sq
        dist_to_leg = float(np.dot(w_vec, n_vec))
        u_vec = n_vec * (-dist_to_leg)
        return n_vec, u_vec


class ActionProjector:
    def __init__(self, env):
        self.env = env

    def project_velocity(self, robot_idx: int, desired_velocity: np.ndarray) -> list[float]:
        return self.env.robots[robot_idx].cal_effective_cmd(np.asarray(desired_velocity, dtype=float))


class SICNavController:
    def __init__(self, env):
        self.env = env
        self.adapter = EnvStateAdapter(env)
        self.planner = SICNavPlanner(dt=getattr(env, "delta_time", 0.1))
        self.projector = ActionProjector(env)

    def compute_actions(self) -> tuple[list[list[float]], list[float]]:
        robot_states, static_obstacles = self.adapter.snapshot()
        actions = []
        solve_times = []

        for robot_state in robot_states:
            dynamic_obstacles = [
                DynamicObstacleState(position=other.position, velocity=other.velocity, radius=other.radius)
                for other in robot_states
                if other.idx != robot_state.idx
            ]
            safe_velocity, solve_time_ms = self.planner.plan(robot_state, dynamic_obstacles, static_obstacles)
            actions.append(self.projector.project_velocity(robot_state.idx, safe_velocity))
            solve_times.append(solve_time_ms)

        return actions, solve_times


class SICNavEvaluator:
    def __init__(self, env, agent_name: str = "SICNav"):
        self.env = env
        self.agent_name = agent_name
        self.controller = SICNavController(env)
        self.robots_num = 0

    def evaluate(self, times: int = DEFAULT_EVAL_TIMES, show_progress: bool = True) -> dict:
        rewards = []
        next_obs, _ = self.env.reset()

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

            while True:
                actions, solve_times = self.controller.compute_actions()
                all_solve_times.extend(solve_times)

                for idx, robot in enumerate(self.env.robots):
                    if robot.reach_goal:
                        actions[idx] = [0.0, 0.0]

                next_obs, reward, te, tr, _ = self.env.step(actions)
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
                    rewards.append(ep_return_vec)

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
                    break

                next_obs = self.env.reset(tr=tr, te=te)[0] if all(item for item in te) else next_obs

        tot_test_times = times * self.robots_num
        solve_failures = self.controller.planner.solve_failures
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
            "solve_failures": int(solve_failures),
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
            print("policy:", "SICNav", "robot_nums:", env.robots_num, "obstacle_nums:", env.random_obstacles)
        evaluator = SICNavEvaluator(env)
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
            desc="SICNav batch",
        ):
            all_stats[key] = stats
            print(f"finished robot={key[0]} obs={key[1]} reach={stats['reach_rate']:.3f} collision={stats['collision_rate']:.3f}")
    return all_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Run SICNav evaluation on my_env.")
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
