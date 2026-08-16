from __future__ import annotations

import math

import numpy as np


class PaperReward:
    def __init__(self, config):
        self.cfg = config

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _hazard_terms(self, ego_state, human_state):
        rel_pos = np.asarray([human_state["x"] - ego_state["x"], human_state["y"] - ego_state["y"]], dtype=np.float32)
        distance = float(np.linalg.norm(rel_pos))

        vh = np.asarray([human_state["vx"], human_state["vy"]], dtype=np.float32)
        vr = np.asarray([ego_state["vx"], ego_state["vy"]], dtype=np.float32)
        vhr = vh - vr
        speed = float(np.linalg.norm(vhr))

        base_radius = self.cfg.human_radius + self.cfg.discomfort_dist
        risk_radius = self.cfg.velocity_weight * speed + base_radius
        sector_angle = float((11.0 * np.pi / 6.0) * np.exp(-1.5 * speed) + np.pi / 6.0)
        zeta = float(np.arctan2(vhr[1], vhr[0])) if speed > 1e-6 else 0.0
        beta = float(np.arctan2(rel_pos[1], rel_pos[0])) if distance > 1e-6 else 0.0

        in_ra = False
        if distance < max(risk_radius - self.cfg.human_radius, 0.0):
            angle_error = abs(self.wrap_to_pi(beta - zeta))
            in_ra = angle_error < (sector_angle / 2.0)

        in_da = distance < self.cfg.discomfort_dist and not in_ra
        return distance, risk_radius, in_ra, in_da

    def compute(self, meta):
        ego_id = meta["ego_id"]
        ego_state = meta["states"][ego_id]

        hazard_penalty = 0.0
        discomfort_penalty = 0.0
        hazard_detected = False
        discomfort_detected = False
        min_human_dist = math.inf

        for rid, human_state in enumerate(meta["states"]):
            if rid == ego_id:
                continue
            distance, risk_radius, in_ra, in_da = self._hazard_terms(ego_state, human_state)
            min_human_dist = min(min_human_dist, distance)
            if in_ra:
                hazard_detected = True
                denom = max(risk_radius - self.cfg.human_radius, 1e-6)
                hazard_penalty = min(hazard_penalty, -0.1 * (1.0 - distance / denom))
            elif in_da:
                discomfort_detected = True
                discomfort_penalty = min(discomfort_penalty, -0.1 * float(np.exp(-12.0 * distance)))

        social_penalty = hazard_penalty if hazard_detected else discomfort_penalty if discomfort_detected else 0.0
        info = {
            "reward_hazard": float(hazard_penalty),
            "reward_discomfort": float(discomfort_penalty),
            "reward_social": float(social_penalty),
            "hazard_detected": hazard_detected,
            "discomfort_detected": discomfort_detected,
            "min_human_dist": float(min_human_dist) if np.isfinite(min_human_dist) else float("inf"),
        }
        return float(social_penalty), info

    def compute_all(self, meta_list):
        rewards = []
        infos = []
        for meta in meta_list:
            reward, info = self.compute(meta)
            rewards.append(reward)
            infos.append(info)
        return np.asarray(rewards, dtype=np.float32), infos
