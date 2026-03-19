import os
import numpy as np
import math
import random
import config
from tqdm import tqdm


class ActiveInteractionScenarioEngine:
    def __init__(self, num_frames=50):
        self.num_frames = num_frames
        self.area_size = (1000, 1000)
        self.clutter_rate = 8
        self.detection_prob = 0.95

        self.min_speed = 10.0
        self.max_speed = 25.0
        self.next_member_id = 1

    def generate_episode(self):
        self.next_member_id = 1
        prob = random.random()
        if prob < 0.4:
            return self._run_converge_scenario()
        elif prob < 0.7:
            return self._run_diverge_scenario()
        else:
            return self._run_mixed_scenario()

    def _new_member_ids(self, num_members):
        member_ids = np.arange(self.next_member_id, self.next_member_id + num_members, dtype=np.int64)
        self.next_member_id += num_members
        return member_ids

    def _append_members(self, group, new_offsets):
        new_offsets = np.asarray(new_offsets, dtype=float).reshape(-1, 2)
        if len(new_offsets) == 0:
            return
        existing_ids = np.asarray(group.get('member_ids', np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        group['member_ids'] = np.concatenate([existing_ids, self._new_member_ids(len(new_offsets))])
        group['slot_offsets'] = self._generate_slot_offsets(len(group['member_ids']))
        group['offsets'] = self._rotate_local_offsets(group['slot_offsets'], self._group_heading(group)) + np.random.randn(len(group['member_ids']), 2) * 1.0
        group['member_vels'] = np.zeros((len(group['member_ids']), 2), dtype=float)

    def _new_frame_info(self):
        return {'meas': [], 'labels': [], 'gt_centers': [], 'gt_points': [], 'point_ids': []}

    def _normalize(self, vec, fallback=None):
        vec = np.asarray(vec, dtype=float).reshape(-1)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            return vec / norm
        if fallback is None:
            fallback = np.array([1.0, 0.0], dtype=float)
        fallback = np.asarray(fallback, dtype=float).reshape(-1)
        fallback_norm = np.linalg.norm(fallback)
        if fallback_norm > 1e-6:
            return fallback / fallback_norm
        return np.array([1.0, 0.0], dtype=float)

    def _group_heading(self, group):
        if 'v' in group:
            vel = np.asarray(group['v'], dtype=float)
            if np.linalg.norm(vel) > 1e-6:
                return self._normalize(vel)
        if 'target' in group and 'c' in group:
            return self._normalize(np.asarray(group['target'], dtype=float) - np.asarray(group['c'], dtype=float))
        return np.array([1.0, 0.0], dtype=float)

    def _rotate_local_offsets(self, local_offsets, heading):
        local_offsets = np.asarray(local_offsets, dtype=float).reshape(-1, 2)
        if len(local_offsets) == 0:
            return np.zeros((0, 2), dtype=float)
        forward = self._normalize(heading)
        lateral = np.array([-forward[1], forward[0]], dtype=float)
        return np.outer(local_offsets[:, 0], forward) + np.outer(local_offsets[:, 1], lateral)

    def _generate_slot_offsets(self, num_members):
        if num_members <= 0:
            return np.zeros((0, 2), dtype=float)

        longitudinal_step = 14.0
        lateral_step = 9.0
        slots = []
        row = 0
        while len(slots) < num_members:
            x = -row * longitudinal_step
            if row == 0:
                lateral_indices = [0.0]
            else:
                lateral_indices = []
                if row % 2 == 0:
                    lateral_indices.append(0.0)
                for k in range(1, row + 1):
                    lateral_indices.extend([-float(k), float(k)])

            for lat_idx in lateral_indices:
                slots.append([x, lat_idx * lateral_step])
                if len(slots) >= num_members:
                    break
            row += 1

        slots = np.asarray(slots[:num_members], dtype=float)
        slots -= np.mean(slots, axis=0, keepdims=True)
        return slots

    def _reassign_slots(self, previous_slots, member_indices):
        member_indices = np.asarray(member_indices, dtype=int).reshape(-1)
        if len(member_indices) == 0:
            return np.zeros((0, 2), dtype=float)

        previous_slots = np.asarray(previous_slots, dtype=float).reshape(-1, 2)
        selected_slots = previous_slots[member_indices]
        new_slots = self._generate_slot_offsets(len(member_indices))
        if len(member_indices) == 1:
            return new_slots

        old_order = np.lexsort((selected_slots[:, 1], -selected_slots[:, 0]))
        new_order = np.lexsort((new_slots[:, 1], -new_slots[:, 0]))
        reassigned = np.zeros_like(new_slots)
        reassigned[old_order] = new_slots[new_order]
        return reassigned

    def _create_group_state(
        self,
        pos,
        vel,
        target,
        num_members=None,
        member_ids=None,
        role='merger',
        offsets=None,
        member_vels=None,
        slot_offsets=None,
    ):
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        target = np.asarray(target, dtype=float)

        if member_ids is None:
            if num_members is None:
                raise ValueError('num_members or member_ids must be provided')
            member_ids = self._new_member_ids(int(num_members))
        member_ids = np.asarray(member_ids, dtype=np.int64).reshape(-1)
        count = len(member_ids)

        if slot_offsets is None:
            slot_offsets = self._generate_slot_offsets(count)
        else:
            slot_offsets = np.asarray(slot_offsets, dtype=float).reshape(-1, 2)

        if offsets is None:
            offsets = self._rotate_local_offsets(slot_offsets, self._group_heading({'v': vel, 'c': pos, 'target': target}))
            offsets = offsets + np.random.randn(count, 2) * 1.0
        else:
            offsets = np.asarray(offsets, dtype=float).reshape(-1, 2)

        if member_vels is None:
            member_vels = np.zeros((count, 2), dtype=float)
        else:
            member_vels = np.asarray(member_vels, dtype=float).reshape(-1, 2)

        return {
            'c': pos,
            'v': vel,
            'offsets': offsets,
            'slot_offsets': slot_offsets,
            'member_vels': member_vels,
            'member_ids': member_ids,
            'active': True,
            'target': target,
            'role': role,
        }

    def _merge_slot_offsets(self, slot_offsets_a, slot_offsets_b):
        slot_offsets_a = np.asarray(slot_offsets_a, dtype=float).reshape(-1, 2)
        slot_offsets_b = np.asarray(slot_offsets_b, dtype=float).reshape(-1, 2)
        if len(slot_offsets_a) == 0:
            return slot_offsets_b.copy()
        if len(slot_offsets_b) == 0:
            return slot_offsets_a.copy()

        width_a = np.ptp(slot_offsets_a[:, 1]) if len(slot_offsets_a) > 1 else 0.0
        width_b = np.ptp(slot_offsets_b[:, 1]) if len(slot_offsets_b) > 1 else 0.0
        wing_gap = max(18.0, 0.5 * (width_a + width_b) + 10.0)

        merged_a = slot_offsets_a.copy()
        merged_b = slot_offsets_b.copy()
        merged_a[:, 1] -= wing_gap * 0.5
        merged_b[:, 1] += wing_gap * 0.5

        merged = np.vstack([merged_a, merged_b])
        merged -= np.mean(merged, axis=0, keepdims=True)
        return merged

    def _merge_groups(self, keep_group, merge_group):
        n_keep = len(keep_group['member_ids'])
        n_merge = len(merge_group['member_ids'])
        total = max(1, n_keep + n_merge)

        keep_group['c'] = (keep_group['c'] * n_keep + merge_group['c'] * n_merge) / total
        keep_group['v'] = (keep_group['v'] * n_keep + merge_group['v'] * n_merge) / total
        keep_group['target'] = (keep_group['target'] * n_keep + merge_group['target'] * n_merge) / total
        keep_group['offsets'] = np.vstack([keep_group['offsets'], merge_group['offsets']])
        keep_group['member_vels'] = np.vstack([keep_group['member_vels'], merge_group['member_vels']])
        keep_group['member_ids'] = np.concatenate([keep_group['member_ids'], merge_group['member_ids']])
        keep_group['slot_offsets'] = self._merge_slot_offsets(keep_group['slot_offsets'], merge_group['slot_offsets'])
        keep_group['role'] = 'merged'

    def _split_group(self, parent, orth_vec):
        num_members = len(parent['member_ids'])
        half = num_members // 2
        if half < 3:
            return None

        slot_offsets = np.asarray(parent['slot_offsets'], dtype=float).reshape(-1, 2)
        heading = self._group_heading(parent)
        lateral_world = np.array([-heading[1], heading[0]], dtype=float)
        side_sign = 1.0 if np.dot(orth_vec, lateral_world) >= 0 else -1.0
        order = np.argsort(side_sign * slot_offsets[:, 1])
        child_idx = np.sort(order[-half:])
        parent_idx = np.sort(order[:-half])

        child_slot_offsets = self._reassign_slots(slot_offsets, child_idx)
        parent_slot_offsets = self._reassign_slots(slot_offsets, parent_idx)

        child = self._create_group_state(
            pos=parent['c'].copy() + orth_vec * 18.0,
            vel=parent['v'].copy() + orth_vec * 4.0,
            target=parent['target'].copy() + orth_vec * 400.0,
            member_ids=parent['member_ids'][child_idx].copy(),
            offsets=parent['offsets'][child_idx].copy(),
            member_vels=parent['member_vels'][child_idx].copy(),
            slot_offsets=child_slot_offsets,
            role='split_child',
        )

        parent['c'] = parent['c'] - orth_vec * 8.0
        parent['v'] = parent['v'] - orth_vec * 2.0
        parent['member_ids'] = parent['member_ids'][parent_idx].copy()
        parent['offsets'] = parent['offsets'][parent_idx].copy()
        parent['member_vels'] = parent['member_vels'][parent_idx].copy()
        parent['slot_offsets'] = parent_slot_offsets
        return child

    def _spawn_group_aiming_at(self, group_id, target_pos, start_area=None):
        if start_area:
            x = random.uniform(start_area[0], start_area[1])
            y = random.uniform(start_area[2], start_area[3])
            pos = np.array([x, y])
        else:
            side = random.randint(0, 3)
            if side == 0:
                pos = np.array([-50.0, random.uniform(0, 1000)])
            elif side == 1:
                pos = np.array([1050.0, random.uniform(0, 1000)])
            elif side == 2:
                pos = np.array([random.uniform(0, 1000), -50.0])
            else:
                pos = np.array([random.uniform(0, 1000), 1050.0])

        dist_to_target = np.linalg.norm(target_pos - pos)
        avg_speed = (self.min_speed + self.max_speed) / 2
        max_flyable_dist = avg_speed * (self.num_frames - 10)
        if dist_to_target > max_flyable_dist:
            ratio = max_flyable_dist / (dist_to_target + 1e-5)
            pos = target_pos + (pos - target_pos) * ratio
            pos += np.random.randn(2) * 20.0

        dir_vec = self._normalize(target_pos - pos)
        angle_noise = random.uniform(-0.15, 0.15)
        c, s = math.cos(angle_noise), math.sin(angle_noise)
        dir_vec = np.array([dir_vec[0] * c - dir_vec[1] * s, dir_vec[0] * s + dir_vec[1] * c])

        speed = random.uniform(self.min_speed + 5, self.max_speed)
        vel = dir_vec * speed

        num_members = random.randint(5, 15)
        return self._create_group_state(pos, vel, target_pos, num_members=num_members, role='merger')

    def _run_converge_scenario(self):
        groups = {}
        num_rps = random.choice([1, 1, 2])
        rps = [np.array([random.uniform(300, 700), random.uniform(300, 700)]) for _ in range(num_rps)]

        num_groups = random.randint(2, 5)
        for i in range(num_groups):
            assigned_rp = random.choice(rps)
            groups[i + 1] = self._spawn_group_aiming_at(i + 1, assigned_rp)

        episode_data = []
        for _ in range(self.num_frames):
            frame_info = self._new_frame_info()

            active_ids = list(groups.keys())
            for i in range(len(active_ids)):
                id1 = active_ids[i]
                if id1 not in groups:
                    continue

                for j in range(i + 1, len(active_ids)):
                    id2 = active_ids[j]
                    if id2 not in groups:
                        continue

                    dist = np.linalg.norm(groups[id1]['c'] - groups[id2]['c'])
                    if dist < 30.0:
                        self._merge_groups(groups[id1], groups[id2])
                        del groups[id2]

            for g in groups.values():
                self._apply_guidance(g, g['target'])
                self._apply_wander(g)

            self._update_members_and_record(groups, frame_info)
            episode_data.append(frame_info)

        return episode_data

    def _run_diverge_scenario(self):
        groups = {}
        num_groups = random.randint(1, 3)

        for i in range(num_groups):
            if random.random() < 0.5:
                y_start = random.uniform(100, 900)
                start_pos = np.array([-50.0, y_start])
                target_pos = np.array([1050.0, random.uniform(100, 900)])
            else:
                x_start = random.uniform(100, 900)
                start_pos = np.array([x_start, -50.0])
                target_pos = np.array([random.uniform(100, 900), 1050.0])

            g = self._spawn_group_aiming_at(i + 1, target_pos)
            g['c'] = start_pos

            dist_total = np.linalg.norm(target_pos - start_pos)
            if dist_total > 1200:
                g['v'] *= 1.5

            if len(g['offsets']) < 12:
                extra = np.random.randn(8, 2) * 8.0
                self._append_members(g, extra)

            g['split_time'] = random.randint(int(self.num_frames * 0.3), int(self.num_frames * 0.6))
            groups[i + 1] = g

        episode_data = []
        next_id = max(groups.keys()) + 1

        for t in range(self.num_frames):
            frame_info = self._new_frame_info()

            current_ids = list(groups.keys())
            for gid in current_ids:
                if gid not in groups:
                    continue
                g = groups[gid]
                if 'split_time' in g and t == g['split_time']:
                    vel_norm = self._group_heading(g)
                    orth_vec = np.array([-vel_norm[1], vel_norm[0]])
                    if random.random() > 0.5:
                        orth_vec *= -1

                    child = self._split_group(g, orth_vec)
                    del g['split_time']
                    if child is not None:
                        groups[next_id] = child
                        next_id += 1

            for g in groups.values():
                self._apply_guidance(g, g['target'])
                self._apply_wander(g)

            self._update_members_and_record(groups, frame_info)
            episode_data.append(frame_info)

        return episode_data

    def _run_mixed_scenario(self):
        groups = {}

        start_pos = np.array([100.0, 200.0])
        target_pos = np.array([900.0, 800.0])
        g1 = self._spawn_group_aiming_at(1, target_pos)
        g1['c'] = start_pos
        g1['split_time'] = 20
        g1['v'] = self._normalize(g1['v']) * 20.0
        extra = np.random.randn(12, 2) * 8.0
        self._append_members(g1, extra)
        groups[1] = g1

        center = np.array([500.0, 500.0])

        g2 = self._spawn_group_aiming_at(2, center)
        g2['c'] = np.array([200.0, 800.0])
        g2['v'] = self._normalize(center - g2['c']) * 22.0
        groups[2] = g2

        g3 = self._spawn_group_aiming_at(3, center)
        g3['c'] = np.array([800.0, 200.0])
        g3['v'] = self._normalize(center - g3['c']) * 22.0
        groups[3] = g3

        next_id = 4
        episode_data = []

        for t in range(self.num_frames):
            frame_info = self._new_frame_info()

            if 1 in groups and t == groups[1].get('split_time', -1):
                parent = groups[1]
                vel_norm = self._group_heading(parent)
                orth_vec = np.array([-vel_norm[1], vel_norm[0]])
                child = self._split_group(parent, orth_vec)
                del parent['split_time']
                if child is not None:
                    groups[next_id] = child
                    next_id += 1

            if 2 in groups and 3 in groups:
                dist = np.linalg.norm(groups[2]['c'] - groups[3]['c'])
                if dist < 40.0:
                    self._merge_groups(groups[2], groups[3])
                    del groups[3]

            for g in groups.values():
                self._apply_guidance(g, g['target'])
                self._apply_wander(g)

            self._update_members_and_record(groups, frame_info)
            episode_data.append(frame_info)

        return episode_data

    def _apply_guidance(self, group, target_pos):
        desired_vec = target_pos - group['c']
        dist = np.linalg.norm(desired_vec)
        if dist > 1.0:
            desired_vec /= dist

        curr_speed = np.linalg.norm(group['v'])
        target_speed = curr_speed
        if dist > 300:
            target_speed = max(curr_speed, self.max_speed * 1.2)
        elif dist < 100:
            target_speed = min(curr_speed, self.max_speed * 0.8)

        desired_vel = desired_vec * target_speed
        inertia = 0.85
        group['v'] = group['v'] * inertia + desired_vel * (1 - inertia)

        actual_speed = np.linalg.norm(group['v'])
        if actual_speed > self.max_speed * 1.5:
            group['v'] = group['v'] / actual_speed * (self.max_speed * 1.5)

    def _apply_wander(self, group):
        speed = np.linalg.norm(group['v'])
        speed += random.uniform(-1.0, 1.0)
        speed = np.clip(speed, self.min_speed, self.max_speed * 1.5)

        angle = random.uniform(-0.03, 0.03)
        c, s = math.cos(angle), math.sin(angle)
        vx, vy = group['v']
        group['v'] = np.array([vx * c - vy * s, vx * s + vy * c]) / (np.linalg.norm([vx, vy]) + 1e-5) * speed

        group['c'] += group['v']

    def _update_members_and_record(self, groups, frame_info):
        n_clutter = np.random.poisson(self.clutter_rate)
        for _ in range(n_clutter):
            frame_info['meas'].append([random.uniform(0, 1000), random.uniform(0, 1000)])
            frame_info['labels'].append(0)
            frame_info['point_ids'].append(0)

        for gid, g in groups.items():
            if 'member_ids' not in g:
                g['member_ids'] = self._new_member_ids(len(g['offsets']))
            if 'slot_offsets' not in g:
                g['slot_offsets'] = self._generate_slot_offsets(len(g['member_ids']))
            if 'member_vels' not in g or len(g['member_vels']) != len(g['member_ids']):
                g['member_vels'] = np.zeros((len(g['member_ids']), 2), dtype=float)
            if 'offsets' not in g or len(g['offsets']) != len(g['member_ids']):
                g['offsets'] = self._rotate_local_offsets(g['slot_offsets'], self._group_heading(g))

            target_offsets = self._rotate_local_offsets(g['slot_offsets'], self._group_heading(g))
            formation_error = target_offsets - g['offsets']
            g['member_vels'] = 0.55 * g['member_vels'] + 0.25 * formation_error + np.random.randn(*g['offsets'].shape) * 0.15
            g['offsets'] += g['member_vels']

            frame_info['gt_centers'].append([gid, g['c'][0], g['c'][1]])

            true_pts = g['c'] + g['offsets']
            for member_id, pt in zip(g['member_ids'], true_pts):
                if pt[0] < 0 or pt[0] > 1000 or pt[1] < 0 or pt[1] > 1000:
                    continue

                frame_info['gt_points'].append([member_id, pt[0], pt[1]])

                if random.random() < self.detection_prob:
                    noise = np.random.randn(2) * 1.5
                    frame_info['meas'].append(pt + noise)
                    frame_info['labels'].append(gid)
                    frame_info['point_ids'].append(member_id)

        if len(frame_info['meas']) > 0:
            frame_info['meas'] = np.array(frame_info['meas'], dtype=np.float32)
            frame_info['labels'] = np.array(frame_info['labels'], dtype=np.int64)
            frame_info['point_ids'] = np.array(frame_info['point_ids'], dtype=np.int64)
        else:
            frame_info['meas'] = np.zeros((0, 2), dtype=np.float32)
            frame_info['labels'] = np.zeros((0,), dtype=np.int64)
            frame_info['point_ids'] = np.zeros((0,), dtype=np.int64)

        if len(frame_info['gt_centers']) > 0:
            frame_info['gt_centers'] = np.array(frame_info['gt_centers'], dtype=np.float32)
        else:
            frame_info['gt_centers'] = np.zeros((0, 3), dtype=np.float32)

        if len(frame_info['gt_points']) > 0:
            frame_info['gt_points'] = np.array(frame_info['gt_points'], dtype=np.float32)
        else:
            frame_info['gt_points'] = np.zeros((0, 3), dtype=np.float32)


def save_dataset(split_name, num_samples):
    folder = os.path.join(config.DATA_ROOT, split_name)
    if os.path.exists(folder):
        import shutil
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

    sim = ActiveInteractionScenarioEngine(num_frames=config.FRAMES_PER_SAMPLE)
    print(f"Generating {split_name} data ({num_samples} HIGH SPEED INTERACTION scenarios)...")
    for i in tqdm(range(num_samples)):
        episode = sim.generate_episode()
        save_path = os.path.join(folder, f"sample_{i:05d}.npy")
        np.save(save_path, episode, allow_pickle=True)


if __name__ == "__main__":
    save_dataset("train", config.NUM_TRAIN_SAMPLES)
    save_dataset("val", config.NUM_VAL_SAMPLES)
    save_dataset("test", config.NUM_TEST_SAMPLES)
