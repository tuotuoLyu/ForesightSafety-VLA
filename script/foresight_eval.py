"""ForesightSafety-VLA rollout loop, called from script/eval_policy.py when ``safety_eval`` is true.

Protocol (paper Sec. V-A):
* ``warmup_episodes`` unscored canonical episodes (L0/W0/V0) on a disjoint seed block, no parameter updates;
* ``num_episodes`` scored episodes (default 50) distributed over ``seeds`` (default 0, 1, 2 -> 17/17/16);
* scene seeds are screened with the scripted expert exactly as in RoboTwin (``expert_check``); the expert run
  also provides the episode info used to fill the instruction templates;
* per episode: monitor attached to the environment, every physics step evaluated (Eq. (1)-(6)), task success
  read with the scenario's safety gating suspended, outcome quadrant and SASR recorded.
"""

import json
import subprocess
import traceback

import numpy as np

from envs.utils.create_actor import UnStableError
from foresight_safety.monitor import EpisodeWriter, SafetyMonitor
from foresight_safety.perturbations import VisualPerturber, language_variant
from foresight_safety.protocol import (EPISODES_PER_SETTING, SEEDS, WARMUP_EPISODES, WARMUP_SEED, episodes_per_seed,
                                       robotwin_start_seed)
from foresight_safety.adapters import scenario_flags
from foresight_safety.tasks import get_task


def _ffmpeg(path, video_size):
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", video_size,
         "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23", str(path)],
        stdin=subprocess.PIPE)


def _full_description(task_name):
    with open(f"description/task_instruction/{task_name}.json", "r", encoding="utf-8") as f:
        return json.load(f)["full_description"]


def _next_valid_seed(TASK_ENV, args, now_seed, now_id, expert_check, max_trials):
    """Return (seed, episode_info) of the next scene the scripted expert solves (RoboTwin protocol)."""
    trials = 0
    while True:
        if trials >= max_trials:
            raise RuntimeError(f"no valid scene in {max_trials} consecutive seeds from {now_seed - trials}")
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        try:
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            episode_info = TASK_ENV.play_once()
            ok = TASK_ENV.plan_success and TASK_ENV.check_success()
            TASK_ENV.close_env()
        except UnStableError:
            TASK_ENV.close_env()
            ok, episode_info = False, None
        except Exception:
            traceback.print_exc()
            TASK_ENV.close_env()
            ok, episode_info = False, None
        args["render_freq"] = render_freq
        if isinstance(episode_info, dict) and "info" in episode_info and (ok or not expert_check):
            return now_seed, episode_info
        now_seed += 1
        trials += 1


def run_foresight_evaluation(task_name, TASK_ENV, args, usr_args, model, save_dir, video_size, levels,
                             eval_function_decorator, generate_episode_descriptions):
    spec = get_task(task_name)
    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")
    args["eval_mode"] = True

    total = int(usr_args.get("num_episodes", EPISODES_PER_SETTING))
    seeds = usr_args.get("seeds", list(SEEDS))
    seeds = [int(s) for s in (seeds if isinstance(seeds, (list, tuple)) else [seeds])]
    warmup = int(usr_args.get("warmup_episodes", WARMUP_EPISODES))
    lam = float(usr_args.get("safety_lambda", 1.0))
    keep_trace = bool(usr_args.get("foresight_keep_trace", False))
    expert_check = bool(usr_args.get("expert_check", True))
    max_trials = int(usr_args.get("max_seed_trials", 100))
    weights = usr_args.get("safety_channel_weights") or None
    structure, language, visual = levels

    writer = EpisodeWriter(save_dir, {
        "task_name": task_name, "safe_core_suite": spec.suite, "policy_name": policy_name,
        "task_config": args["task_config"], "ckpt_setting": args["ckpt_setting"],
        "structure_level": structure, "language_level": language, "visual_level": visual,
        "embodiment": args.get("embodiment"), "lambda": lam, "seeds": seeds, "num_episodes": total,
        "channels": {c.name: {"category": c.category, "eps": c.eps, "doc": c.doc} for c in spec.channels},
    })
    print(f"\033[34mForesightSafety-VLA | {task_name} ({spec.suite}) | {policy_name} | "
          f"{structure}/{language}/{visual} | {total} episodes over seeds {seeds}\033[0m")

    plan = [("warmup", WARMUP_SEED, warmup)] + [("scored", s, n) for s, n in episodes_per_seed(total, seeds).items()]
    episode = 0
    for phase, seed_idx, count in plan:
        is_warmup = phase == "warmup"
        now_seed = robotwin_start_seed(seed_idx)
        lv = ("L0", "W0", "V0") if is_warmup else levels
        args["foresight_structure_level"] = lv[0]
        for _ in range(count):
            now_seed, info = _next_valid_seed(TASK_ENV, args, now_seed, episode, expert_check, max_trials)
            TASK_ENV.setup_demo(now_ep_num=episode, seed=now_seed, is_test=True, **args)
            TASK_ENV.test_num = episode

            descs = generate_episode_descriptions(task_name, [info["info"]], 100)
            desc = descs[0] if descs else {"seen": [], "unseen": []}
            rng = np.random.default_rng(now_seed)
            pool = desc["seen"] or desc["unseen"] or [_full_description(task_name)]
            canonical = str(rng.choice(pool))
            variant = language_variant(lv[1], canonical, desc["unseen"], spec.suite, now_seed)
            TASK_ENV.set_instruction(instruction=variant["instruction"])

            monitor = SafetyMonitor(spec, lam=lam, keep_trace=keep_trace, channel_weights=weights)
            visual_fn = VisualPerturber(lv[2], seed=now_seed) if lv[2] != "V0" else None
            TASK_ENV.attach_foresight(monitor=monitor, visual=visual_fn)

            if TASK_ENV.eval_video_path is not None and not is_warmup:
                TASK_ENV._set_eval_video_ffmpeg(_ffmpeg(TASK_ENV.eval_video_path / f"episode{episode}.mp4", video_size))

            reset_func(model)
            succ = False
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                observation = TASK_ENV.get_obs()
                eval_func(TASK_ENV, model, observation)
                if TASK_ENV.eval_success:
                    succ = True
                    break
            if TASK_ENV.eval_video_path is not None and not is_warmup:
                TASK_ENV._del_eval_video_ffmpeg()

            record = monitor.finish(succ, episode=episode, seed=seed_idx, scene_seed=now_seed, warmup=is_warmup,
                                    policy_name=policy_name, structure_level=lv[0], language_level=lv[1],
                                    visual_level=lv[2], language_category=variant["category"],
                                    instruction=variant["instruction"], embodiment=args.get("embodiment"),
                                    scenario_flags=scenario_flags(TASK_ENV))
            TASK_ENV.attach_foresight(None, None)
            TASK_ENV.close_env(clear_cache=((episode + 1) % args["clear_cache_freq"] == 0))
            if TASK_ENV.render_freq:
                TASK_ENV.viewer.close()

            if not is_warmup:
                writer.add(record)
                o = writer.write_summary()["overall"]
                print(f"[{episode}] {record['outcome']} CC={record['CC']:.3f} RET/T={record['RET']:.3f} "
                      f"SASR={record['SASR']:.3f} | running SSR={o['SSR']:.2f} USR={o['USR']:.2f} "
                      f"SFR={o['SFR']:.2f} UFR={o['UFR']:.2f} CC={o['CC']:.3f} SASR={o['SASR']:.3f}")
            else:
                print(f"[warm-up] {record['outcome']} (not scored)")
            episode += 1
            now_seed += 1
    return writer.write_summary()
