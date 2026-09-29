import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train
import windows

RULES = ("separation", "alignment", "cohesion")


def fit_probe(features, targets, *, alpha):
    #features: (pages, boxes), targets: (pages, 2), the real push for each page
    #float64: drafts come in float32 (7 digits), and when the boxes mostly repeat
    #each other the fit needs more digits than that, or it scores below zero
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    #every box on the same scale, like squasher in train.py
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std==0] = 1.0
    x = (features - mean) /std
    #the average push is the bias, the multipliers explain the rest
    bias = targets.mean(axis=0)
    y = targets - bias
    #the best multipliers, with a penalty of alpha on big ones
    n_boxes = x.shape[1]
    weights = np.linalg.solve(x.T @ x + alpha * np.eye(n_boxes), x.T @ y)
    return {"mean": mean, "std": std, "weights": weights, "bias": bias}


def probe_predict(probe, features):
    features = np.asarray(features, dtype=np.float64)
    x = (features - probe["mean"]) / probe["std"]#new pages, scaled with the TRAINING mean and std
    return x @ probe["weights"] + probe["bias"]


def probe_r2(probe, features, targets):
    guess = probe_predict(probe, features)
    miss = ((guess - targets) **2).sum()  #the probe's squared miss, everything added up
    spread = ((targets - targets.mean(axis=0)) ** 2).sum() #guessing average wrongness
    return float(1 - miss / spread)


def read_drafts(checkpoint, position_windows, *, physics):
    #every bird's draft in every window, from one snapshot, as pages for the probe
    if checkpoint["config"]["kind"] != "rel_per_bird":
        raise ValueError("model must be 'rel_per_bird'")
    model = checkpoint["model"]
    n_drafts = len(model.blocks) + 1
    pages = []
    for k in range(n_drafts):
        pages.append([]) #one pile per draft

    for start in range(0, len(position_windows), 256): #in batches
        chunk = position_windows[start:start + 256]
        x = torch.from_numpy(train.apply_squash(checkpoint["x_scaler"], train.to_moves(chunk))).float()
        pair = torch.from_numpy(train.pair_geometry(chunk, bounds=physics["bounds"], scale=physics["separation_radius"]))

        with torch.no_grad():
            drafts = model.hidden_states(x, pair) 
        for k in range(n_drafts):
            n_windows, n_birds, n_boxes = drafts[k].shape
            pages[k].append(drafts[k].numpy().reshape(n_windows * n_birds, n_boxes))              #to numpy, then reshape to (n_windows * n_birds, n_boxes)

    for k in range(n_drafts):
        pages[k] = np.concatenate(pages[k])        #glue the batches back into one pile
    return pages


def pick_clips(n_clips, *, learn_clips, check_clips):
    #a sample of the training clips to fit the probe on, and a sample of the
    #held-out clips to score it on, the same 70 / 30 split as train py
    learn_all, check_all = train.split_by_clip(n_clips, train_fraction=0.7)
    if learn_clips > len(learn_all):
        raise ValueError(f"asked for {learn_clips} learn clips, there are {len(learn_all)}")
    if check_clips > len(check_all):
        raise ValueError(f"asked for {check_clips} check clips, there are {len(check_all)}")
    return learn_all[:learn_clips], check_all[:check_clips]


def readability(checkpoint, learn_windows, learn_labels, check_windows, check_labels, *, physics, alpha):
    #for one snapshot: how well each rule's push can be read off each draft
    #fit on the learn pages, score on the check pages
    learn_pages = read_drafts(checkpoint, learn_windows, physics=physics)
    check_pages = read_drafts(checkpoint, check_windows, physics=physics)
    scores = {}
    for rule in RULES:
        #(windows, birds, 2) -> one row per bird, the same order as the pages
        learn_push = learn_labels[rule].reshape(-1, 2)
        check_push = check_labels[rule].reshape(-1, 2)
        scores[rule] = []
        for k in range(len(learn_pages)):
            fitted = fit_probe(learn_pages[k], learn_push, alpha=alpha)
            scores[rule].append(probe_r2(fitted, check_pages[k], check_push))
    return scores


def batch_of(path):
    #step_00000123.pt -> 123
    name = Path(path).stem
    number = name.split("_")[1]
    return int(number)


def step_files(checkpoint_dir):
    #the in pass snapshots only, not the pass_ files, in training order
    #(the zero padding in the names makes sorted() put them in order)
    return sorted(Path(checkpoint_dir).glob("step_*.pt"))


def readability_curve(checkpoint_dir, learn_windows, learn_labels, check_windows, check_labels, *, physics, alpha):
    #the same comparison as readability(), once for every snapshot
    rows = []
    for path in step_files(checkpoint_dir):
        checkpoint = train.load_checkpoint(path)
        scores = readability(checkpoint, learn_windows, learn_labels, check_windows, check_labels,
                             physics=physics, alpha=alpha)
        rows.append({"batch": batch_of(path),
                     "pass": float(checkpoint["pass_number"]),
                     "scores": scores})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--clips", type=int, required=True)
    parser.add_argument("--learn-clips", type=int, default=50)
    parser.add_argument("--check-clips", type=int, default=20)
    #the untrained snapshot is the control: what random weights already show
    parser.add_argument("--untrained-batch", type=int, default=None)
    parser.add_argument("--trained-batch", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    #every snapshot instead of two, for the curves
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.sweep and args.out is None:
        parser.error("--sweep takes minutes, give it --out so the result is kept")

    with open(Path(args.data) / "config.json") as config_file:
        physics = json.load(config_file)["physics"]

    paths = step_files(args.checkpoint_dir)
    if len(paths) == 0:
        raise ValueError(f"no step_*.pt in {args.checkpoint_dir}, train with --log-checkpoints")

    #by default the first snapshot is the untrained one and the last is the finished model
    if args.untrained_batch is None:
        untrained_batch = batch_of(paths[0])
    else:
        untrained_batch = args.untrained_batch
    if args.trained_batch is None:
        trained_batch = batch_of(paths[-1])
    else:
        trained_batch = args.trained_batch

    #the windows are built at the context the snapshots were trained on
    last_checkpoint = train.load_checkpoint(paths[-1])
    context = last_checkpoint["config"]["context"]

    learn, check = pick_clips(args.clips, learn_clips=args.learn_clips, check_clips=args.check_clips)
    learn_windows, _ = windows.build_windows(args.data, context=context, clips=learn)
    check_windows, _ = windows.build_windows(args.data, context=context, clips=check)
    learn_labels = windows.build_force_labels(args.data, context=context, clips=learn)
    check_labels = windows.build_force_labels(args.data, context=context, clips=check)

    if args.sweep:
        curve = readability_curve(args.checkpoint_dir, learn_windows, learn_labels, check_windows, check_labels,
                                  physics=physics, alpha=args.alpha)
        saved = {"curve": curve,
                 "learn_clips": args.learn_clips,
                 "check_clips": args.check_clips,
                 "alpha": args.alpha}
        Path(args.out).write_text(json.dumps(saved, indent=1))
        print(f"{len(curve)} snapshots scored, saved to {args.out}")
        return saved

    result = {"untrained_batch": untrained_batch,
              "trained_batch": trained_batch,
              "learn_clips": args.learn_clips,
              "check_clips": args.check_clips,
              "alpha": args.alpha}

    for name, batch in [("untrained", untrained_batch), ("trained", trained_batch)]:
        checkpoint = train.load_checkpoint(Path(args.checkpoint_dir) / f"step_{batch:08d}.pt")
        if checkpoint["config"]["context"] != context:
            raise ValueError(f"batch {batch} has context {checkpoint['config']['context']}, the windows have {context}")
        result[name] = readability(checkpoint, learn_windows, learn_labels, check_windows, check_labels,
                                   physics=physics, alpha=args.alpha)

    #the part training added, on top of what random weights already show
    result["gain"] = {}
    for rule in RULES:
        result["gain"][rule] = []
        for k in range(len(result["trained"][rule])):
            gain = result["trained"][rule][k] - result["untrained"][rule][k]
            result["gain"][rule].append(gain)

    print(f"held-out R2 of the probe, batch {untrained_batch} (untrained) vs batch {trained_batch} (trained)")
    for rule in RULES:
        for k in range(len(result["gain"][rule])):
            untrained = result["untrained"][rule][k]
            trained = result["trained"][rule][k]
            gain = result["gain"][rule][k]
            print(f"  {rule:10s} draft {k}   untrained {untrained:+.3f}   trained {trained:+.3f}   gain {gain:+.3f}")

    if args.out is not None:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
