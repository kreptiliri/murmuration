import argparse
import json
from pathlib import Path

import numpy as np
import torch

import probe
import train
import windows

RULES = ("separation", "alignment", "cohesion")
NOTES = ("average", "zero", "self")


def self_listening(attention, x, pair):
    #what the listening step writes when every bird hears only itself.
    #all the attention on yourself means the weights are 1 on you and 0 on everyone else,
    #so what comes back is your own v plus the "where" for your own place (distance 0)
    batch, n_birds, d_model = x.shape
    own_place = []
    for i in range(n_birds):
        own_place.append(pair[:, i, i])
    own_place = torch.stack(own_place, dim=1) #(batch, birds, 3)
    out = attention.v(x) + attention.pair_value(own_place)
    return attention.proj(out)


def ablated_forward(model, x, pair, *, block, note):
    #the model's forward, but block number "block" hears "note" instead of its neighbours
    #note is a fixed note (128 numbers) or "self": every bird hears only itself
    if not (0 <= block < len(model.blocks)):
        raise ValueError(f"no block {block}, the model has {len(model.blocks)}")
    if isinstance(note, str) and note != "self":
        raise ValueError(f"note must be 128 numbers or 'self', got {note!r}")
    x = x[:, 1:]
    batch, T, n_features = x.shape
    n_birds = n_features // 2
    tokens = x.view(batch, T, n_birds, 2).permute(0, 2, 1, 3)
    tokens = tokens.reshape(batch, n_birds, 2 * T)
    emb = model.embedding(tokens)

    for k in range(len(model.blocks)):
        this_block = model.blocks[k]
        if k == block:
            if isinstance(note, str):
                emb = emb + self_listening(this_block.attention, this_block.norm_one(emb), pair)
            else:
                emb = emb + note
            emb = emb + this_block.feedforward(this_block.norm_two(emb))
        else: #normal listening round
            emb = this_block(emb, pair)


    ans = model.head(model.norm(emb)) #each bird's answer
    return ans.reshape(batch, n_features)


def model_inputs(checkpoint, chunk, physics):
    #the same two things train.py hands the model: squashed moves and pair geometry
    x = torch.from_numpy(train.apply_squash(checkpoint["x_scaler"], train.to_moves(chunk))).float()
    pair = torch.from_numpy(train.pair_geometry(chunk, bounds=physics["bounds"], scale=physics["separation_radius"]))
    return x, pair


def average_listening(checkpoint, position_windows, *, physics, block):
    #the note block number "block" writes on average, over every bird in every window
    if checkpoint["config"]["kind"] != "rel_per_bird":
        raise ValueError("model must be 'rel_per_bird'")
    model = checkpoint["model"]
    if not (0 <= block < len(model.blocks)):
        raise ValueError(f"no block {block}, the model has {len(model.blocks)}")
    this_block = model.blocks[block]

    #a running sum and a count, divided once at the end, so a short last batch
    #counts as much per bird as a full one
    total = 0
    count = 0
    for start in range(0, len(position_windows), 256):
        chunk = position_windows[start:start + 256]
        x, pair = model_inputs(checkpoint, chunk, physics)
        with torch.no_grad():
            drafts = model.hidden_states(x, pair)
            #block k reads draft k
            heard = this_block.attention(this_block.norm_one(drafts[block]), pair)
        #heard is (windows, birds, 128)
        total = total + heard.sum(dim=(0, 1))
        count = count + heard.shape[0] * heard.shape[1]
    return total / count


def predict(checkpoint, position_windows, *, physics, block=None, note=None):
    #the model's answers in real units (change in velocity), normal or with one block switched off
    if block is None and note is not None:
        raise ValueError("a note needs a block to go into")
    if block is not None and note is None:
        raise ValueError("switching a block off needs a note to put in its place")
    model = checkpoint["model"]
    answers = []

    for start in range(0, len(position_windows), 256):
        chunk = position_windows[start:start + 256]
        x, pair = model_inputs(checkpoint, chunk, physics)
        with torch.no_grad():
            if block is None:
                out = model(x, pair)
            else:
                out = ablated_forward(model, x, pair, block=block, note=note)

        answers.append(out.numpy())
    #these runs trained on the plain target (--residual none), so un-squashing is all it takes
    return train.cancel_squash(checkpoint["y_scaler"], np.concatenate(answers))


def score(prediction, truth):
    #the same R2 train.py prints: 1 is perfect, 0 is guessing the average
    miss = ((prediction - truth) ** 2).sum()
    spread = ((truth - truth.mean(axis=0)) ** 2).sum()
    return float(1 - miss / spread)


def rules_explain(targets, labels, *, dt, tol=1e-5):
    #one flag per bird per window: True where the three pushes add up to the real change.
    #the rest hit a speed limit, and there the pushes are not the whole story
    total = 0
    for rule in RULES:
        total = total + np.asarray(labels[rule], dtype=np.float64).reshape(-1, 2) * dt

    miss = np.abs(np.asarray(targets, dtype=np.float64).reshape(-1, 2) - total)
    return miss.max(axis=1) < tol


def pushes_and_lost(lost, labels, dt, keep):
    #the three pushes (times dt) as three columns, and what was lost as one,
    #x and y of every kept bird one after the other
    lost = np.asarray(lost, dtype=np.float64).reshape(-1, 2)
    if keep is None:
        keep = np.ones(len(lost), dtype=bool)
    columns = []
    for rule in RULES:
        rule_push = np.asarray(labels[rule], dtype=np.float64).reshape(-1, 2) * dt
        columns.append(rule_push[keep].reshape(-1))
    pushes = np.stack(columns, axis=1)
    return pushes, lost[keep].reshape(-1)


def push_lost(lost, labels, *, dt, keep=None):
    #how much of each rule's push the switched off model stopped doing
    #lost = a * separation + b * alignment + c * cohesion (each push times dt), fitted together
    #so pushes that point against each other are told apart, 1 = all of it gone, 0 = untouched
    #a fraction, so alignment being the biggest push does not drown the other two
    pushes, lost = pushes_and_lost(lost, labels, dt, keep)
    solution = np.linalg.lstsq(pushes, lost, rcond=None)
    fractions = solution[0]
    result = {}
    for r in range(len(RULES)):
        result[RULES[r]] = float(fractions[r])
    return result


def push_explained(lost, labels, *, dt, keep=None):
    #how much of what was lost the three pushes account for: 1 = all of it, 0 = none.
    #near 0 means the fractions from push_lost describe almost none of the damage
    pushes, lost = pushes_and_lost(lost, labels, dt, keep)
    total = (lost ** 2).sum()
    if total == 0:
        return 0.0 #nothing was lost, so there is nothing to explain
    solution = np.linalg.lstsq(pushes, lost, rcond=None)
    left_over = ((lost - pushes @ solution[0]) ** 2).sum()
    return float(1 - left_over / total)


def ablation_report(checkpoint, learn_windows, check_windows, check_targets, check_labels, *, physics):
    #every block switched off in turn: the average note, a blank one, and hearing only yourself
    #the average note comes from the learn clips, the damage is scored on the check clips
    dt = physics["dt"]
    normal = predict(checkpoint, check_windows, physics=physics)
    keep = rules_explain(check_targets, check_labels, dt=dt)
    d_model = checkpoint["model"].head.in_features

    rows = []
    for block in range(len(checkpoint["model"].blocks)):
        notes = {"average": average_listening(checkpoint, learn_windows, physics=physics, block=block),
                 "zero": torch.zeros(d_model),
                 "self": "self"}
        
        for name in NOTES:
            ablated = predict(checkpoint, check_windows, physics=physics, block=block, note=notes[name])
            #what the switched off model no longer does
            lost = normal - ablated
            rows.append({"block": block,
                         "note": name,
                         "score": score(ablated, check_targets),
                         "lost": push_lost(lost, check_labels, dt=dt, keep=keep),
                         "explained": push_explained(lost, check_labels, dt=dt, keep=keep)})

    return {"normal": score(normal, check_targets), "kept": float(keep.mean()), "rows": rows}


def self_ablation(checkpoint, check_windows, check_targets, check_labels, *, physics):
    #one snapshot: how much of each push the model does at all, and how much of it
    #goes away when one round hears only itself. no average note, so no learn clips needed
    dt = physics["dt"]
    keep = rules_explain(check_targets, check_labels, dt=dt)
    normal = predict(checkpoint, check_windows, physics=physics)
    blocks = []
    for block in range(len(checkpoint["model"].blocks)):
        ablated = predict(checkpoint, check_windows, physics=physics, block=block, note="self")
        lost = normal - ablated
        blocks.append({"block": block,
                       "score": score(ablated, check_targets),
                       "lost": push_lost(lost, check_labels, dt=dt, keep=keep),
                       "explained": push_explained(lost, check_labels, dt=dt, keep=keep)})
    #the same fit as push_lost, on the whole answer: 1 = the model does all of that rule
    done = push_lost(normal, check_labels, dt=dt, keep=keep)
    return {"normal": score(normal, check_targets), "done": done, "blocks": blocks}


def ablation_curve(checkpoint_dir, check_windows, check_targets, check_labels, *, physics):
    #self_ablation for every snapshot, in training order
    rows = []
    for path in probe.step_files(checkpoint_dir):
        checkpoint = train.load_checkpoint(path)
        row = self_ablation(checkpoint, check_windows, check_targets, check_labels, physics=physics)
        row["batch"] = probe.batch_of(path)
        row["pass"] = float(checkpoint["pass_number"])
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--clips", type=int, required=True)
    parser.add_argument("--learn-clips", type=int, default=50)
    parser.add_argument("--check-clips", type=int, default=20)
    #which snapshot, the finished model by default
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--out", default=None)
    #every snapshot, hearing only itself, for the curves
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args(argv)
    if args.sweep and args.out is None:
        parser.error("--sweep takes minutes, give it --out so the result is kept")

    with open(Path(args.data) / "config.json") as config_file:
        physics = json.load(config_file)["physics"]

    paths = probe.step_files(args.checkpoint_dir)
    if len(paths) == 0:
        raise ValueError(f"no step_*.pt in {args.checkpoint_dir}, train with --log-checkpoints")
    if args.batch is None:
        batch = probe.batch_of(paths[-1])
    else:
        batch = args.batch
    checkpoint = train.load_checkpoint(Path(args.checkpoint_dir) / f"step_{batch:08d}.pt")
    context = checkpoint["config"]["context"]

    learn, check = probe.pick_clips(args.clips, learn_clips=args.learn_clips, check_clips=args.check_clips)
    check_windows, check_targets = windows.build_windows(args.data, context=context, clips=check)
    check_labels = windows.build_force_labels(args.data, context=context, clips=check)

    if args.sweep:
        curve = ablation_curve(args.checkpoint_dir, check_windows, check_targets, check_labels, physics=physics)
        saved = {"curve": curve, "check_clips": args.check_clips}
        Path(args.out).write_text(json.dumps(saved, indent=1))
        print(f"{len(curve)} snapshots scored, saved to {args.out}")
        return saved

    learn_windows, _ = windows.build_windows(args.data, context=context, clips=learn)

    report = ablation_report(checkpoint, learn_windows, check_windows, check_targets, check_labels, physics=physics)
    result = {"batch": batch,
              "pass": float(checkpoint["pass_number"]),
              "learn_clips": args.learn_clips,
              "check_clips": args.check_clips,
              "normal": report["normal"],
              "kept": report["kept"],
              "rows": report["rows"]}

    print(f"batch {batch} (pass {result['pass']:.1f}), held-out score of the normal model {report['normal']:.4f}")
    print(f"push lost is fitted on the {report['kept']:.0%} of birds not held back by a speed limit")
    print("push lost: 1 = that push is gone, 0 = untouched")
    print("rules explain: how much of the damage those three numbers cover, trust them only when it is high")
    for row in report["rows"]:
        lost = row["lost"]
        print(f"  block {row['block']} off, {row['note']:7s} note   score {row['score']:+.4f}   push lost:"
              f"  separation {lost['separation']:+.2f}  alignment {lost['alignment']:+.2f}  cohesion {lost['cohesion']:+.2f}"
              f"   rules explain {row['explained']:.2f}")

    if args.out is not None:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
