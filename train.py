import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import flocker
import windows

def split_by_clip(n_clips, *, train_fraction):
    n_train = int(n_clips * train_fraction)
    return range(n_train), range(n_train, n_clips)

def squasher(values):
    values = np.asarray(values)
    mean = values.mean(axis=0)
    std  = values.std(axis = 0)
    #slid = []
    #for i in values:
        #slid.append(i - mean)
    std[std == 0] = 1.0
    return mean, std

def shared_squasher(values):
    values = np.asarray(values)
    n_birds = values.shape[-1] // 2
    birds = values.reshape(*values.shape[:-1], n_birds, 2)
    mean = birds.mean(axis=(0, -2))
    mean = np.tile(mean, n_birds)
    std = birds.std(axis=(0, -2))
    std = np.tile(std, n_birds)
    std[std == 0] = 1.0
    return mean, std
    
def apply_squash(scaler, values):
    mean, std = scaler
    return (values - mean) / std

def cancel_squash(scaler, values):
    mean , std = scaler
    return values * std + mean



def skill(prediction, truth, baseline):
    model_error = ((prediction - truth) ** 2).sum()
    copier_error = ((baseline - truth) ** 2).sum()
    return 1 - model_error / copier_error #please be better

class Model(torch.nn.Module):
    def __init__(self, context, n_features): #192*24
        super().__init__()
        self.layer = torch.nn.Linear(context * n_features, n_features)

    def forward(self, x):
        x = x.flatten(1)
        x = self.layer(x)
        return x

def build_model(*, context, n_features, kind="linear",
                d_model=128, n_heads=4, n_blocks=2): #callable, everywhere
    if kind == "linear":
        return Model(context, n_features)
    if kind == "transformer":
        return flocker.Transformer(n_features=n_features, d_model=d_model,
                                   n_heads=n_heads, n_blocks=n_blocks,
                                   max_context=context)
    if kind == "per_bird":
        return flocker.PerBirdTransformer(n_features=n_features, d_model=d_model,
                                          n_heads=n_heads, n_blocks=n_blocks,
                                          context=context)
    #a typo must not quietly train the wrong model for an hour
    raise ValueError(f"unknown model kind {kind!r}, expected 'linear' or 'transformer' or 'per_bird'")



def save_checkpoint(path, *, model, pass_number, config, x_scaler, y_scaler, score=None):
    torch.save({"state_dict": model.state_dict(), "pass_number": pass_number, "config": config, "x_scaler": x_scaler, "y_scaler": y_scaler, "score": score}, path)
    #pass is look through trsining pile

def load_checkpoint(path):
    checkpoint_data = torch.load(path, weights_only=False)
    checkpoint_model = build_model(**checkpoint_data["config"])
    checkpoint_model.load_state_dict(checkpoint_data["state_dict"]) #loading fresh model with weights, theyre under state_dict
    checkpoint_model.eval()
    return {"model": checkpoint_model,
            "pass_number": checkpoint_data["pass_number"],
            "config": checkpoint_data["config"],
            "x_scaler": checkpoint_data["x_scaler"],
            "y_scaler": checkpoint_data["y_scaler"],
            "score": checkpoint_data["score"]}




#no clip overlapping, and complete splitted coverage
#  train_x, train_y = build_windows("data/clips", context=8, clips=train_clips)
    #test_x,  test_y  = build_windows("data/clips", context=8, clips=test_clips)


def to_moves(position_windows):
    #pos(0), then the move between each pair of ticks
    return np.concatenate([position_windows[:, :1, :], np.diff(position_windows, axis=1)], axis=1)

def permute_slots(position_windows, targets, rng):
    #shuffling bird nametags, while keeping (x,y) real
    if rng is None:
        rng = np.random.default_rng()
    #fresh copies
    pw_out = np.array(position_windows, copy=True)
    tg_out = np.array(targets, copy=True)
    n_rows = pw_out.shape[0]
    #no 12 hardcode
    n_slots = pw_out.shape[-1] // 2
    #shuffling the range(12) array
    perms = np.argsort(rng.random((n_rows, n_slots)), axis=1)
    #vertical to pin each colomn to it's data
    row_idx = np.arange(n_rows)[:, None]
    #(N, 8, 24) -> (N, 8 ,12, (x,y))
    pw_reshaped = pw_out.reshape(pw_out.shape[0], pw_out.shape[1], n_slots, 2)
    #numpy annoying broadcast:
    tg_reshaped = tg_out.reshape(n_rows, n_slots, 2)
    pw_shuffled = np.take_along_axis(pw_reshaped, perms[:, None, :, None], axis=2)
    tg_shuffled = tg_reshaped[row_idx, perms]
    return pw_shuffled.reshape(pw_out.shape), tg_shuffled.reshape(tg_out.shape)


def copier_baseline(position_windows, *, dt):
    #the next velocity change looks like the last one from positions alone
    if position_windows.shape[1] < 3:
        raise ValueError(
            f"the copier needs 3 ticks, got context {position_windows.shape[1]}")
    return (position_windows[:, -1] - 2 * position_windows[:, -2]
            + position_windows[:, -3]) / dt

def clamped_copier_baseline(position_windows, *, dt, min_speed, max_speed):
    curr = (position_windows[:, -1] - position_windows[:, -2]) / dt
    next_velocity = curr + copier_baseline(position_windows, dt=dt)
    pos = np.array(position_windows, copy=True)
    rows = pos.shape[0]
    n_slots = pos.shape[-1] // 2

    final = next_velocity.reshape(rows, n_slots, 2)
    length  = np.linalg.norm(final, axis=-1, keepdims=True)
    wanted  = np.clip(length, min_speed, max_speed)
    length = np.where(length > 0, length, 1.0)
    final   = final * (wanted / length)
    return  (final.reshape(next_velocity.shape) - curr)
    
def main(argv=None):
    parser = argparse.ArgumentParser(description="Train on a window of positions.")
    parser.add_argument("--data", default="data/clips")
    parser.add_argument("--clips", type=int, default=200)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--passes", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=256)
    #1e-3 is what every transformer score in the project was measured at
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--model", choices=["linear", "transformer", "per_bird"], default="linear")
    #relabelling the birds is an exact symmetry of the simulator, so a permuted
    #window is an example it could have produced and did not
    parser.add_argument("--augment", action="store_true")
    #only windows after settling
    parser.add_argument("--settled-only", action="store_true")
    #the baseline is a function of the window alone, so subtracting it rewords
    #the question rather than leaking the answer
    parser.add_argument("--residual", choices=["none", "copier", "clamped"],
                        default="none")
    #0 is never, a 1000-pass sweep must not start filling the disk by accident
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args(argv)

    with open(Path(args.data) / "config.json") as config_file:
        physics = json.load(config_file)["physics"]
    #data/ holds clips at min_speed 2.0 and at 1.0, so a hardcoded floor reads
    #the wrong one silently
    dt = physics["dt"]
    min_speed, max_speed = physics["min_speed"], physics["max_speed"]

    learn, check = split_by_clip(args.clips, train_fraction=args.train_fraction)
    x_learn, y_learn = windows.build_windows(args.data, context=args.context, clips=learn)
    x_check, y_check = windows.build_windows(args.data, context=args.context, clips=check)
    #filter before fitting scalers
    if args.settled_only:
        keep_learn = windows.settled_mask(args.data, context=args.context, clips=learn)
        keep_check = windows.settled_mask(args.data, context=args.context, clips=check)
        if not keep_learn.any():
            raise ValueError("no settled windows in the learn clips")
        if not keep_check.any():
            raise ValueError("no settled windows in the held-out clips")
        x_learn, y_learn = x_learn[keep_learn], y_learn[keep_learn]
        x_check, y_check = x_check[keep_check], y_check[keep_check]

    moves_and_first_pos_learn = to_moves(x_learn) #pos(0), travel(x) = pos(x) -pos(x-1), x = {1,7}
    moves_and_first_pos_check = to_moves(x_check)

    if args.residual == "copier":
        baseline_learn = copier_baseline(x_learn, dt=dt)
        baseline_check = copier_baseline(x_check, dt=dt)
    elif args.residual == "clamped":
        baseline_learn = clamped_copier_baseline(x_learn, dt=dt, min_speed=min_speed, max_speed=max_speed)
        baseline_check = clamped_copier_baseline(x_check, dt=dt, min_speed=min_speed, max_speed=max_speed)
    else:
        baseline_learn, baseline_check = np.zeros_like(y_learn), np.zeros_like(y_check)

    #what the model is asked to learn, and so what the scaler has to describe -
    #residuals are far smaller than the targets they came from
    residual_learn, residual_check = y_learn - baseline_learn, y_check - baseline_check
    
    squash = shared_squasher if args.model == "per_bird" else squasher
    X_mean_std, Y_mean_std = squash(moves_and_first_pos_learn), squash(residual_learn)

    X_check, Y_check = apply_squash(X_mean_std, moves_and_first_pos_check), apply_squash(Y_mean_std, residual_check)

    X_tensor_check, Y_tensor_check = torch.from_numpy(X_check).float(), torch.from_numpy(Y_check).float()
    torch.manual_seed(args.seed)
    model_config = {"kind":args.model, "context": args.context, "n_features": x_learn.shape[2], "d_model": 128, "n_heads": 4, "n_blocks": 2}
    model = build_model(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = torch.nn.MSELoss()
    #its own stream, so the batch order below stays exactly what torch.randperm
    #always gave and every score already recorded still reproduces
    shuffler = np.random.default_rng(args.seed)
    run = 0
    start_time = time.time()
    if args.checkpoint_every:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    
    for passes in range(args.passes): #selection pass 0 as untrained for the model, plot starting from (0.0)
        if args.checkpoint_every and passes % args.checkpoint_every == 0: #falsy guard
            save_checkpoint(checkpoint_dir / f"pass_{passes:04d}.pt", model=model, pass_number=passes, config=model_config, x_scaler=X_mean_std, y_scaler=Y_mean_std )
        total = 0
        batches_number = 0
        indices = torch.randperm(len(x_learn)).numpy() #0,1,.,41019   

        for start in range(0, len(x_learn), args.batch):
            #raw windows, so the permutation happens before to_moves and before
            #the scaler - permuting scaled data mixes columns that were divided
            #by different numbers, and nothing raises
            rows = indices[start:start+args.batch]
            raw_batch, true_velo_batches = x_learn[rows], residual_learn[rows]
            if args.augment:
                raw_batch, true_velo_batches = permute_slots(raw_batch, true_velo_batches, shuffler)
            pos_batches = torch.from_numpy(
                apply_squash(X_mean_std, to_moves(raw_batch))).float()
            true_velo_batches = torch.from_numpy(
                apply_squash(Y_mean_std, true_velo_batches)).float()
            prediction = model(pos_batches)
            loss = loss_function(prediction, true_velo_batches) #computes average wrongness across batch to one number, to pass down to backwards(only accept 1 number)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches_number += 1

        if args.report_every and passes % args.report_every == 0:
            run += 1
            print(f"R2 {1 - total / batches_number}, time: {time.time() - start_time}s, run: {run}")
    if args.checkpoint_every:
        #the loop checkpoints before each pass, so the trained model is not on
        #disk anywhere until this line
        save_checkpoint(checkpoint_dir / f"pass_{args.passes:04d}.pt", model=model,
                        pass_number=args.passes, config=model_config,
                        x_scaler=X_mean_std, y_scaler=Y_mean_std)
    elapsed = time.time() - start_time

    with torch.no_grad():
        predicts = []
        for start in range(0, len(X_tensor_check), args.batch):
            check_batches = X_tensor_check[start:start+args.batch]
            predict_check = model(check_batches)
            predicts.append(predict_check)
        stacked = torch.cat(predicts) #stack to one array
        #the model only ever predicted the leftover, so the baseline goes back on
        pred_real = cancel_squash(Y_mean_std, stacked.numpy()) + baseline_check
        #true_real = train.cancel_squash(Y_mean_std, Y_check) just y_check
        r2 = 1 - ((pred_real - y_check)**2).sum() / ((y_check.mean(axis=0) - y_check)**2).sum()

    print(f"{len(x_learn)} windows, {args.passes} passes, seed {args.seed}, {elapsed}s")
    print(f"R2 model {r2}")
    if args.context >= 3:  
        copier = copier_baseline(x_check, dt=dt)
        average = np.full_like(y_check, y_check.mean(axis=0))
        print(f"R2 copier {skill(copier, y_check, average)}")
        print(f"skill against the copier {skill(pred_real, y_check, copier)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
