import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

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

def build_model(*, context, n_features): #callable, everywhere
    return Model(context, n_features)  






#no clip overlapping, and complete splitted coverage
#  train_x, train_y = build_windows("data/clips", context=8, clips=train_clips)
    #test_x,  test_y  = build_windows("data/clips", context=8, clips=test_clips)


def to_moves(position_windows):
    #pos(0), then the move between each pair of ticks
    return np.concatenate([position_windows[:, :1, :], np.diff(position_windows, axis=1)], axis=1)


def copier_baseline(position_windows, *, dt):
    #the next velocity change looks like the last one from positions alone
    if position_windows.shape[1] < 3:
        raise ValueError(
            f"the copier needs 3 ticks, got context {position_windows.shape[1]}")
    return (position_windows[:, -1] - 2 * position_windows[:, -2]
            + position_windows[:, -3]) / dt


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train on a window of positions.")
    parser.add_argument("--data", default="data/clips")
    parser.add_argument("--clips", type=int, default=200)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--passes", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=50)
    args = parser.parse_args(argv)

    with open(Path(args.data) / "config.json") as config_file:
        dt = json.load(config_file)["physics"]["dt"]

    learn, check = split_by_clip(args.clips, train_fraction=args.train_fraction)
    x_learn, y_learn = windows.build_windows(args.data, context=args.context, clips=learn)
    x_check, y_check = windows.build_windows(args.data, context=args.context, clips=check)

    moves_and_first_pos_learn = to_moves(x_learn) #pos(0), travel(x) = pos(x) -pos(x-1), x = {1,7}
    moves_and_first_pos_check = to_moves(x_check)

    X_mean_std, Y_mean_std = squasher(moves_and_first_pos_learn), squasher(y_learn)
    X_scaled, Y_scaled = apply_squash(X_mean_std, moves_and_first_pos_learn), apply_squash(Y_mean_std, y_learn)

    X_check, Y_check = apply_squash(X_mean_std, moves_and_first_pos_check), apply_squash(Y_mean_std, y_check)

    X_tensor_learn, Y_tensor_learn = torch.from_numpy(X_scaled).float(), torch.from_numpy(Y_scaled).float()
    X_tensor_check, Y_tensor_check = torch.from_numpy(X_check).float(), torch.from_numpy(Y_check).float()
    torch.manual_seed(args.seed)
    model = build_model(context=args.context, n_features=x_learn.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = torch.nn.MSELoss()
    run = 0
    start_time = time.time()
    for passes in range(args.passes):
        total = 0
        batches_number = 0
        indices = torch.randperm(len(X_tensor_learn)) #0,1,.,41019
        #X_shuffled = 41020 row in random order
        X_shuffled, Y_shuffled = X_tensor_learn[indices], Y_tensor_learn[indices]

        for start in range(0, len(X_tensor_learn), args.batch):
            pos_batches = X_shuffled[start:start+args.batch]
            true_velo_batches = Y_shuffled[start:start+args.batch]
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
    elapsed = time.time() - start_time

    with torch.no_grad():
        predicts = []
        for start in range(0, len(X_tensor_check), args.batch):
            check_batches = X_tensor_check[start:start+args.batch]
            predict_check = model(check_batches)
            predicts.append(predict_check)
        stacked = torch.cat(predicts) #stack to one array
        pred_real = cancel_squash(Y_mean_std, stacked.numpy())
        #true_real = train.cancel_squash(Y_mean_std, Y_check) just y_check
        r2 = 1 - ((pred_real - y_check)**2).sum() / ((y_check.mean(axis=0) - y_check)**2).sum()

    copier = copier_baseline(x_check, dt=dt)
    average = np.full_like(y_check, y_check.mean(axis=0))

    print(f"{len(x_learn)} windows, {args.passes} passes, seed {args.seed}, {elapsed}s")
    print(f"R2 model {r2}")
    print(f"R2 copier {skill(copier, y_check, average)}")
    print(f"skill against the copier {skill(pred_real, y_check, copier)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
