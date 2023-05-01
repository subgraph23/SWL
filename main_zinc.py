from src.utils import *
from src.model import *

# --------------------------------- ARGPARSE --------------------------------- #

import argparse
import os
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True, help="type of GNN layer")
parser.add_argument("--dir", type=str, default="zinc", help="dataset directory")
parser.add_argument('--full', dest="subset", action="store_false", help="run full ZINC")
parser.add_argument('--subset', dest="subset", action="store_true", help="run subset of ZINC")
parser.add_argument("--device", type=int, default=0, help="CUDA device")
parser.add_argument("--seed", type=int, default=1, help="random seed")

parser.add_argument("--max_dis", type=int, default=5, help="distance encoding")
parser.add_argument("--num_layer", type=int, default=6, help="number of layers")
parser.add_argument("--dim_embed", type=int, default=96, help="embedding dimension")

parser.add_argument("--bs", type=int, default=128, help="batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--wd", type=float, default=0, help="weight decay")
parser.add_argument("--epochs", type=int, default=400, help="training epochs")

parser.add_argument("--outdir", type=str, default="result", help="output directory")

# ----------------------------------- MODEL ---------------------------------- #

args = parser.parse_args()

if args.model ==  "SWL_VS": layer, pool = ["uL"], "uG"
if args.model ==  "SWL_SV": layer, pool = ["uL"], "vG"
if args.model == "PSWL_VS": layer, pool = ["uL", "vv"], "uG"
if args.model == "PSWL_SV": layer, pool = ["uL", "vv"], "vG"
if args.model == "GSWL"   : layer, pool = ["uL", "vG"], "uG"
if args.model == "SSWL"   : layer, pool = ["uL", "vL"], "uG"
if args.model == "SSWL_P" : layer, pool = ["uL", "vL", "vv"], "uG"
if args.model == "SSWL_G" : layer, pool = ["uL", "vL", "vG"], "uG"
if args.model == "SSWL_PG": layer, pool = ["uL", "vL", "vv", "vG"], "uG"

torch.manual_seed(args.seed)
device = torch.device(f"cuda:{args.device}")
dataloader = {
    name: data.DataLoader(
        pyg.datasets.ZINC(
                    split=name,
                    root=args.dir,
                    subset=args.subset,
                    transform=subgraph(layer + [pool])),
                    batch_size=args.bs,
                    num_workers=4,
                    shuffle=True)
    for name in ["train", "val", "test"]
}

model = GNN(args.dim_embed, 1, args.max_dis, True,
            As=[(Agg(layer), args.dim_embed)] * args.num_layer \
            + [(Agg([pool], gin=False), args.dim_embed)])

# ------------------------------ MODEL PARAMETERS ---------------------------- #

print(f"model size = {sum(param.numel() for param in model.parameters())}")
unused_atom_embed_params = sum(sum(param.numel() for param in m.parameters())
                               - 30 * args.dim_embed for m in model.modules()
                               if isinstance(m, AtomEncoder))
unused_bond_embed_params = sum(sum(param.numel() for param in m.parameters())
                               - 5 * args.dim_embed for m in model.modules()
                               if isinstance(m, BondEncoder))
print(f"unused parameters = {unused_atom_embed_params + unused_bond_embed_params}")

# ------------------------------------ RUN ----------------------------------- #


def train(model, loader, critn, optim):
    model.train()

    loss_list = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).view(batch.y.shape)

        optim.zero_grad()
        loss = critn(pred, batch.y)
        loss.backward()
        optim.step()

        loss_list.append(loss.item())
    
    return loss_list


def eval(model, loader, critn, split):
    model.eval()

    pred, true = [], []
    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            true.append(batch.y)
            pred.append(model(batch).view(batch.y.shape))

    metric = critn(torch.cat(pred), torch.cat(true))

    return metric.item()


# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

model = model.to(device)
critn = torch.nn.L1Loss()
optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                   mode='min',
                                                   factor=0.5,
                                                   patience=20,
                                                   verbose=True)

tag = f"{args.subset}.{args.model}.{args.bs}.{args.max_dis}.{args.wd}.{args.seed}"
output_dir = args.outdir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_file = open(os.path.join(output_dir, f"{tag}.txt"), 'w')

from tqdm import tqdm
pbar = tqdm(range(args.epochs))

for epoch in pbar:

    loss_list = train(model, dataloader["train"], critn, optim)
    val_metric = eval(model, dataloader["val"], critn, "val")
    test_metric = eval(model, dataloader["test"], critn, "test")
    
    print(optim.param_groups[0]['lr'], np.mean(loss_list), val_metric, test_metric, sep='\t', file=log_file, flush=True)
    pbar.set_postfix({
        "lr": optim.param_groups[0]['lr'],
        "loss": np.mean(loss_list),
        "val": val_metric,
        "test": test_metric
    })

    sched.step(val_metric)

log_file.close()