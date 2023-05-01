from src.utils import *
from src.model import *
import os

# --------------------------------- ARGPARSE --------------------------------- #

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True, help="type of GNN layer")
parser.add_argument("--outdir", type=str, default="result", help="dataset directory")
parser.add_argument("--data", type=str, required=True, help="dataset name")
parser.add_argument("--task", type=str, required=True, help="dataset task")
parser.add_argument("--device", type=int, default=0, help="CUDA device")
parser.add_argument("--seed", type=int, default=19260817, help="random seed")

parser.add_argument("--max_dis", type=int, default=5, help="distance encoding")
parser.add_argument("--num_layer", type=int, default=6, help="number of layers")
parser.add_argument("--dim_embed", type=int, default=96, help="embedding dimension")

parser.add_argument("--bs", type=int, default=128, help="batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--decay_rate", type=float, default=0.5, help="lr decay rate")
parser.add_argument("--epochs", type=int, default=400, help="training epochs")

args = parser.parse_args()
print(f"""Run:
    model: {args.model}
    data: {args.data}
    task: {args.task}
    seed: {args.seed}
""")

# ----------------------------------- MODEL ---------------------------------- #

if args.model ==  "SWL_SV": layer, pool = ["uL"], "uG"
if args.model ==  "SWL_VS": layer, pool = ["uL"], "vG"
if args.model == "PSWL_SV": layer, pool = ["uL", "vv"], "uG"
if args.model == "PSWL_VS": layer, pool = ["uL", "vv"], "uG"
if args.model == "GSWL"   : layer, pool = ["uL", "vG"], "uG"
if args.model == "SSWL"   : layer, pool = ["uL", "vL"], "uG"
if args.model == "SSWL_P" : layer, pool = ["uL", "vL", "vv"], "uG"

torch.manual_seed(args.seed)
device = torch.device(f"cuda:{args.device}")

from src import dataset
dataloader = {
    name: data.DataLoader(
        dataset.GraphCount(
            split=name,
            root=args.data,
            task=args.task,
            transform=subgraph(layer + [pool]),
        ),
        batch_size=args.bs,
        num_workers=2,
        shuffle=True
    )
    for name in ["train", "val", "test"]
}

model = GNN(idim=args.dim_embed, odim=1,
            max_dis=args.max_dis, encode=False,
            As=[(Agg(layer), args.dim_embed)] * args.num_layer \
              +[(Agg([pool], gin=False), args.dim_embed)])

# ----------------------------------- ITER ----------------------------------- #

def train(model, loader, critn, optim):
    model.train()

    losses = []
    for batch in loader:

        batch = batch.to(device)
        pred = model(batch) \
               .view(batch.y.shape)

        optim.zero_grad()
        loss = critn(pred, batch.y)
        loss.backward()
        optim.step()

        losses.append(loss.item())

    return np.array(losses)

def eval(model, loader, critn):
    model.eval()

    pred, true = [], []
    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            true.append(batch.y)
            pred.append(model(batch) \
                        .view(batch.y.shape))

    return critn(torch.cat(pred), torch.cat(true))

# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

model = model.to(device)
critn = torch.nn.L1Loss()
optim = torch.optim.Adam(model.parameters(), lr=args.lr)

# ------------------------------------ RUN ----------------------------------- #

import numpy as np
record = np.zeros((args.epochs, 4))

from tqdm import tqdm
pbar = tqdm(range(args.epochs))

output_dir = args.outdir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
filename = f"{output_dir}/{args.model}-{args.data}-{args.task}-{args.dim_embed}-{args.bs}-{args.lr}-{args.epochs}-{args.max_dis}-{args.seed}.txt"

for epoch in pbar:

    for group in optim.param_groups:
        group['lr'] = (1 + np.cos(np.pi * epoch / args.epochs)) / 2 * args.lr

    losses = train(model, dataloader["train"], critn, optim)
    val_metric = eval(model, dataloader["val"], critn)
    test_metric = eval(model, dataloader["test"], critn)

    record[epoch] = np.array([optim.param_groups[0]['lr'], losses.mean(), val_metric.item(), test_metric.item()])
    pbar.set_postfix({
        "loss": losses.mean(),
        "val": val_metric.item(),
        "test": test_metric.item()
    })

    np.savetxt(filename, record, delimiter='\t')
