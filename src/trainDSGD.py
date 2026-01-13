import argparse
import json
import random
import copy
from datetime import datetime
import time
import distutils.util

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import pandas as pd

from client import *
from datasets import *
from utils import *
from otherFedComponents import *


def train(
    args,
    test_set_id,
    client_subject_id,
    TestAcc_List,
    trace_func=print,
    save_path="./checkpoint.pth",
):
    import copy
    import random
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm


    seed = random.randint(1, 100)
    data_transform = [
        ArrayToTensor(),
    ]
    label_transform = [ArrayToTensor()]

    test_dataset = MIDataset(
        random_state=seed,
        subject_id=test_set_id,
        root=args.data_path,
        mode="all",
        data_transform=data_transform,
        label_transform=label_transform,
    )

    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=False,
        delta=0,
        path=save_path,
        trace_func=trace_func,
        counter_info=False,
        is_save=True,
        early=args.early,
    )

    trace_func(f"Begin Initing Clients {client_subject_id}")
    clients = []
    for sid in client_subject_id:
        sid_list = [sid]
        clients.append(
            Client(
                args,
                MIDataset(
                    random_state=seed,
                    subject_id=sid_list,
                    root=args.data_path,
                    mode="all",
                    data_transform=data_transform,
                    label_transform=label_transform,
                ),
                test_dataset, 
                id=sid,
            )
        )

    client_ids = [c.client_id for c in clients]

    # Star topology 
    def build_star_in_neighbors(ids, center_id):
        in_map = {}
        for cid in ids:
            if cid == center_id:
                in_map[cid] = [j for j in ids if j != center_id]  # all leaves
            else:
                in_map[cid] = [center_id]  # the hub
        return in_map

    trace_func("Init decentralized STAR topology with random center per round")

    # Test evaluator (held-out subject)
    def evaluate_model(model, dataset, device):
        test_loader = DataLoader(dataset, batch_size=8, shuffle=False)
        model.eval()
        test_acc = 0.0
        test_loss = 0.0
        crit_sum = torch.nn.CrossEntropyLoss(reduction="sum")

        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                y_hat = model(X)
            loss_sum = crit_sum(y_hat, y)
            test_loss += loss_sum.item()
            pred = y_hat.max(-1, keepdim=True)[1]
            y_true = y.max(-1, keepdim=True)[1]
            test_acc += pred.eq(y_true).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader.dataset)
        return test_loss, test_acc

    criterion = torch.nn.CrossEntropyLoss()  
    lambda0 = float(getattr(args, "lr", 0.005))
    gamma0 = float(getattr(args, "gamma", 1.0))

    # DSGD training loop 
    trace_func("Begin Training (DSGD, full-batch grad via Client.local_fullbatch_grad)")
    progress = tqdm(
        range(args.global_epochs),
        desc=f"Subject {test_set_id} Rounds",
        leave=False,
    )
    # ---- choose star center randomly each round ----
    center_id = random.choice(client_ids)
    in_neighbor_map = build_star_in_neighbors(client_ids, center_id)
    for epoch in progress:        
        # ---- snapshot x^k for all clients ----
        snapshot_states = {
            c.client_id: copy.deepcopy(c.local_model.state_dict())
            for c in clients
        }

        # ---- compute full-batch gradients ∇f_i(x_i^k) via client wrapper ----
        grad_map = {c.client_id: {} for c in clients}

        mean_loss = 0.0
        mean_acc = 0.0

        for c in clients:
            # Ensure grads are computed at x_i^k
            c.local_model.load_state_dict(snapshot_states[c.client_id])

            # Get grad
            grad_dict, full_batch_loss = c.local_fullbatch_grad(c.local_model, criterion)

            # Fill grad_map (we will later index by state_dict keys)
            for name, g in grad_dict.items():
                grad_map[c.client_id][name] = g.detach().clone()

            mean_loss += float(full_batch_loss)

            # Optional monitor acc (keeps your current behavior)
            _, a = c.local_eval(c.local_model)
            mean_acc += float(a)

        mean_loss /= len(clients)
        mean_acc /= len(clients)

        progress.set_postfix(
            {
                "loss": f"{mean_loss:.4f}",
                "acc": f"{100*mean_acc:.2f}%",
                "center": str(center_id),
            }
        )

        # ---- apply DSGD update exactly like the original form (no simplification) ----
        gamma_k = gamma0
        lambda_k = lambda0

        for c in clients:
            cid = c.client_id
            x_i = snapshot_states[cid]  # x_i^k
            N_in = in_neighbor_map[cid]

            # uniform weights w_ij
            w = 1.0 / len(N_in) if len(N_in) > 0 else 0.0

            new_state = copy.deepcopy(x_i)

            for name, x_i_name in x_i.items():
                # keep your "bmic running stats" local (no mixing)
                if "running_mean_bmic" in name or "running_std_bmic" in name:
                    continue

                # only update floating tensors; keep integer buffers as-is
                if not torch.is_floating_point(x_i_name):
                    continue

                # consensus term: sum gamma*w_ij*(x_j^k - x_i^k)
                consensus = torch.zeros_like(x_i_name)
                for nj in N_in:
                    consensus.add_(gamma_k * w * (snapshot_states[nj][name] - x_i_name))

                # gradient term: lambda * grad
                g = grad_map[cid].get(name, None)
                if g is None:
                    grad_term = torch.zeros_like(x_i_name)
                else:
                    # match dtype/device if needed
                    if g.device != x_i_name.device:
                        g = g.to(x_i_name.device)
                    if g.dtype != x_i_name.dtype:
                        g = g.to(x_i_name.dtype)
                    grad_term = lambda_k * g

                new_state[name] = x_i_name + consensus - grad_term

            # write back x_i^{k+1}
            c.local_model.load_state_dict(new_state)

        # ---- build proxy model (global mean) for early stopping / checkpoint ----
        
        if early_stopping.early_stop:
            trace_func(f"Stopped early, stop epoch:{early_stopping.best_epoch+1}")
            trace_func(f"Global Model Val Acc: {100*early_stopping.best_val_acc:.2f}%")
            break

    progress.close()

    if not early_stopping.early_stop:
        trace_func(f"Not stopped early, stop epoch:{early_stopping.best_epoch+1}")
        trace_func(f"Global Model Val Acc: {100*early_stopping.best_val_acc:.2f}%")

    # -------------------------
    # Per-node test on held-out subject
    # Store this train() call's per-node acc into TestAcc_List
    # -------------------------
    Node_TestAcc_List = {}
    trace_func("Per-node test accuracy (held-out subject):")

    for c in clients:
        node_model = copy.deepcopy(c.local_model)
        node_model.eval()

        test_loss, test_acc = evaluate_model(node_model, test_dataset, c.device)
        Node_TestAcc_List[c.client_id] = round(100 * test_acc, 2)

        trace_func(f"  Client {c.client_id}: {100*test_acc:.2f}% (loss {test_loss:.4f})")

    TestAcc_List.append(Node_TestAcc_List)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning for BCIs")

    # About model
    parser.add_argument("--model", type=str, default="eegnet")
    parser.add_argument(
        "--sample_rate", type=int, default=250, help="the sample rate of data"
    )
    parser.add_argument(
        "--F1", type=int, default=8, help="the hyperparameter F1 of eegnet"
    )
    parser.add_argument(
        "--D", type=int, default=2, help="the hyperparameter D if eegnet"
    )
    parser.add_argument(
        "--F2", type=int, default=16, help="the hyperparameter F2 of eegnet"
    )
    parser.add_argument(
        "--class_num", type=int, default=4, help="the classes num of label"
    )
    parser.add_argument(
        "--channels", type=int, default=22, help="the channels num of eeg"
    )
    parser.add_argument(
        "--samples", type=int, default=1001, help="the sampling points in each trial"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="the dropout rate of Dropout layer"
    )

    # About basic training setup
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/BNCI2014001",
        help="path to the datasets",
    )
    parser.add_argument(
        "--sub_id",
        type=str,
        default="1,2,3,4,5,6,7,8,9",
        help="the users of the dataset",
    )
    parser.add_argument(
        "--output_path", type=str, default="./output", help="path to store outputs"
    )
    parser.add_argument(
        "--ea",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=True,
        help="if true, EA was performed on each subject data",
    )

    # About federated training setup
    parser.add_argument(
        "--global_epochs",
        type=int,
        default=200,
        help="the number of global communication rounds",
    )

    parser.add_argument(
        "--local_epochs",
        type=int,
        default=2,
        help="the number of local training epochs on the client",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="size of the batches of client training",
    )
    parser.add_argument(
        "--lr", type=float, default=0.005, help="learning rate of client training"
    )
    parser.add_argument(
        "--early",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="if true, early stopping",
    )
    

    # About setup and hyperparameters for federated approaches
    

    args = parser.parse_args()
    print(args)

    # Create output folders
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("timestamp: ", timestamp)
    save_path = "%s/save_models/%s" % (args.output_path, timestamp)
    os.makedirs(save_path, exist_ok=True)

    print(
        "============================================================================================="
    )

    TestAcc_List = []
    subject_id = [int(i) for i in args.sub_id.split(",")]

    for id in subject_id:
        test_set_id = []
        test_set_id.append(id)
        tmp = subject_id.copy()
        tmp.remove(id)
        client_subject_id = tmp

        print("----------------------------------------------------------------")
        t_start = time.time() 
        print("Test subject ID: ", test_set_id)
        print("Client subject ID: ", client_subject_id)
        train(
            args,
            test_set_id,
            client_subject_id,
            TestAcc_List,
            trace_func=tqdm.write,
            save_path="%s/Model_ServerSub%s.pth" % (save_path, str(id)),
        )
        t_end = time.time()
        print(f"Test set id {test_set_id} Time Cost: {(t_end - t_start)/60:.2f} min")
        print(f"Test set id {test_set_id} Complete")
        print("----------------------------------------------------------------")

    mean = round(sum(TestAcc_List) / len(TestAcc_List), 2)
    TestAcc_List.append(mean)

    print("==============================================================\n")

    for r, round_dict in enumerate(TestAcc_List, start=1):
        client_ids = sorted(round_dict.keys())

        accs = [round_dict[cid]["acc"] for cid in client_ids]
        losses = [round_dict[cid]["loss"] for cid in client_ids]

        avg_acc = sum(accs) / len(accs)
        avg_loss = sum(losses) / len(losses)

        print(f"round {r}:")
        print("client :", " ".join(f"{cid:>4}" for cid in client_ids), " avg")
        print(
            "acc    :",
            " ".join(f"{acc:>4.2f}" for acc in accs),
            f"{avg_acc:>4.2f}",
        )
        print(
            "loss   :",
            " ".join(f"{loss:>4.4f}" for loss in losses),
            f"{avg_loss:>4.4f}",
        )
        print("-" * 60)
