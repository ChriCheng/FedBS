import argparse
from datetime import datetime
import time
import distutils.util

import os
from tqdm import tqdm

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


    # DSGD training loop 
    trace_func("Begin Training (DSGD)")
    progress = tqdm(
        range(args.global_epochs),
        desc=f"Subject {test_set_id} Rounds",
        leave=False,
    )
    #  choose star center 
    center_id = random.choice(client_ids)
    in_neighbor_map = build_star_in_neighbors(client_ids, center_id)
    criterion = nn.CrossEntropyLoss()
    trace_func(f"center id is : {center_id}")

    Round_TestAcc_List = []   
    for epoch in progress:
        # local multi-step SGD 
        for c in clients:
            c.local_model.train()

            optimizer = torch.optim.SGD(
                c.local_model.parameters(),
                lr=args.lr,
                weight_decay=1e-4,
                momentum=0.9,
            )

            # mimic FedAvg: local_epochs over local dataloader
            for _ in range(args.local_epochs):
                for X, y in c.train_dataloader:
                    X, y = X.to(c.device), y.to(c.device)

                    y_hat = c.local_model(X)
                    loss = criterion(y_hat, y)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

        # snapshot AFTER local SGD for synchronous gossip 
        snapshot_states = {
            c.client_id: copy.deepcopy(c.local_model.state_dict())
            for c in clients
        }

        # gossip / consensus step (star topology) 
        gamma_k = float(getattr(args, "gamma", 1))  

        for c in clients:
            cid = c.client_id
            x_i = snapshot_states[cid]
            N_in = in_neighbor_map[cid]
            w = 1.0 / len(N_in) if len(N_in) > 0 else 0.0

            new_state = copy.deepcopy(x_i)

            for name, x_i_name in x_i.items():
                if "running_mean_bmic" in name or "running_std_bmic" in name:
                    continue
                if not torch.is_floating_point(x_i_name):
                    continue

                consensus = torch.zeros_like(x_i_name)
                for nj in N_in:
                    consensus.add_(gamma_k * w * (snapshot_states[nj][name] - x_i_name))

                new_state[name] = x_i_name + consensus

            c.local_model.load_state_dict(new_state)

        # ONE-TIME evaluation on held-out test_dataset (per node) 
        Node_TestAcc_List_round = {}
        accs = []
        losses = []

        for c in clients:
            tl, ta = c.local_eval(c.local_model)
            tl = tl.item()
            
            Node_TestAcc_List_round[c.client_id] = {
                "acc": round(100 * ta, 2),
                "loss": round(tl, 4),
            }
            accs.append(ta)
            losses.append(tl)

        avg_acc = sum(accs) / len(accs)
        avg_loss = sum(losses) / len(losses)


        progress.set_postfix({
            "test_loss": f"{avg_loss:.4f}",
            "test_acc": f"{100*avg_acc:.2f}%",
            "center": str(center_id),
        })

        Round_TestAcc_List.append(Node_TestAcc_List_round)


    progress.close()

    Node_TestAcc_List = Round_TestAcc_List[-1] if len(Round_TestAcc_List) > 0 else {}

    trace_func("Per-node test accuracy (held-out subject):")
    for cid in sorted(Node_TestAcc_List.keys()):
        trace_func(
            f"  Client {cid}: {Node_TestAcc_List[cid]['acc']:.2f}% (loss {Node_TestAcc_List[cid]['loss']:.4f})"
        )
    

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
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="the number of communication rounds patience",
    )
    parser.add_argument(
        "--fedprox",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="if true, perform fedprox",
    )
    parser.add_argument("--mu", type=float, default=1.0, help="mu for fedprox")
    parser.add_argument(
        "--scaffold",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="if true, perform scaffold",
    )
    parser.add_argument(
        "--moon",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="if true, perform moon",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5, help="the temperature for moon"
    )
    parser.add_argument("--mu_moon", type=float, default=1.0, help="mu for moon")
    parser.add_argument(
        "--fedfa",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="if true, perform fedfa",
    )
    parser.add_argument("--prob", type=float, default=0.5, help="probability for fedfa")
    parser.add_argument(
        "--GA",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="if true, perform GA",
    )
    parser.add_argument(
        "--step_size", type=float, default=0.05, help="the step size of GA"
    )
    parser.add_argument(
        "--fedbs",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="if true, perform fedbs",
    )
    parser.add_argument("--rho", type=float, default=0.1, help="rho for fedbs")

    

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
    id =1
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


    print("=" * 70)

    for r, round_dict in enumerate(TestAcc_List, start=1):
        client_ids = sorted(round_dict.keys())
        accs = [round_dict[cid]["acc"] for cid in client_ids]
        losses = [round_dict[cid]["loss"] for cid in client_ids]

        avg_acc = sum(accs) / len(accs)
        avg_loss = sum(losses) / len(losses)

        col_w = 8  # 每列宽度
        sep = "|"

        print(f"Round {r}")
        print("-" * 70)

        # header
        header = f"{'Client':<6} {sep} " + " ".join(
            f"{cid:>{col_w}d}" for cid in client_ids
        ) + f" {sep} {'AVG':>{col_w}}"
        print(header)

        print("-" * 70)

        # acc row
        acc_row = f"{'Acc':<6} {sep} " + " ".join(
            f"{acc:>{col_w}.2f}" for acc in accs
        ) + f" {sep} {avg_acc:>{col_w}.2f}"
        print(acc_row)

        # loss row
        loss_row = f"{'Loss':<6} {sep} " + " ".join(
            f"{loss:>{col_w}.4f}" for loss in losses
        ) + f" {sep} {avg_loss:>{col_w}.4f}"
        print(loss_row)

        print("=" * 70)
