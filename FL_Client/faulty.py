import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

TOTAL_CLIENTS = 5
LAMBDA = 1 

LAST_FAULTY = set()

# Attack mode: "noise" (default/legacy) or "label_flip"
ATTACK_MODE = os.getenv("ATTACK_MODE", "noise")

def get_faulty_clients(round_num):
    global LAST_FAULTY

    num_faulty = np.random.poisson(LAMBDA)
    num_faulty = min(num_faulty, TOTAL_CLIENTS      )

    candidates = list(set(range(1, TOTAL_CLIENTS + 1)) - LAST_FAULTY)

    if len(candidates) < num_faulty:
        candidates = list(range(1, TOTAL_CLIENTS + 1))

    faulty_clients = list(np.random.choice(candidates, num_faulty, replace=False))

    LAST_FAULTY = set(faulty_clients)

    print(f"[Round {round_num}] Faulty clients: {faulty_clients} (Poisson λ={LAMBDA})")

    return faulty_clients


def is_faulty_client(client_id, round_num):

    faulty_clients = get_faulty_clients(round_num)

    return client_id in faulty_clients

def corrupt_parameters(params):

    corrupted = []

    for p in params:
        noise = np.random.normal(0, 0.01, p.shape)

        corrupted.append(p + noise)

    return corrupted


def flip_labels(trainloader, num_classes=62):
    """Label-flipping attack: reverses all labels.
    
    Maps label L to (num_classes - 1 - L), causing the model to
    learn systematically wrong associations. This is a more 
    sophisticated attack than random noise because the resulting
    model weights are structurally valid but semantically poisoned.
    """
    all_data = []
    all_targets = []
    
    for data, target in trainloader:
        all_data.append(data)
        # Flip: label L → (num_classes - 1 - L)
        flipped = (num_classes - 1) - target
        all_targets.append(flipped)
    
    all_data = torch.cat(all_data, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    flipped_dataset = TensorDataset(all_data, all_targets)
    flipped_loader = DataLoader(
        flipped_dataset,
        batch_size=trainloader.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    return flipped_loader