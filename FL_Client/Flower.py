import flwr as fl
import torch
import torch.nn as nn
import hashlib
import json
import os

from model import CNN
from utils import load_client_data
from ipfs_utils import upload_to_ipfs
from zkp_utils import generate_proof
from blockchain import submit_update
from FL_Client.train import train
from FL_Client.evaluate import evaluate
from FL_Client.faulty import corrupt_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, split_type):
        self.client_id = client_id
        
        self.model = CNN(num_classes=62).to(DEVICE)  # EMNIST ByClass
        
        self.trainloader, self.testloader = load_client_data(client_id, split_type)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [v.detach().cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = {
            k: torch.tensor(v).to(DEVICE)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict, strict=False)  # strict=False tránh lỗi checkpoint cũ
        
    # def fit(self, parameters, config):
    #     self.set_parameters(parameters)

    #     result = train(self.model, self.trainloader, None, self.criterion)

    #     print(f"[Client {self.client_id}] Acc: {result['accuracy']:.4f} | Loss: {result['loss']:.4f}")

    #     params = self.get_parameters({})

    #     metrics = {
    #         "client_id": self.client_id,
    #         "train_time": result["time"],
    #         "test_acc": result["accuracy"],
    #         "test_loss": result["loss"],
    #     }

    #     return params, len(self.trainloader.dataset), metrics

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        round_num = config.get("server_round", 1)

        faulty_clients = config.get("faulty_clients", [])
        if isinstance(faulty_clients, str):
            faulty_clients = [int(x) for x in faulty_clients.split(",") if x.strip()]
        if not faulty_clients:
            faulty_env = os.getenv("FAULTY_CLIENTS", "")
            faulty_clients = [int(x) for x in faulty_env.split(",") if x.strip()]

        is_faulty = self.client_id in faulty_clients
        attack_mode = os.getenv("ATTACK_MODE", "noise")
        
        if is_faulty:
            print(f"⚠ Client {self.client_id} is FAULTY (mode={attack_mode})")
            
        global_params = [torch.from_numpy(p).to(DEVICE) for p in parameters]
        
        global_params = [p.detach().clone().to(DEVICE) for p in global_params]

        # Label-flipping attack: train on poisoned data
        trainloader_override = None
        if is_faulty and attack_mode == "label_flip":
            from FL_Client.faulty import flip_labels
            trainloader_override = flip_labels(self.trainloader, num_classes=62)
            print(f"🔄 Client {self.client_id} using FLIPPED LABELS for training")
        
        result = train(self.model, self.trainloader, global_params, self.criterion, 
                       trainloader_override=trainloader_override)
        print(f"[Client {self.client_id}] Acc: {result['accuracy']:.4f} | Loss: {result['loss']:.4f}")

        os.makedirs("models/clients", exist_ok=True)
        model_path = f"models/clients/client{self.client_id}_round{round_num}.pth"
        torch.save(self.model.state_dict(), model_path)

        cid = upload_to_ipfs(model_path)
        params = self.get_parameters({})
        if is_faulty and attack_mode != "label_flip":
            params = corrupt_parameters(params)
            print("💣 Sent corrupted update (noise injection)")
        elif is_faulty and attack_mode == "label_flip":
            print("🔄 Sent label-flipped model update (weights look normal, semantics poisoned)")

        proof = generate_proof(params)
        proof_str = json.dumps(proof)
        proof_hash = hashlib.sha256((proof_str + cid).encode()).hexdigest()
        try:
            tx_hash = submit_update(round_num, self.client_id, cid, proof_hash, result["accuracy"])
            print("TX:", tx_hash)
        except Exception as e:
            print("Blockchain error:", e)

        metrics = {
            "client_id": self.client_id,
            "train_time": result["time"],
            "local_accuracy": result["accuracy"],
            "local_loss": result["loss"],
            "cid": cid,
            "proof": proof_str,
        }
        return params, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        result = evaluate(self.model, self.testloader, self.criterion)
        print(f"[Client {self.client_id}] Test Acc: {result['accuracy']:.4f}")
        return result["loss"], len(self.testloader.dataset), result