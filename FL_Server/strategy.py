import flwr as fl
import numpy as np
import json
import time
import os

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

from blockchain import verify_update
from reputation import evaluate_clients
from zkp_utils import verify_proof
from model import CNN

from FL_Server.config import NUM_CLIENTS, BASE_LAMBDA, MAX_LAMBDA
from FL_Server.reputation import reputation_manager
from FL_Server.defense import compute_delta
from FL_Server.fedadam import fedadam_update

os.makedirs("history", exist_ok=True)

HISTORY_DIR = os.getenv("HISTORY_DIR", "history")
os.makedirs(HISTORY_DIR, exist_ok=True)


class SecureFLStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        self.faulty_clients = self._parse_faulty_clients(os.getenv("FAULTY_CLIENTS", ""))
        super().__init__(
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            on_fit_config_fn=self._fit_config
        )
        
        self.start_time = None
        self.global_weights = None
        self.history = {
            "global": {"round": [], "accuracy": [], "loss": [], "verification_time": [], "penalty_clients": [], "round_time": []},
            "clients": {}
        }

    @staticmethod
    def _parse_faulty_clients(raw_value):
        if not raw_value:
            return []

        return [int(value.strip()) for value in raw_value.split(",") if value.strip()]

    def _fit_config(self, server_round):
        return {
            "server_round": server_round,
            "faulty_clients": self.faulty_clients,
        }

    def aggregate_fit(self, server_round, results, failures):
        self.start_time = time.time()
        if not results: 
            return None, {}

        clients_info = []
        round_verify_times = []
        penalty_clients = []

        # ===== VERIFY ZKP =====
        for client, fit_res in results:
            metrics = fit_res.metrics
            cid = str(metrics["client_id"])
            params = parameters_to_ndarrays(fit_res.parameters)
            proof = json.loads(metrics.get("proof", "{}"))

            print(f"\nClient {cid} update received")

            start = time.time()
            verified = verify_proof(params, proof)
            verify_elapsed = time.time() - start
            round_verify_times.append(verify_elapsed)

            proof_type = proof.get("proof_type", "legacy_hash")
            print(f"  🔐 Client {cid} ZKP [{proof_type}] → {'✅ PASS' if verified else '❌ FAIL'} ({verify_elapsed:.4f}s)")

            if not verified:
                print(f"❌ zk-SNARK VERIFICATION FAILED for Client {cid}")
                reputation_manager.update_reputation(cid, -1.0) 
                # penalty_this_round.append(cid)
                continue

            verify_update(cid, server_round, True)

            clients_info.append({
                "params": params,
                "client_id": cid,
                "test_acc": metrics.get("local_accuracy", 0),
                "test_loss": metrics.get("local_loss", 0),
                "train_time": metrics.get("train_time", 0)
            })
            
        if not clients_info: 
            return None, {}
        
        # ===== INITIALIZE GLOBAL MODEL =====
        if self.global_weights is None: 
            self.global_weights = clients_info[0]["params"]

        # ===== CLIENT EVALUATION =====
        client_weights_dict = {info["client_id"]: 
            info["params"] for info in clients_info}
        
        eval_results, Q1, Q3 = evaluate_clients(
            self.global_weights, 
            client_weights_dict, 
            clients_info
        )
        
        # ===== IQR OUTLIER DETECTION =====
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # ===== REWARD + GRADIENT =====

        gradients = []
        final_weights = []
        LAMBDA = min(MAX_LAMBDA, BASE_LAMBDA + server_round * 0.005)

        for info in clients_info:
            cid = info["client_id"]
            res = eval_results[cid]

            reputation = res["reputation"]
            score = res["score"]

            # ===== SKIP CLIENT XẤU =====
            if score < -0.4 or reputation < 0.2:
                print(f"🚫 Skip client {cid} (score={score:.3f}, rep={reputation:.3f})")
                penalty_clients.append(cid)
                continue

            # ===== COMPUTE DELTA =====
            delta = compute_delta(self.global_weights, info["params"])

            # ===== IQR OUTLIER DEFENSE =====
            if delta < lower or delta > upper:

                print(f"⚠ Outlier Client {cid} (Δ={delta:.4f})")

                reputation *= 0.7

                penalty_clients.append(cid)
            
            # ===== COMPUTE REWARD =====    
            gamma = np.exp(-BASE_LAMBDA * delta)

            reward = np.sqrt(reputation) * gamma
            reward = np.clip(reward, 0.05, 1.5)

            # ===== COMPUTE GRADIENT =====
            grad = [
                local_w - global_w
                for local_w, global_w in zip(info["params"], self.global_weights)
            ]

            gradients.append(grad)
            final_weights.append(reward)

            print(f"Client {cid} | reputation={reputation:.3f} | reward={reward:.3f}")

            self._update_client_history(
                cid, 
                server_round, 
                info, 
                reputation
            )

        if not gradients:
            print("❌ No valid clients for aggregation")
            return None, {}

        # ===== WEIGHTED GRADIENT AGGREGATION =====
        total_weight = sum(final_weights) + 1e-8
        agg_grad = []

        for layer_idx in range(len(gradients[0])):

            layer_sum = sum(
                grad[layer_idx] * weight
                for grad, weight in zip(gradients, final_weights)
            )

            layer_avg = layer_sum / total_weight

            layer_avg = np.clip(layer_avg, -1, 1)

            agg_grad.append(layer_avg)

        # ===== FEDADAM UPDATE =====
        self.global_weights = fedadam_update(
            self.global_weights,
            agg_grad
        )
        
        # ===== LOGGING =====
        if round_verify_times:
            self.history["global"]["verification_time"].append(float(np.mean(round_verify_times)))
            self.history["global"]["penalty_clients"].append(penalty_clients)

        return ndarrays_to_parameters(self.global_weights), {}
    
    # ================= EVALUATE (Chèn vào sau hàm aggregate_fit) =================
    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        accuracies = [
            r.metrics.get("accuracy", 0) 
            for _, r in results
        ]
        
        losses = [
            r.loss 
            for _, r in results
        ]

        avg_acc = float(np.mean(accuracies))
        avg_loss = float(np.mean(losses))

        print(f"\n--- Round {server_round} Global Result ---")
        print(f"Average Accuracy: {avg_acc:.4f} | Average Loss: {avg_loss:.4f}")

        self.history["global"]["round"].append(server_round)
        self.history["global"]["accuracy"].append(avg_acc)
        self.history["global"]["loss"].append(avg_loss)
        
        if self.start_time:
            round_duration = time.time() - self.start_time
            self.history["global"]["round_time"].append(float(round_duration))

        try:
            with open(os.path.join(HISTORY_DIR, "server_history_fedadam.json"), 
                      "w", 
                      encoding="utf-8"
            ) as f:
                json.dump(self.history, f, indent=4)
                
        except Exception as e:
            print(f"❌ Error saving history: {e}")

        return avg_loss, {"accuracy": avg_acc}

    def _update_client_history(self, cid, server_round, info, rep_val):
        cid_str = str(cid)
        if cid_str not in self.history["clients"]:
            self.history["clients"][cid_str] = {"round": [], "test_accuracy": [], "test_loss": [], "reputation": [], "train_time": []}
        
        h = self.history["clients"][cid_str]
        h["round"].append(server_round)
        h["test_accuracy"].append(info["test_acc"])
        h["test_loss"].append(info["test_loss"])
        h["reputation"].append({"round": server_round, "value": float(rep_val)})
        h["train_time"].append(info["train_time"])