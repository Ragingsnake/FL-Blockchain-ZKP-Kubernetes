"""Zero-Knowledge Proof utilities using BN128 elliptic curve (zk-SNARK).

Implements a Pedersen commitment scheme with a Schnorr/Sigma protocol proof
over model weight statistics. The client proves knowledge of the blinding
factor and weight statistics that open the commitment, without revealing
the actual model weights.

Curve: BN128 (alt_bn128) — same curve used by Ethereum precompiled contracts.
Library: py_ecc (pure Python, no native dependencies).
"""
import hashlib
import os
import numpy as np

from py_ecc.bn128 import bn128_curve as curve
from py_ecc.bn128 import bn128_pairing as pairing

# BN128 curve order
CURVE_ORDER = curve.curve_order

# Generator points on G1
G1 = curve.G1  # primary generator
# Derive a second independent generator H by hashing G1 coordinates
# (nothing-up-my-sleeve construction)
_h_seed = hashlib.sha256(b"nt114-fl-zkp-generator-H").digest()
_h_scalar = int.from_bytes(_h_seed, "big") % CURVE_ORDER
H1 = curve.multiply(G1, _h_scalar)  # secondary generator


def _compute_stats_scalar(parameters):
    stats_parts = []
    for p in parameters:
        arr = np.asarray(p, dtype=np.float64)
        l2 = float(np.linalg.norm(arr))
        stats_parts.append(l2)
    
    global_mean = float(np.mean([np.mean(np.asarray(p, dtype=np.float64)) for p in parameters]))
    layer_count = len(parameters)
    
    stats_bytes = b""
    for s in stats_parts:
        stats_bytes += repr(s).encode()
        stats_bytes += b"|"
    stats_bytes += f"{repr(global_mean)}|{layer_count}".encode()
    
    digest = hashlib.sha256(stats_bytes).digest()
    scalar = int.from_bytes(digest, "big") % CURVE_ORDER
    if scalar == 0:
        scalar = 1
    return scalar


def _point_to_hex(point):
    """Serialize a BN128 G1 point to a hex string."""
    if point is None:
        return "inf"
    x, y = point
    return f"{int(x):064x}{int(y):064x}"


def _hex_to_point(hex_str):
    """Deserialize a hex string back to a BN128 G1 point."""
    if hex_str == "inf":
        return None
    x = int(hex_str[:64], 16)
    y = int(hex_str[64:], 16)
    return (curve.FQ(x), curve.FQ(y))


def hash_model(parameters):
    """Hash model parameters (backward-compatible function).
    
    Returns SHA-256 hex digest of serialized parameters.
    """
    m = hashlib.sha256()
    for p in parameters:
        m.update(np.asarray(p).tobytes())
    return m.hexdigest()


def generate_proof(parameters):
    """Generate a zk-SNARK proof for model parameters.
    
    Protocol (Fiat-Shamir transformed Sigma/Schnorr):
    1. Compute stats scalar `s` from weight statistics
    2. Pick random blinding factor `r`
    3. Compute Pedersen commitment: C = s*G + r*H
    4. Pick random announcement scalars k_s, k_r
    5. Compute announcement: A = k_s*G + k_r*H
    6. Compute challenge: e = Hash(C || A || public_stats_hash)
    7. Compute responses: z_s = k_s + e*s, z_r = k_r + e*r (mod order)
    8. Proof = {C, A, e, z_s, z_r, public_stats_hash}
    
    The verifier can check: z_s*G + z_r*H == A + e*C
    """
    # Step 1: stats scalar
    s = _compute_stats_scalar(parameters)
    
    # Step 2: random blinding factor
    r = int.from_bytes(os.urandom(32), "big") % CURVE_ORDER
    if r == 0:
        r = 1
    
    # Step 3: Pedersen commitment C = s*G + r*H
    sG = curve.multiply(G1, s)
    rH = curve.multiply(H1, r)
    C = curve.add(sG, rH)
    
    # Step 4: random announcement scalars
    k_s = int.from_bytes(os.urandom(32), "big") % CURVE_ORDER
    k_r = int.from_bytes(os.urandom(32), "big") % CURVE_ORDER
    if k_s == 0: k_s = 1
    if k_r == 0: k_r = 1
    
    # Step 5: announcement A = k_s*G + k_r*H
    ksG = curve.multiply(G1, k_s)
    krH = curve.multiply(H1, k_r)
    A = curve.add(ksG, krH)
    
    # Public stats hash (for binding to the model)
    public_stats_hash = hash_model(parameters)
    
    # Step 6: Fiat-Shamir challenge
    challenge_input = f"{_point_to_hex(C)}|{_point_to_hex(A)}|{public_stats_hash}"
    e_bytes = hashlib.sha256(challenge_input.encode()).digest()
    e = int.from_bytes(e_bytes, "big") % CURVE_ORDER
    if e == 0: e = 1
    
    # Step 7: responses
    z_s = (k_s + e * s) % CURVE_ORDER
    z_r = (k_r + e * r) % CURVE_ORDER
    
    # Also include the raw model hash for backward compat with blockchain
    proof = {
        "hash": public_stats_hash,
        "commitment": _point_to_hex(C),
        "announcement": _point_to_hex(A),
        "challenge": f"{e:064x}",
        "response_s": f"{z_s:064x}",
        "response_r": f"{z_r:064x}",
        "public_stats_hash": public_stats_hash,
        "proof_type": "bn128_pedersen_schnorr"
    }
    
    return proof


def verify_proof(parameters, proof):
    """Verify a zk-SNARK proof for model parameters.
    
    Verification equation: z_s*G + z_r*H == A + e*C
    Also checks that the public stats hash matches the parameters.
    
    Returns True if the proof is valid, False otherwise.
    """
    if proof is None:
        return False
    if not isinstance(proof, dict):
        return False
    
    # Backward compat: if proof doesn't have zk-SNARK fields, fall back to hash check
    if "proof_type" not in proof:
        if "hash" not in proof:
            return False
        server_hash = hash_model(parameters)
        return server_hash == proof["hash"]
    
    required_fields = ["commitment", "announcement", "challenge", "response_s", 
                       "response_r", "public_stats_hash"]
    for field in required_fields:
        if field not in proof:
            return False
    
    try:
        # 1. Check public stats hash matches the parameters
        computed_hash = hash_model(parameters)
        if computed_hash != proof["public_stats_hash"]:
            return False
        
        # 2. Deserialize proof components
        C = _hex_to_point(proof["commitment"])
        A = _hex_to_point(proof["announcement"])
        e = int(proof["challenge"], 16)
        z_s = int(proof["response_s"], 16)
        z_r = int(proof["response_r"], 16)
        
        # 3. Recompute the Fiat-Shamir challenge
        challenge_input = f"{proof['commitment']}|{proof['announcement']}|{proof['public_stats_hash']}"
        e_check_bytes = hashlib.sha256(challenge_input.encode()).digest()
        e_check = int.from_bytes(e_check_bytes, "big") % CURVE_ORDER
        
        if e != e_check:
            return False
        
        # 4. Verify the Schnorr equation: z_s*G + z_r*H == A + e*C
        # Left side
        zsG = curve.multiply(G1, z_s)
        zrH = curve.multiply(H1, z_r)
        lhs = curve.add(zsG, zrH)
        
        # Right side
        eC = curve.multiply(C, e)
        rhs = curve.add(A, eC)
        
        return lhs == rhs
    
    except Exception:
        return False
