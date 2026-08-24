#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '\n[%s] %s\n' "$(date +'%H:%M:%S')" "$*"
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

ensure_helm() {
  if command -v helm >/dev/null 2>&1; then
    return
  fi

  log "helm not found; installing Helm 3 into the Cloud Shell environment"
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
}

aks_size_available() {
  local size="$1"
  local found

  found="$({
    az vm list-skus \
      --location "$LOCATION" \
      --resource-type virtualMachines \
      --size "$size" \
      --query "[?name=='$size' && (restrictions==null || length(restrictions)==\`0\`)].name | [0]" \
      -o tsv
  } 2>/dev/null || true)"

  [[ "$found" == "$size" ]]
}

create_aks_cluster_with_fallback() {
  local fallback_sizes=(
    Standard_B2ps_v2
    Standard_B2pls_v2
  )

  local size_candidates=("$NODE_SIZE")
  local candidate
  for candidate in "${fallback_sizes[@]}"; do
    if [[ "$candidate" != "$NODE_SIZE" ]]; then
      size_candidates+=("$candidate")
    fi
  done

  local count_candidates=("$NODE_COUNT")
  if [[ "$NODE_COUNT" != "1" ]]; then
    count_candidates+=("1")
  fi

  local size count
  local quota_related_error="false"
  local err_file
  err_file="$(mktemp)"

  for size in "${size_candidates[@]}"; do
    if ! aks_size_available "$size"; then
      continue
    fi

    for count in "${count_candidates[@]}"; do
      log "Trying AKS create with NODE_SIZE=$size NODE_COUNT=$count"

      if az aks create \
        --resource-group "$RG" \
        --name "$AKS_NAME" \
        --node-count "$count" \
        --node-vm-size "$size" \
        --generate-ssh-keys \
        --attach-acr "$ACR_NAME" \
        > /dev/null 2>"$err_file"; then
        NODE_SIZE="$size"
        NODE_COUNT="$count"
        rm -f "$err_file"
        log "AKS created successfully with NODE_SIZE=$NODE_SIZE NODE_COUNT=$NODE_COUNT"
        return
      fi

      if grep -qiE 'InsufficientVCPUQuota|ErrCode_InsufficientVCPUQuota|OperationNotAllowed|Quota' "$err_file"; then
        quota_related_error="true"
        log "Quota-related failure with NODE_SIZE=$size NODE_COUNT=$count; trying next option"
        continue
      fi

      cat "$err_file" >&2
      rm -f "$err_file"
      die "AKS creation failed with non-quota error for NODE_SIZE=$size NODE_COUNT=$count"
    done
  done

  rm -f "$err_file"
  if [[ "$quota_related_error" == "true" ]]; then
    die "Unable to create AKS due to quota limits after trying multiple NODE_SIZE/NODE_COUNT combinations. Try NODE_COUNT=1, choose a cheaper size (for example Standard_B2s), or request quota increase."
  fi

  die "Could not find an allowed AKS VM size in $LOCATION for this subscription. Set NODE_SIZE manually from: az vm list-skus --location $LOCATION --resource-type virtualMachines --query \"[?restrictions==null].name\" -o tsv"
}

wait_for_jobs() {
  local timeout_seconds="${1:-7200}"
  local start_time now jobs job failed_count completed_count active_count

  start_time="$(date +%s)"

  while true; do
    mapfile -t jobs < <(kubectl get jobs -n "$NAMESPACE" -l "app.kubernetes.io/instance=nt114-fl" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')

    if [[ "${#jobs[@]}" -eq 0 ]]; then
      now="$(date +%s)"
      if (( now - start_time > timeout_seconds )); then
        die "Timed out waiting for jobs to be created in namespace $NAMESPACE"
      fi
      log "Waiting for release jobs to appear in namespace $NAMESPACE"
      sleep 10
      continue
    fi

    log "Tracking ${#jobs[@]} release job(s): ${jobs[*]}"

    failed_count=0
    completed_count=0
    active_count=0
    for job in "${jobs[@]}"; do
      local succeeded failed active
      succeeded="$(kubectl get job "$job" -n "$NAMESPACE" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
      failed="$(kubectl get job "$job" -n "$NAMESPACE" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
      active="$(kubectl get job "$job" -n "$NAMESPACE" -o jsonpath='{.status.active}' 2>/dev/null || true)"

      if [[ "$failed" =~ ^[1-9][0-9]*$ ]]; then
        log "Job failed: $job"
        kubectl logs job/"$job" -n "$NAMESPACE" --tail=200 || true
        failed_count=$((failed_count + 1))
        continue
      fi

      if [[ "$succeeded" =~ ^[1-9][0-9]*$ ]]; then
        completed_count=$((completed_count + 1))
      elif [[ "$active" =~ ^[1-9][0-9]*$ ]]; then
        active_count=$((active_count + 1))
      fi
    done

    if [[ "$failed_count" -gt 0 ]]; then
      die "One or more jobs failed"
    fi

    local incomplete
    incomplete="false"
    for job in "${jobs[@]}"; do
      if ! kubectl get job "$job" -n "$NAMESPACE" -o jsonpath='{.status.succeeded}' 2>/dev/null | grep -Eq '^1$'; then
        incomplete="true"
      fi
    done

    log "Current status:"
    kubectl get jobs -n "$NAMESPACE" -l "app.kubernetes.io/instance=nt114-fl" \
      -o custom-columns='NAME:.metadata.name,SUCCEEDED:.status.succeeded,FAILED:.status.failed,ACTIVE:.status.active,AGE:.metadata.creationTimestamp' || true
    kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/instance=nt114-fl" \
      -o custom-columns='NAME:.metadata.name,PHASE:.status.phase,READY:.status.containerStatuses[*].ready,RESTARTS:.status.containerStatuses[*].restartCount,NODE:.spec.nodeName' || true

    if [[ "$active_count" -gt 0 && "$completed_count" -eq 0 ]]; then
      log "Jobs are still starting or waiting on dependencies; this is often the contract bootstrap or initContainer wait phase."
    fi

    if [[ "$incomplete" == "false" ]]; then
      break
    fi

    now="$(date +%s)"
    if (( now - start_time > timeout_seconds )); then
      die "Timed out waiting for jobs to complete"
    fi

    sleep 15
  done
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd || pwd)"
if [[ ! -f "$ROOT_DIR/Dockerfile" ]]; then
  ROOT_DIR="$(pwd)"
  if [[ ! -f "$ROOT_DIR/Dockerfile" ]]; then
    die "Cannot find Dockerfile in $ROOT_DIR. Ensure you are running the script from the repo root or with correct path."
  fi
fi
cd "$ROOT_DIR"

require_cmd az
require_cmd kubectl
require_cmd git
ensure_helm
require_cmd curl

RG="${RG:-nt114-rg}"
LOCATION="${LOCATION:-centralindia}"
AKS_NAME="${AKS_NAME:-nt114-aks}"
NAMESPACE="${NAMESPACE:-nt114-fl}"
NODE_COUNT="${NODE_COUNT:-3}"
NODE_SIZE="${NODE_SIZE:-Standard_DS2_v2}"
IMAGE_NAME="${IMAGE_NAME:-project-nt114}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
RESULTS_LOCAL_DIR="${RESULTS_LOCAL_DIR:-$ROOT_DIR/nt114-results}"
FAULTY_CLIENTS="${FAULTY_CLIENTS:-}"
ATTACK_MODE="${ATTACK_MODE:-}"
CLEANUP_AFTER_RUN="${CLEANUP_AFTER_RUN:-true}"
KEEP_RESOURCE_GROUP="${KEEP_RESOURCE_GROUP:-false}"

ACR_NAME="${ACR_NAME:-nt114acr$(date +%s | tail -c 6)}"
if [[ ${#ACR_NAME} -gt 50 ]]; then
  ACR_NAME="${ACR_NAME:0:50}"
fi

log "Using repo root: $ROOT_DIR"
log "Resource group: $RG"
log "AKS cluster: $AKS_NAME"
log "ACR: $ACR_NAME"
log "Namespace: $NAMESPACE"

log "Creating resource group and ACR if needed"
az group create --name "$RG" --location "$LOCATION" >/dev/null
if ! az acr show -n "$ACR_NAME" -g "$RG" >/dev/null 2>&1; then
  az acr create --resource-group "$RG" --name "$ACR_NAME" --sku Standard --admin-enabled false >/dev/null
fi

ACR_LOGIN_SERVER="$(az acr show -n "$ACR_NAME" -g "$RG" --query loginServer -o tsv)"

log "Creating AKS cluster if needed"
if ! az aks show --resource-group "$RG" --name "$AKS_NAME" >/dev/null 2>&1; then
  create_aks_cluster_with_fallback
else
  log "AKS cluster already exists; skipping create"
fi

az aks get-credentials --resource-group "$RG" --name "$AKS_NAME" --overwrite-existing >/dev/null
if kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
  log "Namespace $NAMESPACE already exists; deleting it so Helm can create an owned namespace"
  kubectl delete namespace "$NAMESPACE" --ignore-not-found=true >/dev/null
  kubectl wait --for=delete namespace "$NAMESPACE" --timeout=300s >/dev/null
fi

log "Building and pushing the image with ACR build"
az acr build --registry "$ACR_NAME" --image "$IMAGE_NAME:$IMAGE_TAG" . >/dev/null

# [16:43:07] Trying AKS create with NODE_SIZE=Standard_B2ps_v2 NODE_COUNT=3

# [16:49:25] AKS created successfully with NODE_SIZE=Standard_B2ps_v2 NODE_COUNT=3
# WARNING: Merged "nt114-aks" as current context in /home/tran/.kube/config

# [16:49:26] Building and pushing the image with ACR build
# ERROR: Unable to find '/home/Dockerfile'.

log "Deploying Helm release"
HELM_SET_ARGS=(
  --set-string "image.repository=$ACR_LOGIN_SERVER/$IMAGE_NAME"
  --set-string "image.tag=$IMAGE_TAG"
)
if [[ -n "$FAULTY_CLIENTS" ]]; then
  HELM_SET_ARGS+=(--set-string "fl.faultyClients=$FAULTY_CLIENTS")
fi
if [[ -n "$ATTACK_MODE" ]]; then
  HELM_SET_ARGS+=(--set-string "fl.attackMode=$ATTACK_MODE")
fi

log "Deleting previous FL jobs to avoid immutable Job template upgrade errors"
kubectl delete job -n "$NAMESPACE" -l "app.kubernetes.io/instance=nt114-fl" --ignore-not-found=true >/dev/null 2>&1 || true
kubectl wait --for=delete job -n "$NAMESPACE" -l "app.kubernetes.io/instance=nt114-fl" --timeout=180s >/dev/null 2>&1 || true

helm upgrade --install nt114-fl helm/nt114-fl -n "$NAMESPACE" \
  --create-namespace \
  -f helm/nt114-fl/values-aks.yaml \
  "${HELM_SET_ARGS[@]}"

log "Waiting for Kubernetes jobs to finish"
wait_for_jobs 7200

log "Jobs completed"
kubectl get jobs -n "$NAMESPACE"

RESULTS_LOCAL_DIR="$(cd "$(dirname "$RESULTS_LOCAL_DIR")" && pwd)/$(basename "$RESULTS_LOCAL_DIR")"
mkdir -p "$RESULTS_LOCAL_DIR"

POD_NAME="$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep -m1 'results-viewer' || true)"
if [[ -z "$POD_NAME" ]]; then
  POD_NAME="$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}')"
fi

log "Copying results from pod: $POD_NAME"
kubectl cp -n "$NAMESPACE" "$POD_NAME:/results" "$RESULTS_LOCAL_DIR"
log "Results copied to: $RESULTS_LOCAL_DIR"

if [[ "$CLEANUP_AFTER_RUN" == "true" ]]; then
  log "Cleaning up Helm release and Kubernetes namespace"
  helm uninstall nt114-fl -n "$NAMESPACE" >/dev/null 2>&1 || true
  kubectl delete namespace "$NAMESPACE" --ignore-not-found=true >/dev/null 2>&1 || true
fi

if [[ "$KEEP_RESOURCE_GROUP" == "true" ]]; then
  log "Keeping Azure resources because KEEP_RESOURCE_GROUP=true"
else
  log "Deleting AKS, ACR, and resource group"
  az aks delete --name "$AKS_NAME" --resource-group "$RG" --yes --no-wait >/dev/null 2>&1 || true
  az acr delete --name "$ACR_NAME" --resource-group "$RG" --yes >/dev/null 2>&1 || true
  az group delete --name "$RG" --yes --no-wait >/dev/null 2>&1 || true
fi

log "Done"
