# Required Actions to Run CI/CD Workflows

The implementation for the zk-SNARK ZKP, persistent poisoning attack, and GitHub Actions CI/CD pipelines is complete in the local codebase. 

To use these new features on GitHub, you need to complete the following steps:

## 1. Commit and Push Changes

Commit all the newly created and modified files, and push them to your remote GitHub repository:

```bash
git add .
git commit -m "feat: Add zk-SNARK ZKP, label-flip attack, and CI/CD workflows"
git push
```

## 2. Configure GitHub Secrets

The new GitHub Actions workflows (`.github/workflows/*.yml`) require specific secrets to deploy to Azure Kubernetes Service (AKS).

Go to your GitHub repository -> **Settings** -> **Secrets and variables** -> **Actions** -> **New repository secret**.

Add the following secrets:

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `AZURE_CREDENTIALS` | The JSON output from creating an Azure Service Principal for RBAC. This allows GitHub to run `az login`. | `{"clientId": "...", "clientSecret": "...", "subscriptionId": "...", "tenantId": "..."}` |
| `ACR_NAME` | The name of your Azure Container Registry. | `nt114acr` |
| `AZURE_RG` | The name of the Resource Group to use or create. | `nt114-rg` |
| `AZURE_LOCATION` | The Azure region for deployment. | `centralindia` |

*Note: If `AZURE_RG` or `AZURE_LOCATION` are missing, the workflows will default to `nt114-rg` and `centralindia` respectively.*

### How to get `AZURE_CREDENTIALS`
Run this in your Azure CLI (replace `<subscription-id>`):
```bash
az ad sp create-for-rbac --name "github-actions-nt114" --role contributor --scopes /subscriptions/<subscription-id> --sdk-auth
```
Copy the entire JSON output and paste it as the `AZURE_CREDENTIALS` secret.

## 3. Run the Workflows

Once the code is pushed and secrets are configured, you can trigger the demos from the GitHub Actions tab:

1. Go to the **Actions** tab in your GitHub repository.
2. Select a demo workflow on the left side:
   - **Demo Baseline** (WF3: Clean run, no attacks)
   - **Demo Poisoning Random Noise** (WF4: Random noise attack on clients 3 and 4)
   - **Demo Poisoning Persistent Label-Flip** (WF5: Label-flip attack on client 3)
3. Click **Run workflow** -> **Run workflow**.

The workflow will automatically build the images (calling WF2), deploy the AKS cluster (calling the reusable job), run the demo, and output the results as a downloadable artifact. 

When you are done testing, you can run the **Cleanup Azure** (WF1) workflow to delete the resource group and avoid ongoing Azure charges.
