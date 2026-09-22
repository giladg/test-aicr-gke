"""GKE live test of the Pulumi NVIDIA AICR provider on a real H100.

Zonal GKE Standard cluster with two node pools:
  - a small CPU "system" pool that hosts the AICR operators (the full
    13-component stack requests ~4 vCPU; starving it on the GPU node was a
    hard-won lesson from the EKS campaign), and
  - one spot a3-highgpu-1g (1x NVIDIA H100 80GB, ~$4-6/hr spot) GPU node,
    using preemptible H100 quota.

Deploys the h100-gke-cos-training-kubeflow recipe -- the GKE/COS leaf of the
recipe matrix, previously verified only at the resolution layer.

Cost while running: ~$0.10/hr GKE cluster fee + ~$0.27/hr system pool +
spot H100 (market rate) when a node materializes. `pulumi destroy` when done.
"""

import pulumi
import pulumi_gcp as gcp
import pulumi_kubernetes as k8s
import pulumi_labs_nvidia_aicr as aicr

config = pulumi.Config()
cluster_name = config.get("clusterName") or "aicr-gke-test"
# 0 until the preemptible H100 quota is approved: an empty pool creates
# cleanly (no instances, no quota check) and the rest of the stack deploys.
# Flip with: pulumi config set gpuNodeCount 1 && pulumi up
gpu_node_count = config.get_int("gpuNodeCount") or 0

cluster = gcp.container.Cluster(cluster_name,
    initial_node_count=1,
    remove_default_node_pool=True,
    deletion_protection=False,
    resource_labels={
        "nvidia-aicr": "gpu-test",
        "owner": "gilad",
    },
)

# CPU pool for the AICR control-plane components (operators, prometheus,
# schedulers). Keeping these off the GPU node both avoids CPU starvation and
# lets the spot GPU node come and go without disrupting the stack.
system_pool = gcp.container.NodePool("system-pool",
    cluster=cluster.name,
    node_count=1,
    node_config=gcp.container.NodePoolNodeConfigArgs(
        machine_type="e2-standard-8",
        oauth_scopes=["https://www.googleapis.com/auth/cloud-platform"],
    ),
)

# Single spot H100. Spot consumes the PREEMPTIBLE_NVIDIA_H100_GPUS quota and
# draws from a different capacity pool than on-demand. The managed instance
# group keeps retrying in the background if capacity is momentarily dry.
gpu_pool = gcp.container.NodePool("gpu-pool",
    cluster=cluster.name,
    node_count=gpu_node_count,
    node_config=gcp.container.NodePoolNodeConfigArgs(
        machine_type="a3-highgpu-1g",  # 1x H100 80GB
        spot=True,
        guest_accelerators=[gcp.container.NodePoolNodeConfigGuestAcceleratorArgs(
            type="nvidia-h100-80gb",
            count=1,
            gpu_driver_installation_config=gcp.container.NodePoolNodeConfigGuestAcceleratorGpuDriverInstallationConfigArgs(
                # GKE manages the NVIDIA driver on COS; the gke-cos recipe
                # values expect exactly this arrangement.
                gpu_driver_version="LATEST",
            ),
        )],
        oauth_scopes=["https://www.googleapis.com/auth/cloud-platform"],
    ),
    # GKE normalizes/back-fills nodeConfig server-side; after a refresh the
    # stored state differs cosmetically from the program and Pulumi proposes
    # a REPLACE — which would terminate the live (spot-lottery-won) H100
    # node. The pool's config is static for this rig; ignore the drift.
    opts=pulumi.ResourceOptions(ignore_changes=["node_config"]),
)

# Kubeconfig via the gke-gcloud-auth-plugin exec plugin (same construction as
# the repo's gcp-gke-training example).
kubeconfig = pulumi.Output.all(
    cluster.endpoint,
    cluster.master_auth.cluster_ca_certificate,
).apply(lambda args: f"""apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: {args[1]}
    server: https://{args[0]}
  name: gke-cluster
contexts:
- context:
    cluster: gke-cluster
    user: gke-user
  name: gke-context
current-context: gke-context
kind: Config
users:
- name: gke-user
  user:
    exec:
      apiVersion: client.authentication.k8s.io/v1beta1
      command: gke-gcloud-auth-plugin
      installHint: gcloud components install gke-gcloud-auth-plugin
      provideClusterInfo: true
""")

gpu_stack = aicr.ClusterStack("nvidia-aicr",
    kubeconfig=kubeconfig,
    accelerator="h100",
    service="gke",
    intent="training",
    platform="kubeflow",
    os="cos",
    nodes=1,
    skip_components=[
        # nodewright's GKE tuning package requires a node drain (containerd
        # restart interrupt). On a single-CPU-pool cluster the drain can
        # never succeed -- skyhook cannot evict its own operator pod (PDB)
        # -- so it loops forever, evicting every stack pod each attempt.
        # GPU-node tuning is the package's purpose; skipping it on this
        # test rig loses nothing until multi-node.
        "nodewright-customizations",
    ],
    component_overrides={
        # GKE auto-taints GPU nodes nvidia.com/gpu=present:NoSchedule. The
        # NFD worker must tolerate it or the node never gets its PCI label,
        # gpu-operator sees no GPU nodes, and deploys zero operands
        # ("ready" with an empty fleet). The SDK's own bundler injects
        # accelerated-node tolerations from the registry; the client/v1
        # facade does not (facade gap #3) -- carry it here until the
        # provider synthesizes it.
        "nfd": aicr.ComponentOverrideArgs(
            values={
                "worker": {
                    "tolerations": [{
                        "key": "nvidia.com/gpu",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                    }],
                },
            },
        ),
    },
    # Don't block on readiness: the spot GPU node may materialize after the
    # deploy, and GPU DaemonSets stay pending until it does.
    skip_await=True,
    opts=pulumi.ResourceOptions(depends_on=[system_pool, gpu_pool]),
)

# GKE admission refuses pods carrying system-*-critical priority classes in
# namespaces without a PriorityClass-scoped ResourceQuota; gpu-operator sets
# system-node-critical, so without this its pods can never be created (the
# Deployment sits 0/1 with FailedCreate). The AICR SDK synthesizes this quota
# only in its own bundler (registry gkeCriticalPriority flag, upstream #915)
# -- the client/v1 facade does not (facade gap #2) -- so carry it here until
# the provider does. depends_on the stack: the namespace must exist first;
# the operator's ReplicaSet retries pod creation within seconds once the
# quota lands.
quota_provider = k8s.Provider("quota-k8s", kubeconfig=kubeconfig)
k8s.core.v1.ResourceQuota("gpu-operator-critical-pods",
    metadata=k8s.meta.v1.ObjectMetaArgs(
        name="gpu-operator-critical-pods",
        namespace="gpu-operator",
    ),
    spec=k8s.core.v1.ResourceQuotaSpecArgs(
        hard={"pods": "100"},
        scope_selector=k8s.core.v1.ScopeSelectorArgs(
            match_expressions=[k8s.core.v1.ScopedResourceSelectorRequirementArgs(
                operator="In",
                scope_name="PriorityClass",
                values=["system-node-critical", "system-cluster-critical"],
            )],
        ),
    ),
    opts=pulumi.ResourceOptions(provider=quota_provider, depends_on=[gpu_stack]),
)

pulumi.export("kubeconfig", pulumi.Output.secret(kubeconfig))
pulumi.export("cluster_name", cluster.name)
pulumi.export("recipe_name", gpu_stack.recipe_name)
pulumi.export("recipe_version", gpu_stack.recipe_version)
pulumi.export("deployed_components", gpu_stack.deployed_components)
pulumi.export("component_count", gpu_stack.component_count)
