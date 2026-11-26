# Source this file:  source src/cluster/env.sh
# Local (Mac) settings for talking to the BharatGen HyperPod cluster over SSM.
# Credentials live in ~/.aws/credentials under the "bharatgen" profile (not in this repo).
export PATH="$HOME/.local/bin:$PATH"          # session-manager-plugin lives here
export AWS_PROFILE=bharatgen
export AWS_DEFAULT_REGION=ap-south-1

# bgen-cluster (Slurm, ml.p5.48xlarge = 8xH100). Your Linux account `ravi.sah` exists here.
# NOTE: on bgen-cluster-c the account does NOT exist yet (SSM fails with "RunAs user ravi.sah does not exist").
export OML_CLUSTER_ID=tnnnangk68xy
export OML_LOGIN_NODE=i-0b62c9b3c3ff968fa     # login-group instance id (re-check with `aws sagemaker list-cluster-nodes` if it changes)
export OML_TARGET="sagemaker-cluster:${OML_CLUSTER_ID}_login-group-${OML_LOGIN_NODE}"
export OML_USER=ravi.sah
export OML_REMOTE_DIR=/home/ravi.sah/oml       # == /fsx/ravi.sah/oml, shared with compute nodes
