#!/bin/bash

export PROJECT_ID=vietai-research
gcloud config set project ${PROJECT_ID}
gcloud beta services identity create --service tpu.googleapis.com --project $PROJECT_ID

tpu_create() {
    vm_name=$1
    gcloud compute tpus execution-groups create \
        --vm-only \
        --name=$vm_name \
        --zone=us-central1-f \
        --machine-type=n1-standard-8 \
        --tf-version=1.15.5
}

for f in {2..4}; do
    vm_name=translate-$f
    echo $vm_name
    tpu_create $vm_name &
    
    
done