#!/usr/bin/env bash
az vm list-skus \
  --location swedencentral \
  --query "[?capabilities[?name=='GPUs' && to_number(value) >= \`1\`]]" \
  > vms.json
