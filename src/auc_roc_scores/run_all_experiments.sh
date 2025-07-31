#!/bin/bash

# Define job variables
num_workers=1
random_seed=1234
experiment_name=error-mechanisms-testing-experiments
dataset_ids=(1046 1049 1050 1063 1067 1068 1461 1464 1480 1494 151 1510 23 23517 251 31 310 37 40701 40975 40994 4135 42493 44 44025 44026 44054 44055 44056 44059 44061 44062 44063 44064 44066 44089 44090 44091 44120 44122 44123 44124 44125 44126 44127 44130 44131 44132 44133 44134 44136 44137 44138 44139 44140 44141 44142 44144 44145 44147 44148 44156 44157 44158 44160 44162 44956 44957 44958 44959 44960 44962 44963 44964 44966 44967 44969 44970 44971 44972 44974 44976 44977 44978 44979 44980 44981 44983 44984 44987 44989 44990 44993 44994 45012 4534 45402 54 6 725 823)
error_rates=(0.1 0.25 0.5 0.75 0.9)

# Result generation - parallelizable
for id in "${dataset_ids[@]}"; do
  for error_rate in "${error_rates[@]}"; do
    echo "Running script with dataset: $id, error rate: $error_rate, experiment name: $experiment_name"
    uv venv exec python testing.py "$num_workers" "$random_seed" "$experiment_name" "$id" "$error_rate"
  done
done

# Result aggregation
uv venv exec python make_data_cube.py
