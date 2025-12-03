#!/bin/bash

cd /data/sarth/operational-hydrologic-emulators

# Define variable dataset_name with value 'CAMELS-US_HUCAll_lumped'
# DATASET_NAME='CAMELS-US_HUCAll_lumped'
# PREFIX='CAMELS-US'
DATASET_NAME='hysets_HUCAll_lumped'
PREFIX='HYSETS'

cd experiments/hist/runs
python eval.py --config config.json --params model_params.json --cuda_num 0 --dataset $DATASET_NAME --prediction_save_path "${PREFIX}_hist.pt"
cd /data/sarth/operational-hydrologic-emulators

cd experiments/oper_NoMeteo/runs
python eval.py --config config.json --params model_params.json --cuda_num 0 --dataset $DATASET_NAME --prediction_save_path "${PREFIX}_oper_NoMeteo.pt"
cd /data/sarth/operational-hydrologic-emulators

cd experiments/oper_FilteredERA5/runs
python eval.py --config config.json --params model_params.json --cuda_num 0 --dataset $DATASET_NAME --prediction_save_path "${PREFIX}_oper_FilteredERA5.pt"
cd /data/sarth/operational-hydrologic-emulators

cd experiments/oper_GPM-Final/runs
python eval.py --config config.json --params model_params.json --cuda_num 0 --dataset $DATASET_NAME --prediction_save_path "${PREFIX}_oper_GPM-Final.pt"
cd /data/sarth/operational-hydrologic-emulators