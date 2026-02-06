#!/bin/bash

#python3 trainer_adam.py --tasks-ids 2 3 --out-dim 6 40 --out-dir=out_adam
#python3 trainer_adam.py --tasks-ids 4 5 --out-dim 32 32 --out-dir=out_adam
python3 trainer_adam.py --tasks-ids 2 4 5 --out-dim 6 32 32 --out-dir=out_adam
python3 trainer_adam.py --tasks-ids 2 3 4 5 --out-dim 6 40 32 32 --out-dir=out_adam
