#!/usr/bin/env bash

mkdir -p data/kcover

python code/kcover.py data/kcover/2-5-0.5/ 1000 5 0.5 2 0 &
python code/kcover.py data/kcover/3-7-0.5/ 1000 7 0.5 3 0 &
python code/kcover.py data/kcover/4-8-0.5/ 1000 8 0.5 4 0 &
python code/kcover.py data/kcover/5-9-0.5/ 1000 9 0.5 5 0 &
