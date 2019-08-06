#!/bin/bash
timidity -iA &
source ./venv/bin/activate sh
python ./rt_pi/instrument.py
