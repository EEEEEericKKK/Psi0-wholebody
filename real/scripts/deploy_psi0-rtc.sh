#!/bin/bash

PORT=8014
# TASK="Spray the bowl and wipe it and stack it up."
# TASK="Pick bottle and turn and pour into cup."
# TASK="Pick toys into box and lift and turn and put on the chair new"
TASK="g1/Pick_bottle_and_turn_and_pour_into_cup"

cd "$(dirname "$0")/../teleop"

python ../deploy/psi-inference_rtc.py \
    --port "$PORT" \
    --task "$TASK"
