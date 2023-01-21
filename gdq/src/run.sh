#!/bin/bash

ENV=env0
ITERATION=10
EPISODES=30
SEED=1

LOOKAHEAD=100
RMAX=1000
GDQUCOUNT=1

START=5
GOAL=15
MDP=REAL_S${START}G${GOAL}E${EPISODES}I${ITERATION}
python experiment.py --mdp ${MDP} --env ${ENV} --start ${START} --goal ${GOAL} -i ${ITERATION} -k ${EPISODES} -s ${SEED} -n 100 -r ${RMAX} -u ${GDQUCOUNT}  -a gdq
python experiment.py --mdp ${MDP} --env ${ENV} --start ${START} --goal ${GOAL} -i ${ITERATION} -k ${EPISODES} -s ${SEED} -n 10                             -a dynaq
python graphic.py --mdp ${ENV}${MDP} -w 1
