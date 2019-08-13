#!/bin/bash

nb_actors=1 #3 example with 3 actors
game=space_invaders #pong
#This is the gpu for the learner and then for each actor, this should be of length nb_actors+1
tab_gpu_number=(0 0) #(0 1 1 2) example with 3 actors and 3 gpu, actors 0 and 1 will be on gpu 1, actor 2 on gpu 2 and learner is on gpu 0
if [ "${#tab_gpu_number[@]}" == "$(($nb_actors + 1))" ];
then
    echo "Start learning with $nb_actors actors"

    echo "launching the redis servor first"

    gnome-terminal -e "redis-server redis_rainbow_6379.conf"

    sleep 1

    echo "now launching the learner"

    gnome-terminal -e "python launch_learner.py --nb-actor $nb_actors --game $game --gpu-number ${tab_gpu_number[0]}"

    echo "launching all actors"

    sleep 5

    for index_actor in `seq 1 $nb_actors`;

    do
        gnome-terminal -e "python launch_actor.py --nb-actor $nb_actors --id-actor $(($index_actor - 1)) --game $game --gpu-number ${tab_gpu_number[$index_actor]}"
        sleep 5 # # Let some time between each actors
    done

    echo "Finished to launch the $nb_actors actors"

    echo "Everything started now..."

else
    echo "Wrong number of gpu number regarding number of actors, you should indicate gpu number for learner and for each actors"
fi

