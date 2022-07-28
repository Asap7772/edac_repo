environments=(antmaze-medium-noisy-v2 antmaze-medium-biased-v2 antmaze-large-noisy-v2 antmaze-large-biased-v2)
num_qs=(10 20 50)
etas=(0.0 1.0 5.0)
num_seeds=2
dry_run=true

start_index=$1
gpus=(0 1 2 3 4 5 6 7)
num_repeats=100
end_index=$(($start_index + $num_repeats * ${#gpus[@]}))

echo "Starting from index $start_index to $end_index"

total_exps=0
for ((i=0; i<$num_seeds; i++)); do
    for environment in ${environments[@]}; do
        for num_q in ${num_qs[@]}; do
            for eta in ${etas[@]}; do
                if [ $start_index -le $total_exps ] && [ $total_exps -lt $end_index ]; then
                    echo "Running experiment $total_exps"
                    which_index=$((total_exps % ${#gpus[@]}))
                    which_gpu=${gpus[$which_index]}
                    echo "GPU: $which_gpu"

                    command="python -m scripts.sac --env_name $environment --num_q $num_q --eta $eta --seed $i"
                    echo $command

                    if [ "$dry_run" = false ]; then
                        export CUDA_VISIBLE_DEVICES=$which_gpu
                        eval $command
                        sleep 20.0
                    fi
                    total_exps=$((total_exps+1))
                fi
            done
        done
    done
done

echo "Total experiments: $total_exps"