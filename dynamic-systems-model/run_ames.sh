trap 'exit' SIGINT SIGTERM SIGHUP SIGQUIT
for control_strategy in "greedy" "smart_greedy"
do
    export control_strategy &&
    control_init=0.1 data_type="ames_housing" mldev run -f ./experiment.yml --no-commit run_delta_experiment
done
