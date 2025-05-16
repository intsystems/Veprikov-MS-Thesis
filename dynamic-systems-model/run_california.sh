trap 'exit' SIGINT SIGTERM SIGHUP SIGQUIT
for control_strategy in "none" "greedy" "smart_greedy"
do
    export control_strategy &&
    control_init=0.3 data_type="california" mldev run -f ./experiment.yml --no-commit run_delta_experiment
done
