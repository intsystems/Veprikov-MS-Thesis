import argparse
import json
import os
from tqdm import tqdm

from experiment import *
import warnings
warnings.filterwarnings("ignore")

parser = None
folder = None
data_type = None


def create_model(model_params):
    if 'gbr_model' in model_params:
        model_name = 'gbr_model'
        model = lambda : gbr_model(**model_params['gbr_model'])
        print(f"Using model: {model_name}")
        yield model, model_name
    if 'ridgecv_model' in model_params:
        model_name = 'ridgecv_model'
        model = lambda : ridgecv_model(**model_params['ridgecv_model'])
        print(f"Using model: {model_name}")
        yield model, model_name
    if 'ridge_model' in model_params:
        model_name = f"ridge_model_{model_params['ridge_model']['alpha']}"
        model = lambda : ridge_model(**model_params['ridge_model'])
        print(f"Using model: {model_name}")
        yield model, model_name
    if 'linear_model' in model_params:
        model_name = 'linear_model'
        model = lambda : linear_model(**model_params['linear_model'])
        print(f"Using model: {model_name}")
        yield model, model_name
    if 'sgd_model' in model_params:
        model_name = f"sgd_model_{model_params['sgd_model']['max_iter']}"
        model = lambda : sgd_model(**model_params['sgd_model'])
        print(f"Using model: {model_name}")
        yield model, model_name


def hidden_sample(model_params, params, folder):
    data_type = params['data_type']
    control_strategy = params["control_strategy"]
    control_init = params["control_init"]
    
    if data_type == 'california':
        X, y = get_california_dataset(data_len=int(params['N']) // 4)
    elif data_type == 'synthetic':
        X, y = get_synthetic_dataset(float(params['noise']), int(params['random_seed']))
    elif data_type == 'ames_housing':
        X, y = get_ames_dataset(data_len=None)
    else:
        raise ValueError('Wrong data type! ' + data_type)

    for model, model_name in create_model(model_params):
        print(f"Running hidden-sample experiment with data type {data_type}")
        print(f"Model: {model_name}")
        print(f"Control strategy: {control_strategy}, control initial vatue: {control_init}")
        if control_strategy != "none":
            path = f"{folder}/{data_type}/{model_name}/{control_strategy}_{control_init}"
        else:
            path = f"{folder}/{data_type}/{model_name}/none"
        print(f"All results will be uploaded to the folder {path}")
        for trial in tqdm(range(0, int(params['run_times']))):
            os.makedirs(f"{path}/{trial}", exist_ok=True)
            hle = HiddenLoopExperiment(
                X, y, model, model_name, path=f"{path}/{trial}", N=int(params['N']),
                trial=trial, seed=int(params['random_seed']),
                control_strategy=control_strategy, control_init=control_init
            )
            prepare_params = {k: params[k] for k in params.keys() & {'train_size'}}
            hle.prepare_data(**prepare_params)

            loop_params = {k: params[k] for k in params.keys() & {'adherence', 'usage', 'step'}}
            hle.hidden_sampling_experiment(**loop_params)
        

            #results.add_results(**vars(hle))

        #results.plot_multiple_results(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", type=str,
                        help="Kind of experiment: single-model, hidden-loop or hidden-sample")
    parser.add_argument("--params", type=str,
                        help="A json string with experiment parameters")
    parser.add_argument("--model_params", type=str,
                        help="A json string with model name and parameters")
    parser.add_argument("--folder", type=str,
                        help="Save results to this folder", default="./results")
    args = parser.parse_args()
    model_str = args.model_params
    params_str = args.params
    kind = args.kind
    folder = args.folder

    os.makedirs(folder, exist_ok=True)

    model_dict = json.loads(model_str)
    params_dict = json.loads(params_str)

    random_seed = int(params_dict['random_seed'])
    
    init_random_state(random_seed)

    if kind == "hidden-sample":
        #os.makedirs(folder + "/hists", exist_ok=True)
        #os.makedirs(f"{folder}/deviations", exist_ok=True)
        
        hidden_sample(model_dict, params_dict, folder)
    else:
        parser.error("Unknown experiment kind: " + kind)
