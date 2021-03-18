import argparse
import os
import json
import multiprocessing
from pomdp_runreplay import PomdpRunReplay
from util import ReplayParams


if __name__ == '__main__':
    """
    Parse generic params for the POMDP runner, and configurations for the chosen algorithm.
    Algorithm configurations the JSON files in ./configs

    Example usage:
        > python main.py pomcp --env Tiger-2D.POMDP
        > python main.py pbvi --env Tiger-2D.POMDP
    """
    parser = argparse.ArgumentParser(description='Solve pomdp')
    parser.add_argument('config', type=str, help='The file name of algorithm configuration (without JSON extension)')
    parser.add_argument('--env', type=str, default='GridWorld.POMDP', help='The name of environment\'s config file')
    parser.add_argument('--budget', type=float, default=float('inf'), help='The total action budget (defeault to inf)')
    parser.add_argument('--snapshot', type=bool, default=False, help='Whether to snapshot the belief tree after each episode')
    parser.add_argument('--logfile', type=str, default=None, help='Logfile path')
    parser.add_argument('--random_prior', type=bool, default=False,
                        help='Whether or not to use a randomly generated distribution as prior belief, default to False')
    parser.add_argument('--max_play', type=int, default=100, help='Maximum number of play steps')
    parser.add_argument('--sim', type=int, default=100, help='Maximum number of simulations')
    parser.add_argument('--policyfile', type=str, default='alphavecfile.policy', help='alphaVec policy file')
    parser.add_argument('--option', type=str, default='onsolve', 
                        help='please choose between : onsolve - for online solving; offsolve - for offline solving; simulate - for simulating a policyfile; replay - for a experience replay')
    parser.add_argument('--expfile', type=str, default='data/dfsub_19alldataproc.csv', help='experimental data processed file')
    parser.add_argument('--classif', type=str, default='data/classifier.joblib', help='classifier model')
    parser.add_argument('--fnames', type=str, default='HRV,HRnorm,mode,nav,tank,nbAOI1,nbAOI2,nbAOI3,nbAOI4,nbAOI5', help='a string with a list of features names separated by comma ex. HRV,HRnorm')
    
    args = vars(parser.parse_args())
    params = ReplayParams(**args)

    with open(params.algo_config) as algo_config:
        algo_params = json.load(algo_config)
        runner = PomdpRunReplay(params)
        if params.option == 'onsolve':
            runner.run(**algo_params)
        if params.option == 'offsolve':
            runner.offsolving(**algo_params)
        if params.option == 'simulate':
            runner.policy_eval(**algo_params)
        if params.option == 'replay' :
            runner.replay(**algo_params)
