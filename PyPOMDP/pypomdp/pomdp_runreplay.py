import os
import numpy as np
import pandas as pd
from models import RockSampleModel, Model
from solvers import POMCP, PBVI
from parsers import PomdpParser, GraphViz
from logger import Logger as log
from belief_update_animation import AnimateBeliefPlot

from joblib import dump, load

class PomdpRunReplay:

    def __init__(self, params):
        self.params = params
        if params.logfile is not None:
            log.new(params.logfile)
        if params.expfile is not None:
            self.expfile_data = pd.read_csv(params.expfile)
            index = params.expfile.find('.csv')
            output_name = params.expfile[:index] + 'comp' + params.expfile[index:]
            self.expfile_compdata = pd.read_csv(output_name)
        if params.classif_model is not None:
            self.classif_model = load(params.classif_model)        
        self.features_names = []
        if params.fnames is not None:
            fnames = params.fnames.split(",")
            for feature in fnames:
                self.features_names.append(self.mapping_fnames(feature))
                
            
    def mapping_fnames(self, name):
        mapping={
                "HRV": [0, 'HRV10norm'],
                "HR": [0, 'HR10'],
                "HRnorm": [0, 'nHR10'],
                "nav": [1, 'nav_counts'],
                "tank": [1, 'tank_counts'],
                "space" : [1, 'space_counts'],
                "trees" : [1, 'trees_counts'],
                "nbAOI1" : [1, 'nbAOI1'],
                "nbAOI2" : [1, 'nbAOI2'],
                "nbAOI3" : [1, 'nbAOI3'],
                "nbAOI4" : [1, 'nbAOI4'],
                "nbAOI5" : [1, 'nbAOI5'],
                "durAOI1" : [1, 'durAOI1'],
                "durAOI2" : [1, 'durAOI2'],
                "durAOI3" : [1, 'durAOI3'],
                "durAOI4" : [1, 'durAOI4'],
                "durAOI5" : [1, 'durAOI5'],
                "tank_local_score" : [0, 'tank'],
                "mode" : [0, 'mode']
             }
        return mapping.get(name,"Invalid feature name")
     

    def create_model(self, env_configs):
        """
        Builder method for creating model (i,e, agent's environment) instance
        :param env_configs: the complete encapsulation of environment's dynamics
        :return: concrete model
        """
        MODELS = {
            'RockSample': RockSampleModel,
        }
        return MODELS.get(env_configs['model_name'], Model)(env_configs)

    def create_solver(self, algo, model):
        """
        Builder method for creating solver instance
        :param algo: algorithm name
        :param model: model instance, e.g, TigerModel or RockSampleModel
        :return: concrete solver
        """
        SOLVERS = {
            'pbvi': PBVI,
            'pomcp': POMCP,
        }
        return SOLVERS.get(algo)(model)

    def snapshot_tree(self, visualiser, tree, filename):
        visualiser.update(tree.root)
        visualiser.render('./dev/snapshots/{}'.format(filename))  # TODO: parametrise the dev folder path

    def replay(self, algo, T, **kwargs):
        visualiser = GraphViz(description='tmp')
        params, pomdp = self.params, None
        total_rewards, budget = 0, params.budget

        log.info('~~~ initialising experience replay ~~~')
        ## 4 experiences 
        with PomdpParser(params.env_config) as ctx:

            for simulation in range(4):                
                log.info('~~~ initialising simulation: ' + str(simulation) + '~~~' )
            
                # creates model and solver
                model = self.create_model(ctx.copy_env())
                pomdp = self.create_solver(algo, model)
    
                # supply additional algo params
                belief = ctx.random_beliefs() if params.random_prior else ctx.generate_beliefs()
    
                if algo == 'pbvi':
                    # charging alphavec policy file
                    # belief_points = pomdp.generate_reachable_belief_points(belief, 50)
                    # pomdp.add_configs(belief_points)
                    pomdp.charging_policy(params.policyfile)
                    # pomdp.solve(T)
    
                elif algo == 'pomcp':
                    pomdp.add_configs(budget, belief, **kwargs)
                
                total_rewards = 0
                # have fun!
                log.info('''
                ++++++++++++++++++++++
                Init Belief: {}
                Max Play: {}
                ++++++++++++++++++++++'''.format(belief, params.max_play))
                
                
                for i in range(params.max_play):
                    # plan, take action and receive environment feedbacks
                    if algo == 'pomcp':
                        pomdp.solve(T)
                    # take action
                    action = pomdp.get_action(belief)
                    # new_state, obs, reward, cost = pomdp.take_action(action)
                    # getting exp action
                    exp_action = self.getting_mode_from_expfile(i, simulation, pomdp)
                    
                    if exp_action == -1:
                        log.info('\n'.join([
                        'Observation: {}'.format(obs),
                        'Mission ended'
                        ]))
                        plotting.destroy()
                        break  
                    
                    if params.snapshot and isinstance(pomdp, POMCP):
                        # takes snapshot of belief tree before it gets updated
                        self.snapshot_tree(visualiser, pomdp.tree, '{}.gv'.format(i))
                        
                    if i == 0:
                        plotting = AnimateBeliefPlot(belief,action,exp_action)
                    else:
                        plotting.update(belief, action, exp_action, obs)
            
                    # getting features to play symbolic observation
                    features = self.getting_features_from_expfile(i, simulation, pomdp)
                    #print(features)
                    label = self.classif_model.predict(features)
                    #print(label)
                    # transforming label in pomdp observation
                    available_observations = pomdp.model.observations 
                    obs = available_observations[int(label[0])]
                    
                    belief = pomdp.update_belief(belief, exp_action, obs)

                    # print loginfo
                    log.info('\n'.join([
                     'Observation: {}'.format(obs),
                     'POMDP would take action: {}'.format(action),
                     'action taken during EXPERIMENT: {}'.format(exp_action),
                     'New Belief: {}'.format(belief),
                     '=' * 20
                    ]))
                             
            
        return pomdp
    
    def getting_features_from_expfile(self, time_step, simulation, pomdp):
#        self.features_names = [[0,'HRV10norm'], [0,'HR10'], 
#                            [0,'mode'], [1,'nav_counts'], [1,'tank_counts'], 
#                            [1,'nbAOI1'], [1,'nbAOI2'], [1,'nbAOI3'], [1,'nbAOI4'], [1,'nbAOI5']]
        
        mission_data = self.expfile_data.loc[self.expfile_data['mission'] == simulation+1]
        comp_mission_data = self.expfile_compdata[self.expfile_compdata['mission'] == simulation+1]
        action_time = 600 - (time_step+1)*10
        if action_time in mission_data["time"].values:
            indexf = np.where(mission_data["time"]==action_time)[0][0]
        else:
            indexf = len(mission_data["time"].values)
        if action_time == 590:
            indexb = 0
        else:
            indexb = (indexf - 10)
        #print(indexf,indexb)
        data = mission_data[indexb:indexf]
        #print(data)
        features = []
        #print(self.features_names)
        for f in self.features_names :
            if f[0]>0:
                features.append(np.sum(data[f[1]].values))
            else:
                if "nHR10"==f[1]:
                    #print(data[f[1][1:]].values)
                    features.append(np.mean(data[f[1][1:]].values) - comp_mission_data['HRrest'].values[0])
                else:
                    #print(data[f[1]].values)
                    features.append(np.mean(data[f[1]].values))
        #print(features)    
        return [features]
    
    def getting_mode_from_expfile(self, time_step, simulation, pomdp):
        available_actions = pomdp.model.actions
        mission_data = self.expfile_data.loc[self.expfile_data['mission'] == simulation+1]
        if time_step == 0:
            action_time = 599
        else:
            action_time = 600 - (time_step+1)*10
        
        if action_time in mission_data["time"].values:
            exp_action = available_actions[int(mission_data.loc[mission_data["time"]==action_time]["mode"].values[0])]
        else:
            exp_action = -1
        #print(exp_action)
        return exp_action # mission finished
    
    