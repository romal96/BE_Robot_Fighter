import os
import numpy as np
import random
from models import RockSampleModel, Model
from solvers import POMCP, PBVI
from parsers import PomdpParser, GraphViz
from logger import Logger as log


class PomdpRunner:

    def __init__(self, params):
        self.params = params
        if params.logfile is not None:
            log.new(params.logfile)

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

    def run(self, algo, T, **kwargs):
        visualiser = GraphViz(description='tmp')
        params, pomdp = self.params, None
        total_rewards, budget = 0, params.budget

        log.info('~~~ initialising ~~~')
        with PomdpParser(params.env_config) as ctx:
            # creates model and solver
            model = self.create_model(ctx.copy_env())
            pomdp = self.create_solver(algo, model)

            # supply additional algo params
            belief = ctx.random_beliefs() if params.random_prior else ctx.generate_beliefs()

            if algo == 'pbvi':
                #belief_points = ctx.generate_belief_points(kwargs['stepsize'])
                belief_points = pomdp.generate_reachable_belief_points(belief, 50)
                #print(belief_points)
                pomdp.add_configs(belief_points)
            elif algo == 'pomcp':
                pomdp.add_configs(budget, belief, **kwargs)

        # have fun!
        log.info('''
        ++++++++++++++++++++++
            Starting State:  {}
            Starting Budget:  {}
            Init Belief: {}
            Time Horizon: {}
            Max Play: {}
        ++++++++++++++++++++++'''.format(model.curr_state, budget, belief, T, params.max_play))

        for i in range(params.max_play):
            # plan, take action and receive environment feedbacks
            pomdp.solve(T)
            action = pomdp.get_action(belief)
            new_state, obs, reward, cost = pomdp.take_action(action)

            if params.snapshot and isinstance(pomdp, POMCP):
                # takes snapshot of belief tree before it gets updated
                self.snapshot_tree(visualiser, pomdp.tree, '{}.gv'.format(i))
            
            # update states
            belief = pomdp.update_belief(belief, action, obs)
            total_rewards += reward
            budget -= cost

            # print ino
            log.info('\n'.join([
              'Taking action: {}'.format(action),
              'Observation: {}'.format(obs),
              'Reward: {}'.format(reward),
              'Budget: {}'.format(budget),
              'New state: {}'.format(new_state),
              'New Belief: {}'.format(belief),
              '=' * 20
            ]))

            if budget <= 0:
                log.info('Budget spent.')


        log.info('{} games played. Total reward = {}'.format(i + 1, total_rewards))
        return pomdp

    def offsolving(self, algo, T, **kwargs):
        visualiser = GraphViz(description='tmp')
        params, pomdp = self.params, None
        total_rewards, budget = 0, params.budget

        log.info('~~~ initialising ~~~')
        with PomdpParser(params.env_config) as ctx:
            # creates model and solver
            model = self.create_model(ctx.copy_env())
            pomdp = self.create_solver(algo, model)

            # supply additional algo params
            belief = ctx.random_beliefs() if params.random_prior else ctx.generate_beliefs()

            if algo == 'pbvi':
                #belief_points = ctx.generate_belief_points(kwargs['stepsize'])                
                belief_points = pomdp.generate_reachable_belief_points(belief, kwargs['Bsize'])
                #belief_points = pomdp.generate_reachable_belief_points(belief, 500)
                print('Belief points generated: ', len(belief_points))
                pomdp.add_configs(belief_points)
                pomdp.solve(T)

            elif algo == 'pomcp':
                pomdp.add_configs(budget, belief, **kwargs)
                # have fun!
                log.info('''
                ++++++++++++++++++++++
                Starting State:  {}
                Starting Budget:  {}
                Init Belief: {}
                Time Horizon: {}
                Max Play: {}
                ++++++++++++++++++++++'''.format(model.curr_state, budget, belief, T, params.max_play))

                for i in range(params.max_play):
                  # plan, take action and receive environment feedbacks
                    pomdp.solve(T)
                    action = pomdp.get_action(belief)
                    new_state, obs, reward, cost = pomdp.take_action(action)
                    
                    if params.snapshot and isinstance(pomdp, POMCP):
                        # takes snapshot of belief tree before it gets updated
                        self.snapshot_tree(visualiser, pomdp.tree, '{}.gv'.format(i))
            
                    # update states
                    belief = pomdp.update_belief(belief, action, obs)
                    total_rewards += reward
                    budget -= cost
                 
                    # print ino
                    log.info('\n'.join([
                     'Taking action: {}'.format(action),
                     'Observation: {}'.format(obs),
                     'Reward: {}'.format(reward),
                     'Budget: {}'.format(budget),
                     'New state: {}'.format(new_state),
                     'New Belief: {}'.format(belief),
                     '=' * 20
                    ]))
                  
                    if budget <= 0:
                        log.info('Budget spent.')
                log.info('{} games played. Total reward = {}'.format(i + 1, total_rewards))
        return pomdp

    def policy_eval(self, algo, T, **kwargs):
        visualiser = GraphViz(description='tmp')
        params, pomdp = self.params, None
        total_rewards, budget = 0, params.budget

        log.info('~~~ initialising simulations ~~~')
        with PomdpParser(params.env_config) as ctx:
            total_rewards_simulations = []
            for simulation in range(params.sim):                
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
                Starting State:  {}
                Init Belief: {}
                Max Play: {}
                ++++++++++++++++++++++'''.format(model.curr_state, belief, params.max_play))

                for i in range(params.max_play):
                    # plan, take action and receive environment feedbacks
                    if algo == 'pomcp':
                        pomdp.solve(T)
                        
                    if params.random_policy :
                        action = random.choice(pomdp.model.actions)
                    else:
                        action = pomdp.get_action(belief)
                    
                    new_state, obs, reward, cost = pomdp.take_action(action)
                    
                    if params.snapshot and isinstance(pomdp, POMCP):
                        # takes snapshot of belief tree before it gets updated
                        self.snapshot_tree(visualiser, pomdp.tree, '{}.gv'.format(i))
            
                    # update states
                    belief = pomdp.update_belief(belief, action, obs)
                    total_rewards += reward
                    budget -= cost
                 
                    # print ino
                    log.info('\n'.join([
                     'Taking action: {}'.format(action),
                     'Observation: {}'.format(obs),
                     'Reward: {}'.format(reward),
                     'Budget: {}'.format(budget),
                     'New state: {}'.format(new_state),
                     'New Belief: {}'.format(belief),
                     '=' * 20
                    ]))
                  
                    if budget <= 0:
                        log.info('Budget spent.')
                log.info('{} games played. Total reward = {}'.format(i + 1, total_rewards))
                total_rewards_simulations.append(total_rewards) 
            
            exp_total_reward = np.mean(total_rewards_simulations)
            std_exp_total_reward = np.std(total_rewards_simulations)
            print(params.sim, 'simulations played.')
            print('Exp total reward = ', exp_total_reward)
            print('Std Exp total reward = ', std_exp_total_reward)
            log.info('{} simulations played. Exp total reward = {}'.format(params.sim, exp_total_reward))
            log.info('Total rewards observed = {}'.format(total_rewards_simulations))
        return pomdp
    

    def replay(self, algo, T, **kwargs):
        visualiser = GraphViz(description='tmp')
        params, pomdp = self.params, None
        total_rewards, budget = 0, params.budget

        log.info('~~~ initialising experience replay ~~~')
        with PomdpParser(params.env_config) as ctx:
            total_rewards_simulations = []
            for simulation in range(params.sim):                
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
                Starting State:  {}
                Init Belief: {}
                Max Play: {}
                ++++++++++++++++++++++'''.format(model.curr_state, belief, params.max_play))

                for i in range(params.max_play):
                    # plan, take action and receive environment feedbacks
                    if algo == 'pomcp':
                        pomdp.solve(T)
                    action = pomdp.get_action(belief)
                    new_state, obs, reward, cost = pomdp.take_action(action)
                    
                    if params.snapshot and isinstance(pomdp, POMCP):
                        # takes snapshot of belief tree before it gets updated
                        self.snapshot_tree(visualiser, pomdp.tree, '{}.gv'.format(i))
            
                    # update states
                    belief = pomdp.update_belief(belief, action, obs)
                    total_rewards += reward
                    budget -= cost
                 
                    # print ino
                    log.info('\n'.join([
                     'Taking action: {}'.format(action),
                     'Observation: {}'.format(obs),
                     'Reward: {}'.format(reward),
                     'Budget: {}'.format(budget),
                     'New state: {}'.format(new_state),
                     'New Belief: {}'.format(belief),
                     '=' * 20
                    ]))
                  
                    if budget <= 0:
                        log.info('Budget spent.')
                log.info('{} games played. Total reward = {}'.format(i + 1, total_rewards))
                total_rewards_simulations.append(total_rewards) 
            
            exp_total_reward = np.mean(total_rewards_simulations)
            std_exp_total_reward = np.std(total_rewards_simulations)
            print(params.sim, 'simulations played.')
            print('Exp total reward = ', exp_total_reward)
            print('Std Exp total reward = ', std_exp_total_reward)
            log.info('{} simulations played. Exp total reward = {}'.format(params.sim, exp_total_reward))
            log.info('Total rewards observed = {}'.format(total_rewards_simulations))
        return pomdp    
    
    
    
    
    
    