import numpy as np
import json

from solvers import Solver
from util.alpha_vector import AlphaVector
from util.json_encoder import toJSON
from util.helper import rand_choice, randint, round
from array import array

MIN = -np.inf

class PBVI(Solver):
    def __init__(self, model):
        Solver.__init__(self, model)
        self.belief_points = None
        self.alpha_vecs = None
        self.solved = False

    def add_configs(self, belief_points):
        Solver.add_configs(self)
        self.alpha_vecs = [AlphaVector(a=-1, v=np.zeros(self.model.num_states))] # filled with a dummy alpha vector
        self.belief_points = belief_points
        self.compute_gamma_reward()

    def compute_gamma_reward(self):
        """
        :return: Action_a => Reward(s,a) matrix
        """

        self.gamma_reward = {
            a: np.frombuffer(array('d', [self.model.reward_function(a, s) for s in self.model.states]))
            for a in self.model.actions
        }

    def compute_gamma_action_obs(self, a, o):
        """
        Computes a set of vectors, one for each previous alpha
        vector that represents the update to that alpha vector
        given an action and observation

        :param a: action index
        :param o: observation index
        """
        m = self.model

        gamma_action_obs = []
        for alpha in self.alpha_vecs:
            v = np.zeros(m.num_states)  # initialize the update vector [0, ... 0]
            for i, si in enumerate(m.states):
                for j, sj in enumerate(m.states):
                    v[i] += m.transition_function(a, si, sj) * \
                        m.observation_function(a, sj, o) * \
                        alpha.v[j]
                v[i] *= m.discount
            gamma_action_obs.append(v)
        return gamma_action_obs

    def solve(self, T):
        if self.solved:
            return

        m = self.model
        print("Steps for PBVI ",T)
        for step in range(T):
            #print("Step ", step)
            # First compute a set of updated vectors for every action/observation pair
            # Action(a) => Observation(o) => UpdateOfAlphaVector (a, o)
            gamma_intermediate = {
                a: {
                    o: self.compute_gamma_action_obs(a, o)
                    for o in m.observations
                } for a in m.actions
            }

            # Now compute the cross sum
            gamma_action_belief = {}
            for a in m.actions:

                gamma_action_belief[a] = {}
                for bidx, b in enumerate(self.belief_points):

                    gamma_action_belief[a][bidx] = self.gamma_reward[a].copy()

                    for o in m.observations:
                        # only consider the best point
                        best_alpha_idx = np.argmax(np.dot(gamma_intermediate[a][o], b))
                        gamma_action_belief[a][bidx] += gamma_intermediate[a][o][best_alpha_idx]

            # Finally compute the new(best) alpha vector set
            self.alpha_vecs, max_val = [], MIN

            for bidx, b in enumerate(self.belief_points):
                best_av, best_aa = None, None

                for a in m.actions:
                    val = np.dot(gamma_action_belief[a][bidx], b)
                    if best_av is None or val > max_val:
                        max_val = val
                        best_av = gamma_action_belief[a][bidx].copy()
                        best_aa = a

                self.alpha_vecs.append(AlphaVector(a=best_aa, v=best_av))

        self.solved = True
        # print(self.alpha_vecs)
        self.saving_policy()
        

    def get_action(self, belief):
        max_v = -np.inf
        best = None
        for av in self.alpha_vecs:
            v = np.dot(av.v, belief)
            if v > max_v:
                max_v = v
                best = av

        return best.action
    
    def update_belief(self, belief, action, obs):
        m = self.model

        b_new = []
        for sj in m.states:
            # print("POMDP PBVI: ", action, obs)
            p_o_prime = m.observation_function(action, sj, obs)
            summation = 0.0
            for i, si in enumerate(m.states):
                p_s_prime = m.transition_function(action, si, sj)
                summation += p_s_prime * float(belief[i])
            b_new.append(p_o_prime * summation)

        # normalize
        total = sum(b_new)
        return [x / total for x in b_new]

    def generate_reachable_belief_points(self, belief, max_belief_points):
        m = self.model
        ntrials = 10
        n_tentatives = 0
        beliefs=[]
        beliefs.append(belief.copy())
        bel = belief.copy()        
        while len(beliefs)<max_belief_points and n_tentatives < ntrials*100:
            for n in range(ntrials):
               si = rand_choice(m.states)
               ai = rand_choice(m.get_legal_actions(si))
               #print(si,ai)
               sj, oj, r, cost = m.simulate_action(si, ai, debug=False)
               new_bel = self.update_belief(bel, ai, oj)
               #print(new_bel)
               if new_bel not in beliefs:
                   beliefs.append(new_bel.copy())
               if len(beliefs) >= max_belief_points:
                   break
               bel = new_bel.copy()
            n_tentatives = n_tentatives + 1
            #print(n_tentatives)
               
        return beliefs

    def saving_policy(self):
        encoder = toJSON("alphavecfile.policy",self.alpha_vecs)
        encoder.saving_belief_points(self.belief_points)
        encoder.write_json()
        
    def charging_policy(self,policy_file):
        with open(policy_file) as f:
            data = json.load(f)
            # print(data['alphavec'])
            self.alpha_vecs = []
            for alpha in data['alphavec']:
                self.alpha_vecs.append(AlphaVector(a=alpha['action'], v=alpha['v']))
            
        
