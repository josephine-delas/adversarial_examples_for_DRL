import os


class base_agent:
    '''
    Parent class - creates model directories
    '''

    def __init__(self, args, name, victim):
        '''
        args -> victim_agent_mode, attacker_agent_mode, env_name, game_mode
        '''

        if victim:
            agent_mode = args.victim_agent_mode
        else:
            agent_mode = args.attacker_agent_mode

        model_name = agent_mode #dqn, ppo, etc.
        
        if args.game_mode == "train":
            self.model_path = "/home/jdelas/projects/def-fcuppens/jdelas/models/" + args.env_name + "/" + model_name + "/" + args.game_mode + "/"
        else:
            self.model_path ="/home/jdelas/projects/def-fcuppens/jdelas/models/" + args.env_name + "/" + model_name + "/" + args.game_mode + "/" + name  
            self.defense_path = "/home/jdelas/projects/def-fcuppens/jdelas/models/" + args.env_name + "/" + model_name + "/" + args.game_mode + "/" 

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    '''
    def save_run(self, score, step, run):
        #self.logger.add_score(score)
        pass

    def select_action(self, obs, explore_eps=0.5, rnn_hxs=None, masks=None, deterministic=False):
        pass

    def remember(self,obs, action, reward, next_obs, done):
        pass

    def update_agent(self, total_step, rollouts=None, advmask=None):
        pass
    '''