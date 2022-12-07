import gym 
import ale_py

if __name__=='__main__':
    print(gym.__version__)
    #print(ale_py.__version__)
    print(gym.envs.registry.keys())
    print('ok')
    gym.make('ALE/Pong-v5')
    print('Ca a marche !!')