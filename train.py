''' Demo SDK for LiveStreaming
    Author Dan Yang
    Time 2018-10-15
    For LiveStreaming Game'''
# import the env from pip
import LiveStreamingEnv.env as env
import LiveStreamingEnv.load_trace as load_trace
import matplotlib.pyplot as plt
import time
import numpy as np
import random


from PR_DQN import DQNPrioritizedReplay
from sklearn import preprocessing
# path setting
TRAIN_TRACES = './train_sim_traces/'   #train trace path setting,
video_size_file = './video_size_'      #video trace path setting,
LogFile_Path = './Log/'                #log file trace path setting,
# Debug Mode: if True, You can see the debug info in the logfile
#             if False, no log ,but the training speed is high
DEBUG = False
DRAW = False
# load the trace
all_cooked_time, all_cooked_bw, all_cooked_rtt,_ = load_trace.load_trace(TRAIN_TRACES)
#random_seed 
random_seed = 2


BIT_RATE      = [500,800 ] # kpbs
TARGET_BUFFER = [2,3]   # seconds

MAX_EPISODES = 30
RL = DQNPrioritizedReplay(len(BIT_RATE)*len(TARGET_BUFFER), 10,
                  learning_rate=0.01,
                  reward_decay=0.9,
                  e_greedy=0.9,
                  replace_target_iter=200,
                  memory_size=2000,
                  # output_graph=True
                  )
for i in range(MAX_EPISODES):
    S_time_interval = []
    S_send_data_size = []
    S_chunk_len = []
    S_rebuf = []
    S_buffer_size = []
    S_end_delay = []
    S_chunk_size = []
    S_rtt = []
    S_play_time = []
    RESEVOIR = 0.5
    CUSHION  = 2
    last_bit_rate = 0
    reward_all = 0
    
    bit_rate = round(random.random())
    target_buffer = round(random.random())

    #init the environment
    #setting one:
    #     1,all_cooked_time : timestamp
    #     2,all_cooked_bw   : throughput
    #     3,all_cooked_rtt  : rtt
    #     4,agent_id        : random_seed
    #     5,logfile_path    : logfile_path
    #     6,VIDEO_SIZE_FILE : Video Size File Path
    #     7,Debug Setting   : Debug
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              all_cooked_rtt=all_cooked_rtt,
                              random_seed=random_seed,
                              logfile_path=LogFile_Path,
                              VIDEO_SIZE_FILE=video_size_file,
                              Debug = DEBUG)
    cnt = 0

    end_of_video = False
    while True:
        # input the train steps
        if end_of_video:
            plt.ioff()
            break

        time, time_interval, send_data_size, chunk_len, rebuf, buffer_size, rtt, play_time_len,end_delay, decision_flag, buffer_flag,cdn_flag, end_of_video = net_env.get_video_frame(bit_rate,TARGET_BUFFER[target_buffer])
        
        
        if decision_flag :
            # reward formate = play_time * BIT_RATE - 4.3 * rebuf - 1.2 * end_delay
            reward =  sum(S_play_time) *  BIT_RATE[bit_rate] - 0.8 *  sum(S_rebuf) -  0.2 * (end_delay - 3)  - abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate])
            reward_all += reward
            # last_bit_rate
            last_bit_rate = bit_rate
            
            cnt += 1
            # -------------------------------------------Your Althgrithom-------------------------------------------
            if buffer_flag:
                buffer_flag = 1
            else:
                buffer_flag = 0
            if cdn_flag:
                cdn_flag = 1
            else:
                cdn_flag = 0

            observation_ = np.array([time_interval, send_data_size, chunk_len, rebuf, buffer_size, rtt, play_time_len,end_delay, buffer_flag, cdn_flag])
            observation_ = preprocessing.scale(observation_)

            if cnt > 1:
                RL.store_transition(observation, action_, reward, observation_)
            if RL.memory_full:
                RL.learn()
          


            action_ = RL.choose_action(observation_)

            bit_rate = int(action_ / len(BIT_RATE))
            target_buffer = int(action_ % len(BIT_RATE))
           
            observation = observation_


            # ------------------------------------------- End  -------------------------------------------
            
            
            S_time_interval = []
            S_send_data_size = []
            S_chunk_len = []
            S_rebuf = []
            S_buffer_size = []
            S_end_delay = []
            S_rtt = []
            S_play_time = []
            S_chunk_size = []
    
    
        S_time_interval.append(time_interval)
        S_send_data_size.append(send_data_size)
        S_chunk_len.append(chunk_len)
        S_buffer_size.append(buffer_size)
        S_rebuf.append(rebuf)
        S_end_delay.append(end_delay)
        S_rtt.append(rtt)
        S_play_time.append(play_time_len)
    print('No.%d, reward:%f' % (i,reward_all))

#RL.save()
