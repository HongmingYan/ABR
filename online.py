''' Demo SDK for LiveStreaming
    Author Dan Yang
    Time 2018-10-15
    For LiveStreaming Game'''
# import the env from pip
import LiveStreamingEnv.fixed_env as fixed_env
#import fixed_env
import LiveStreamingEnv.load_trace as load_trace
import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf


import random
from Double_DQN import DoubleDQN
# from Dueling_DQN import DuelingDQN
from sklearn import preprocessing

RL = DoubleDQN(4, 
                single_fea = 6,
                multi_fea = 9,
                frame_len = 10,
                learning_rate=0.005,
                reward_decay=0.9,
                e_greedy=0.9,
                replace_target_iter=200,
                memory_size=3000,
                # output_graph=True
                )

# path setting
def test(user_id):
    #TRAIN_TRACES = '/home/game/test_sim_traces/'   #train trace path setting,
    #video_size_file = '/home/game/video_size_'      #video trace path setting,
    #LogFile_Path = "/home/game/log/"                #log file trace path setting,
    
    TRAIN_TRACES = './network_trace/'   #train trace path setting,
    video_size_file = './video_trace/AsianCup_China_Uzbekistan/frame_trace_'      #video trace path setting,
    LogFile_Path = "./log/"                #log file trace path setting,
    # Debug Mode: if True, You can see the debug info in the logfile
    #             if False, no log ,but the training speed is high
    DEBUG = False
    DRAW = False
    # load the trace
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRAIN_TRACES)
    #random_seed 
    random_seed = 2
    count = 0
    video_count = 0
    FPS = 25
    frame_time_len = 0.04
    reward_all_sum = 0
    #init 
    #setting one:
    #     1,all_cooked_time : timestamp
    #     2,all_cooked_bw   : throughput
    #     3,all_cooked_rtt  : rtt
    #     4,agent_id        : random_seed
    #     5,logfile_path    : logfile_path
    #     6,VIDEO_SIZE_FILE : Video Size File Path
    #     7,Debug Setting   : Debug
    net_env = fixed_env.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw,
                                  random_seed=random_seed,
                                  logfile_path=LogFile_Path,
                                  VIDEO_SIZE_FILE=video_size_file,
                                  Debug = DEBUG)
    

    BIT_RATE      = [500.0,850.0,1200.0,1850.0] # kpbs
    TARGET_BUFFER = [2.0,3.0]   # seconds
    # ABR setting
    RESEVOIR = 0.5
    CUSHION  = 2

    cnt = 0
    # defalut setting
    last_bit_rate = 0
    bit_rate = 0
    target_buffer = 0

    # QOE setting
    reward_frame = 0
    reward_all = 0
    SMOOTH_PENALTY= 0.02
    REBUF_PENALTY = 1.5
    LANTENCY_PENALTY = 0.005
    # past_info setting
    past_frame_num  = 7500
    S_time_interval = [0] * past_frame_num
    S_send_data_size = [0] * past_frame_num
    S_chunk_len = [0] * past_frame_num
    S_rebuf = [0] * past_frame_num
    S_buffer_size = [0] * past_frame_num
    S_end_delay = [0] * past_frame_num
    S_chunk_size = [0] * past_frame_num
    S_play_time_len = [0] * past_frame_num
    S_decision_flag = [0] * past_frame_num
    S_buffer_flag = [0] * past_frame_num
    S_cdn_flag = [0] * past_frame_num
    
    S_bitrate = [0,0]
    # params setting
    
    # plot info
    idx = 0
    id_list = []
    bit_rate_record = []
    buffer_record = []
    throughput_record = []
    # plot the real time image
    if DRAW:
        fig = plt.figure()
        plt.ion()
        plt.xlabel("time")
        plt.axis('off')
    while True:
        
        reward_frame = 0
        # input the train steps
        #if cnt > 5000:
            #plt.ioff()
        #    break
        #actions bit_rate  target_buffer
        # every steps to call the environment
        # time           : physical time 
        # time_interval  : time duration in this step
        # send_data_size : download frame data size in this step
        # chunk_len      : frame time len
        # rebuf          : rebuf time in this step          
        # buffer_size    : current client buffer_size in this step          
        # rtt            : current buffer  in this step          
        # play_time_len  : played time len  in this step          
        # end_delay      : end to end latency which means the (upload end timestamp - play end timestamp)
        # decision_flag  : Only in decision_flag is True ,you can choose the new actions, other time can't Becasuse the Gop is consist by the I frame and P frame. Only in I frame you can skip your frame
        # buffer_flag    : If the True which means the video is rebuffing , client buffer is rebuffing, no play the video
        # cdn_flag       : If the True cdn has no frame to get 
        # end_of_video   : If the True ,which means the video is over.
        time,time_interval, send_data_size, chunk_len,\
               rebuf, buffer_size, play_time_len,end_delay,\
                cdn_newest_id, download_id, cdn_has_frame, decision_flag,\
                buffer_flag, cdn_flag, end_of_video = net_env.get_video_frame(bit_rate,target_buffer)

        # S_info is sequential order
        S_time_interval.pop(0)
        S_send_data_size.pop(0)
        S_chunk_len.pop(0)
        S_buffer_size.pop(0)
        S_rebuf.pop(0)
        S_end_delay.pop(0)
        S_play_time_len.pop(0)
        S_decision_flag.pop(0)
        S_buffer_flag.pop(0)
        S_cdn_flag.pop(0)
        

        
        
        S_time_interval.append(time_interval)
        S_send_data_size.append(send_data_size)
        S_chunk_len.append(chunk_len)
        S_buffer_size.append(buffer_size)
        S_rebuf.append(rebuf)
        S_end_delay.append(end_delay)
        S_play_time_len.append(play_time_len)
        S_decision_flag.append(decision_flag)
        S_buffer_flag.append(buffer_flag)
        S_cdn_flag.append(cdn_flag)
        
        
        '''
        if time_interval != 0:
            # plot bit_rate 
            id_list.append(idx)
            idx += time_interval
            bit_rate_record.append(BIT_RATE[bit_rate])
            # plot buffer 
            buffer_record.append(buffer_size)
            # plot throughput 
            trace_idx = net_env.get_trace_id()
            #print(trace_idx, idx,len(all_cooked_bw[trace_idx]))
            throughput_record.append(all_cooked_bw[trace_idx][int(idx/0.5)] % 2940 )
        '''
            
        # QOE setting 
        if not cdn_flag:
            reward_frame = frame_time_len * float(BIT_RATE[bit_rate]) / 1000  - REBUF_PENALTY * rebuf - LANTENCY_PENALTY  * end_delay
        else:
            reward_frame = -(REBUF_PENALTY * rebuf)
        if decision_flag or end_of_video:
            cnt+=1
            # reward formate = play_time * BIT_RATE - 4.3 * rebuf - 1.2 * end_delay
            reward_frame += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
            # last_bit_rate
            last_bit_rate = bit_rate
            
            
            # draw setting
            if DRAW:
                ax = fig.add_subplot(311)
                plt.ylabel("BIT_RATE")
                plt.ylim(300,2000)
                plt.plot(id_list,bit_rate_record,'-r')
            
                ax = fig.add_subplot(312)
                plt.ylabel("Buffer_size")
                plt.ylim(0,7)
                plt.plot(id_list,buffer_record,'-b')

                ax = fig.add_subplot(313)
                plt.ylabel("throughput")
                plt.ylim(0,2500)
                plt.plot(id_list,throughput_record,'-g')

                plt.draw()
                plt.pause(0.01)
                
            # -------------------------------------------Your Althgrithom ------------------------------------------- 
            # which part is the althgrothm part ,the buffer based , 
            # if the buffer is enough ,choose the high quality
            # if the buffer is danger, choose the low  quality
            # if there is no rebuf ,choose the low target_buffer
            if 0:
                #50.26
                if buffer_size < RESEVOIR:
                    bit_rate = 0
             
                elif buffer_size >= RESEVOIR + CUSHION:
                    bit_rate = 2
               
                elif buffer_size >= CUSHION + CUSHION:
                    bit_rate = 3
                
                else: 
                   bit_rate = 2
                
            
            else:
            
                '''
                time
                end_of_video
                cdn_newest_id
                download_id
                cdn_has_frame
                S_time_interval
                S_send_data_size
                S_chunk_len
                S_buffer_size
                S_rebuf
                S_end_delay
                S_play_time_len
                S_decision_flag
                S_buffer_flag
                S_cdn_flag
                '''
                change=0
                last = S_bitrate[0]
                for each in S_bitrate:
                    if each != last:
                        last = each
                        change+=1
                  
                # 前一段时间gop的平均状况，用来判断处于一个相对高中低的网络，还有网络的变化幅度，再加上前短时间的状况用来预测接下来状况
 
                sqe_fea = 9
                single_fea = 5
                frame_len = 10
                observation_ = np.array([ S_time_interval[-frame_len:]+ S_send_data_size[-frame_len:]+ S_chunk_len[-frame_len:]+ S_buffer_size[-frame_len:]+
                                        S_rebuf[-frame_len:]+S_end_delay[-frame_len:]+ S_play_time_len[-frame_len:]+ S_buffer_flag[-frame_len:]+ S_cdn_flag[-frame_len:]+ 
                                        [time, end_of_video, cdn_newest_id - download_id, download_id, len(cdn_has_frame[0]), change]]).flatten()
                observation_ = preprocessing.scale(observation_)
                if cnt > 1:
                    
                    RL.store_transition(observation, action_, reward_all, observation_)
                
                reward_all_ = reward_all
                
                if RL.memory_full:
                    RL.learn()

                
                if user_id < 1:
                    if buffer_size < RESEVOIR:
                        bit_rate = 0
             
                    elif buffer_size >= RESEVOIR + CUSHION:
                        bit_rate = 2
                   
                    elif buffer_size >= CUSHION + CUSHION:
                        bit_rate = 3
                    
                    else: 
                       bit_rate = 2
                    action_ = bit_rate
                else:
                
                    action_ = RL.choose_action(observation_)
                    bit_rate = action_
                #bit_rate = int(action_ % len(BIT_RATE))
                #target_buffer = int(action_ / len(BIT_RATE))
                S_bitrate.append(bit_rate)
                
                observation = observation_
            #bit_rate , target_buffer = abr.run(time,S_time_interval,S_send_data_size,S_chunk_len,S_rebuf,S_buffer_size, S_play_time_len,S_end_delay,S_decision_flag,S_buffer_flag,S_cdn_flag, end_of_video, cdn_newest_id, download_id,cdn_has_frame)
            # ------------------------------------------- End  ------------------------------------------- 
            
        if end_of_video:
            print("video count", video_count, reward_all)
            reward_all_sum += reward_all / 1000
            video_count += 1
            if video_count >= len(all_file_names):
                    break
            cnt = 0
            last_bit_rate = 0
            reward_all = 0
            bit_rate = 0
            target_buffer = 0

            S_time_interval = [0] * past_frame_num
            S_send_data_size = [0] * past_frame_num
            S_chunk_len = [0] * past_frame_num
            S_rebuf = [0] * past_frame_num
            S_buffer_size = [0] * past_frame_num
            S_end_delay = [0] * past_frame_num
            S_chunk_size = [0] * past_frame_num
            S_play_time_len = [0] * past_frame_num
            S_decision_flag = [0] * past_frame_num
            S_buffer_flag = [0] * past_frame_num
            S_cdn_flag = [0] * past_frame_num
            
        reward_all += reward_frame

    return reward_all_sum

for i in range(5):
    a = test(i)
    print(a)
