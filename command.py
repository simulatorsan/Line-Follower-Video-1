invert_waypoints = [True, False, True, False, True, False]
invert_colours = [True, False, False, True, True, False]

pending_rm = ""


for i in range(50, 801, 10):
    i = f"{i:04d}"
    
    for j in range(6):
        
        iw = '--invert_waypoints' if invert_waypoints[j] else ''
        ic = '--invert_colours' if invert_colours[j] else ''
        
        print(f"python video.py --model_path for_video/saved_models/{i}.pth --save_dir for_video/runs/{i} --filename run_{j+1}.npy --save_threshold -100 {iw} {ic} &")
    print("wait")
    # print("sleep 1")
    print(pending_rm)
        
    print("python /home/aritra/Documents/temp/youtube/rl_line_follower/demo/main.py", end=" ")
    for j in range(6):
        print(f"/home/aritra/Documents/temp/Learning_Reinforcement_Learning/line_follower_v0/for_video/runs/{i}/run_{j+1}.npy", end=" ")
    print(f"{i}.mp4", end=" ")
    print(f"/home/aritra/Documents/temp/youtube/rl_line_follower/demo/parts", end=" ")
    print(f'--episode_number "Episode {int(i)}" ', end=" ")
    print(f'--graph_path /home/aritra/Documents/temp/Learning_Reinforcement_Learning/line_follower_v0/for_video/graphs/{i}.png &')

    # print()
    # print(f"rm -rf for_video/runs/{i}")
    pending_rm = f"rm -rf for_video/runs/{i}"
    print()


# python /home/aritra/Documents/temp/youtube/rl_line_follower/demo/main.py \
#     /home/aritra/Documents/temp/Learning_Reinforcement_Learning/line_follower_v0/for_video/runs/0010/run_1.npy \
#     /home/aritra/Documents/temp/Learning_Reinforcement_Learning/line_follower_v0/for_video/runs/0010/run_2.npy \
#     /home/aritra/Documents/temp/Learning_Reinforcement_Learning/line_follower_v0/for_video/runs/0010/run_3.npy \
#     /home/aritra/Documents/temp/Learning_Reinforcement_Learning/line_follower_v0/for_video/runs/0010/run_4.npy \
#     /home/aritra/Documents/temp/Learning_Reinforcement_Learning/line_follower_v0/for_video/runs/0010/run_5.npy \
#     /home/aritra/Documents/temp/Learning_Reinforcement_Learning/line_follower_v0/for_video/runs/0010/run_6.npy \
#     0010.mp4 \
#     /home/aritra/Documents/temp/youtube/rl_line_follower/demo/ \
#     --episode_number "Episode 10" \
#     --graph_path /home/aritra/Documents/temp/Learning_Reinforcement_Learning/line_follower_v0/for_video/graphs/0010.png