config = {
    "Base": {
        "Env": "FetchPickAndPlaceDense-v1",
        "Agent": "SAC",
        "Run_ID": "6_11_1539",
        "Seed": 0,
    },
    "Transfer": {
        "is_Transfer": False,
        "Weight_path": "./logs/SAC/FetchReachDense-v1/averaged_weights.pkl",
    },
    "Train": {
        "epoch": 1000,
        "env_preprocess": "robotics",
        # 'robotics'
        # 'mujoco'
        "is_Record_log": True,
        "replay_size": 100000,
        "update_after": 50000,
        "samples_per_epoch": 1000,
        "explore_epsilon": 0.05,
        "batch_size": 256,
        "actor_lr": 1e-4,
        "critic_lr": 1e-5,
        "gamma": 0.99,
        "polyak_rho": 0.995,
        "alpha": 0.2,
    },
    "Model": {
        "hidden_size": 512,
    },
    "Video": {
        "Interval": 100,
        "Len": 200,
        "is_Record": True,
    },
}


"""
env_name, observation_space, action_space

------robotics------

FetchPickAndPlaceDense-v1 @ 28 / 4
FetchReachDense-v1 @ 13 / 4
FetchSlideDense-v1 @ 28 / 4
FetchPushDense-v1 @ 28 / 4

FetchPickAndPlace-v1 @ 28 / 4
FetchReach-v1 @ 13 / 4
FetchSlide-v1 @ 28 / 4
FetchPush-v1 @ 28 / 4
+++++++++++++++++++
HandManipulateBlockRotateZTouchSensorsDense-v0 @ 160 / 20
HandManipulateBlockRotateParallelTouchSensorsDense-v0 @ 160 / 20
HandManipulateBlockRotateXYZTouchSensorsDense-v0 @ 160 / 20
HandManipulateBlockTouchSensorsDense-v0 @ 160 / 20
HandManipulateEggRotateTouchSensorsDense-v0 @ 160 / 20
HandManipulateEggTouchSensorsDense-v0 @ 160 / 20
HandManipulateBlockRotateZTouchSensorsDense-v1 @ 160 / 20
HandManipulateBlockRotateParallelTouchSensorsDense-v1 @ 160 / 20
HandManipulateBlockRotateXYZTouchSensorsDense-v1 @ 160 / 20
HandManipulateBlockTouchSensorsDense-v1 @ 160 / 20
HandManipulateEggRotateTouchSensorsDense-v1 @ 160 / 20
HandManipulateEggTouchSensorsDense-v1 @ 160 / 20
+++++++++++++++++++
HandManipulateBlockRotateZDense-v0 @ 68 / 20
HandManipulateBlockRotateParallelDense-v0 @ 68 / 20
HandManipulateBlockRotateXYZDense-v0 @ 68 / 20
HandManipulateBlockDense-v0 @ 68 / 20
HandManipulateEggRotateDense-v0 @ 68 / 20
HandManipulateEggDense-v0 @ 68 / 20

------MuJoCo------

Ant-v2 @ 111 / 8
Humanoid-v2 @ 376 / 17
Swimmer-v2 @ 8 / 2
Hopper-v2 @ 11 / 3
Reacher-v2 @ 11 / 2
Pusher-v2 @ 23 / 7
Thrower-v2 @ 23 / 7
Striker-v2 @ 23 / 7
HalfCheetah-v2 @ 17 / 6
HumanoidStandup-v2 @ 376 / 17
InvertedPendulum-v2 @ 4 / 1

"""
