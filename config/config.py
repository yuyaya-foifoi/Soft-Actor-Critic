config = {
    "Base": {
        "Env": "FetchPickAndPlaceDense-v1",
        "Agent": "SAC",
        "Run_ID": "6_9_1403",
        "is_Transfer": False,
    },
    "Train": {"Epoch": 1000, "Env_preprocess": "dict", "is_Record_log": True},
    "Video": {
        "Interval": 100,
        "is_Record": True,
    },
}
