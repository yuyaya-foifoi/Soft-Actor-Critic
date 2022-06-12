https://user-images.githubusercontent.com/40622501/172090661-0494efe5-babd-47c5-a2a9-6a07d375ff55.mp4




## training

run  `./notebooks/soft-actor-critic.ipynb` on google colab

## setting
`config/config.py`

```
"Base": {
        "Env": "FetchPickAndPlaceDense-v1", # env name
        "Agent": "SAC", # algo name (SAC only so far)
        "Run_ID": {}, # str
        ...
    },
"Transfer": {
    "is_Transfer": False, # bool
    "Weight_path": ${path/to/pkl},
},
```

## code formatter
```
make style
```

## setup gym environment
```
make setup_gym
```

## setup gym environment
```
make setup_brax
```

## reference
```
- “ゼロから作るDeep Learning ❹,” Apr. 06, 2022. https://www.oreilly.co.jp/books/9784873119755/ (accessed Jun. 06, 2022).
- “Soft-Actor-Critic (SAC) ①Soft-Q学習からSACへ - どこから見てもメンダコ,” どこから見てもメンダコ, Dec. 20, 2020. https://horomary.hatenablog.com/entry/2020/12/20/115439 (accessed Jun. 06, 2022).
- “Advanced Reinforcement Learning in Python: from DQN to SAC,” _Udemy_. https://www.udemy.com/course/advanced-reinforcement/ (accessed Jun. 06, 2022).
- “Reinforcement Learning,” Reinforcement Learning. https://rl-book.com/ (accessed Jun. 06, 2022).
```


