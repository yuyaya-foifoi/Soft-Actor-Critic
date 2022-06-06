https://user-images.githubusercontent.com/40622501/172090661-0494efe5-babd-47c5-a2a9-6a07d375ff55.mp4




## training

run  `./notebooks/soft-actor-critic.ipynb` on google colab

## setting
`config/config.yml`

```
Env: 'FetchReachDense-v1' # gym
Agent: 'SAC'
Run_ID: '${}'
Transfer: None # To DO : for transfer learning

Video:
  Interval: 50 # how frequent you want to dump video 
```

## code formatter
```
make style
```

## setup environment
```
make env_setup
```

## reference
```
- “ゼロから作るDeep Learning ❹,” Apr. 06, 2022. https://www.oreilly.co.jp/books/9784873119755/ (accessed Jun. 06, 2022).
- “Soft-Actor-Critic (SAC) ①Soft-Q学習からSACへ - どこから見てもメンダコ,” どこから見てもメンダコ, Dec. 20, 2020. https://horomary.hatenablog.com/entry/2020/12/20/115439 (accessed Jun. 06, 2022).
- “Advanced Reinforcement Learning in Python: from DQN to SAC,” _Udemy_. https://www.udemy.com/course/advanced-reinforcement/ (accessed Jun. 06, 2022).
- P. Winder, _Reinforcement Learning_. O’Reilly Media, Inc.
```


