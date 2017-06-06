RLAgents
========

Reinforcement Learning Agents


Installation
------------

`pip install rlagents`


Todo
----

- [x] RandomAgent
- [x] MonteCarlo
- [x] Q-Learning


Usage
-----

The general way this is used by me is:

```python
import gym
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from rlagents import QLAgent, benchmark, gymwrapper
```

We make environments and agents

```python
env_list = [gymwrapper(gym.make('CartPole-v0'))]
agent_list = [QLAgent()]
```


We run the experiments

```python
paths = benchmark(agent_list, env_list,
                  n_episodes=500,
                  max_steps_per_episode=200,
                  n_trials=200)
```

The data has been written to Disk. We load it for analysis

```python
paths = ['DataDir/'+i for i in os.listdir('DataDir/')]
data = []
for p in tqdm(paths, leave=False):
	with open(p, 'r') as fl:
		da = ujson.load(fl)

		x = [(None, 'Monte' in d[1], None, int(d[3]), float(d[4]), int(d[5]))
			 for d in da]
		data.extend(x)
```

Result Plotting with seaborn.

```python
data = pd.DataFrame(data, columns=['runid', 'agent', 'world', 'ep', 'rew', 'trial'])
data.agent = 'Q Learning'
sns.tsplot(data, time='ep', value='rew', unit='trial', condition='agent', ci=95)
```
