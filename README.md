# Front-running Advantage Simulator

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)

## About <a id="about"></a>

This project implements a simulator for advantage computation and visualization of algorithms in P2P networks. Our simulator implements the following algorithms:
* <span style="color:lightgray">Example Algorithm (example_alg_code_name): example description</span>
* <a id="brute"></a>**Brute Force** (brute): enumerates all possibilities of peer choices, and takes the maximum over their advantages. 
* **Greedy** (greedy): See our paper 
* <a id="peri"></a>**Peri** (perigee): See our paper
* <a id="random"></a>**Random** (random): takes the advantage of a set of peers sampled uniformly at random
* <a id="monte"></a>**Monte-Carlo** (monte): takes the max advantage over multiple randomly-chosen peer sets

## Getting Started <a id = "getting_started"></a>

This project does not have a system requirement. Please change directory here before running the scripts (same one where README is located).  

### Prerequisites

Python 3.8 or newer with following packages installed:
```
pip install numpy networkx matplotlib tqdm tqdm_multiprocess
```

### A Starting Example of Simulating <a id="starting-example"></a>

Provided with `data/sw300s.json` and `nets/sw_25_300_4_0.25.pkl`, we can run simulations on the small-world topology with 300 nodes. 
```
python frontrun.py -f data/sw300s.json
```

The Advantage-Peer-budget curves will be in the new directory `img/`. 

## Usage <a id="usage"></a>

### Synthesizing Topology
For simulations on a different random topology, use `netgen.py` to synthesize one for yourself. For example, 
```
python netgen.py --top er --count 5 --size 100 --density 0.2 --miner 0.1 --dir net/
```
This will produce a file `net/er_5_100_0.2.pkl`, including 5 random Erdos-Renyi network instances. For more hints on arguments, please run `python netgen.py -h`. We support the following topologies:
* <span style="color:lightgray">Example Topology (example_top_code_name): example description</span>
* **Erdős–Rényi** (er)
* **Random Regular** (reg)
* **Scale-Free** (sf): the Barabási–Albert model
* **Small-World** (sw): the Watts–Strogatz model
* **Random Tree** (tree)

We also support the centralization of a topology by introducing hubs, turning them hub-enriched. For example, the following command adds 7 hub nodes to each instance (5 in total), where each hub connects to 20\% of the nodes (=20) in the original topologies. 
```
python centralizer.py --path net/er_5_100_0.2.pkl --num 7 --ratio 0.2 --dir nets/
```
The resulting list of new network instances will be saved to `nets/er_5_100_0.2_c7_r0.2000.pkl`.

### Simulations
#### Starting a Raw Simulation
Now that the topology is synthesized, we plan for a simulation. First, we create a json file and save it under `data/example.json`.
```
{
    "Graph": {
        "graph_path": "net/er_5_100_0.2.pkl"
    },
    "Experiment": {
        "parallel": 1,
        "victims": [2, 3, 5, 7],
        "tau": 0.0,
        "repeat": 10,
        "ratio": 0.25,
        "epoch": [800],
        "max_enum": 1,
        "seed": "12345"
    }
}
```

Here we describe the entries in the json object:
* `graph_path`: link it to your network instances
* `victims`: a list of peer budgets, on each we have an advantage value output by a given algorithm
* `tau`: a static distance penalty of shortcuts through the adversarial agent (used in advantage computation)
* `parallel`: number of repeated trials for [random algorithm](#random)
* `repeat`: number of repeated trials for [Monte-Carlo algorithm](#monte)
* `ratio`: proportion of peers to replace at every [Peri](#peri) period
* `epoch`: a list of numbers of [Peri](#peri) periods; advantage is computed at every period number 
* `max_enum`: advantage is computed by [brute-force](#brute) at all peer budgets between 3 and `max_enum` 
* `seed`: seed of randomness

Now we can start the simulation by running 
```
python frontrun.py -f data/example.json
```
This may take a while. After it finishes running, the data will be saved under `data/example/result.pkl`, and the image will be saved under `img/er_5_100_0.2.jpg`. 

#### Plot the Data from a Previous Simulation
What if we don't want to simulate everything again when we have the data but lose the plots? Just run 
```
python frontrun.py -f data/example.json -r
```
and the plots will be re-created. 

#### Update the Curve of one Single Algorithm
What if we want to update the settings in the json file and recompute the results of just one algorithm (for example, [Peri](#peri))? If we start everything from scratch, we may spend time unnecessarily on other algorithms with data already saved. For a more time-efficient approach, just run 
```
python frontrun.py -f data/example.json -u perigee
```


## Congratulations! You have learned how to use our simulator!