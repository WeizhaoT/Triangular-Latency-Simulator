import math

import argparse
import multiprocessing as mp
from pathlib import Path
from tqdm_multiprocess import TqdmMultiProcessPool

from utils import *
from network import *
from typing import List
from configuration import Config


NCPUCORES = mp.cpu_count()
KEYS = ['ra', 'ga', 'pa', 'fa', 'vics', 'fvic', 'tau', 'ratio', 'epoch', 'gp', 'pp']


# ------------------------------ multi-processed methods for simulation of each algorithm ------------------------------
def mpRandom(net, vics, tau, tqdm_func, bar):
    assert isinstance(net, Network)
    np.random.seed()
    return net.choose_peer_random(vics, tau, bar)


def mpGreedy(net, vics, tau, tqdm_func, bar):
    assert isinstance(net, Network)
    np.random.seed()
    return net.choose_peer_greedy(vics, tau, bar)


def mpPerigee(net, vics, tau, ratio, epochs, seed, tqdm_func, bar):
    assert isinstance(net, Network)
    return net.perigee(vics, tau, ratio, epochs, seed, bar)


def mpBruteForce(net, maxvic, tau, tqdm_func, bar):
    assert isinstance(net, Network)
    return net.enumerate_peer(maxvic, tau, bar)


def mpMonteCarlo(net, vics, repeat, tau, tqdm_func, bar):
    assert isinstance(net, Network)
    np.random.seed()
    return net.monte_carlo(vics, repeat, tau, bar)


def error_callback(result):
    print("Error!")


def done_callback(result):
    pass


# ----------------------------------------------------------------------------------------------------------------------

def getNets(config):
    """Load Network instances by config 

    Args:
        config (Config): config object including the path to the nets

    Returns:
        List[Network]: a list of Network instances
    """
    print('Building Graphs... ', end='')
    time_start = time.time()

    with open(config['Graph']['graph_path'], 'rb') as f:
        nets = pickle.load(f)

    get_timer(time_start)
    return nets


def frontrun(exp, nets: List[Network], update: str = ''):
    """Simulate the algorithms

    Args:
        exp (dict): configs for the experiments
        nets (List[Network]): loaded nets
        update (str, optional): if set to the name of an algorithm (say, Peri), only the algorithm (Peri) is simulated; 
                                otherwise, all algs will be simulated. Defaults to ''.

    Returns:
        a tuple consisting of advantages of all the algorithms + peer choices of Peri and Greedy
        each advantage is a list with the same length as the numbers of peer budget (exp['nvics'])
    """
    nvics, tau, parallel, num_nodes = exp['victims'], exp['tau'], exp['parallel'], nets[0].number_of_nodes()

    rand_adv, greedy_adv, full_adv, peer_adv, monte_adv, g_peers, p_peers = None, None, None, None, None, None, None

    pool = TqdmMultiProcessPool(NCPUCORES)

    # -------- Perigee --------

    if update == '' or update.lower() == 'perigee':
        bar = tqdm(total=len(nets) * len(nvics) * max(exp['epoch']),
                   desc='Perigee', dynamic_ncols=True, ascii=FLAG_ASCII)

        inputs = [(mpPerigee, (net, nvics, tau, exp['ratio'], exp['epoch'], exp['seed']))
                  for net in nets]
        results = pool.map(bar, inputs, error_callback, done_callback)

        peer_adv, p_peers = zip(*list(results))
        peer_adv = np.reshape(peer_adv, (len(nets), len(nvics), -1))

        bar.close()

    # -------- Random --------

    if update == '' or update.lower() == 'random':
        bar = tqdm(total=len(nets) * parallel * len(nvics), desc='Random', dynamic_ncols=True, ascii=FLAG_ASCII)
        inputs = [(mpRandom, (net, nvics, tau)) for net in nets for _ in range(parallel)]

        results = pool.map(bar, inputs, error_callback, done_callback)

        rand_adv = np.reshape(list(results), (len(nets), -1, len(nvics)))

        bar.close()

    # -------- Greedy --------

    if update == '' or update.lower() == 'greedy':
        bar = tqdm(total=len(nets) * (math.comb(num_nodes, 2) + (max(nvics) - 2) *
                                      (2 * num_nodes - max(nvics) - 1) // 2), desc='Greedy', dynamic_ncols=True, ascii=FLAG_ASCII)

        inputs = [(mpGreedy, (net, nvics, tau)) for net in nets]
        results = pool.map(bar, inputs, error_callback, done_callback)

        greedy_adv, g_peers = zip(*results)
        greedy_adv = np.reshape(greedy_adv, (len(nets), -1))

        bar.close()

    # -------- Brute Force --------

    if (update == '' or update.lower() == 'force') and exp['max_enum'] >= 3:
        numiter = sum(math.comb(nets[0].number_of_nodes(), n) for n in range(3, exp['max_enum']+1))
        bar = tqdm(total=len(nets) * numiter, desc='BruteForce', dynamic_ncols=True, ascii=FLAG_ASCII)
        inputs = [(mpBruteForce, (net, exp['max_enum'], tau)) for net in nets]

        results = pool.map(bar, inputs, error_callback, done_callback)

        full_adv = np.reshape(list(results), (len(nets), -1))
        full_adv = np.concatenate([greedy_adv[:, :1], full_adv], axis=1)
        bar.close()

    # -------- Monte-Carlo --------

    if (update == '' or update.lower() == 'monte'):
        bar = tqdm(total=len(nets) * exp['repeat'] * len(nvics),
                   desc='Monte-Carlo', dynamic_ncols=True, ascii=FLAG_ASCII)
        inputs = [(mpMonteCarlo, (net, nvics, exp['repeat'], tau)) for net in nets]
        results = pool.map(bar, inputs, error_callback, done_callback)
        monte_adv = np.reshape(list(results), (len(nets), -1))
        bar.close()

    return rand_adv, greedy_adv, full_adv, peer_adv, monte_adv, g_peers, p_peers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Order Fairness Simulator')
    parser_group = parser.add_mutually_exclusive_group(required=True)

    parser_group.add_argument('-f', '--config-file', type=str,
                              help='path to config file (to start a new simulation)')
    parser.add_argument('-r', '--reload', action='store_true',
                        help='set this flag if you plan to skip simulation and plot the curves with old data')
    parser.add_argument('-u', '--update', type=str, default='',
                        choices=['', 'perigee', 'random', 'greedy', 'monte', 'force'], required=False,
                        help='if not set, all algorithms will be simulated; if set to an algorithm, '
                        'the algorithm will be re-simulated, while the results of the other algorithms will be read from old data')
    parser_args = parser.parse_args()

    update = parser_args.update
    config = Config(parser_args.config_file)

    if config['Experiment']['seed'].lower() == 'none':
        config['Experiment']['seed'] = None
    else:
        config['Experiment']['seed'] = int(config['Experiment']['seed'])

    nets = getNets(config)

    # Prepare direcs to save data and images
    work_dir = join(dirname(abspath(parser_args.config_file)), remove_file_ext(basename(parser_args.config_file)))
    img_dir = join(str(Path(dirname(abspath(parser_args.config_file))).parent), 'img')

    assure_dir(img_dir)
    assure_dir(work_dir)

    print('Configuration:')
    config.showOptions()

    vics = np.array(config['Experiment']['victims'])
    fvic = config['Experiment']['max_enum']
    tau = config['Experiment']['tau']
    ratio = config['Experiment']['ratio']
    epoch = np.array(config['Experiment']['epoch'])
    name = remove_file_ext(basename(config['Graph']['graph_path']))

    time_start_global = time.time()

    if parser_args.reload:
        with open(join(work_dir, 'result.pkl'), 'rb') as f:
            data = pickle.load(f)
    else:
        data = {}

        ra, ga, fa, pa, ma, gp, pp = frontrun(config['Experiment'], nets, update)

        if update != '':
            with open(join(work_dir, 'result.pkl'), 'rb') as f:
                data = pickle.load(f)

            if update == 'perigee':
                data['pa'] = pa
                data['pp'] = pp
            elif update == 'random':
                data['ra'] = ra
            elif update == 'greedy':
                data['ga'] = ga
                data['gp'] = gp
            elif update == 'monte':
                data['ma'] = ma
        else:
            for key in KEYS:
                if eval(key) is not None:
                    data[key] = eval(key)

        with open(join(work_dir, 'result.pkl'), 'wb') as f:
            pickle.dump(data, f)

    print('Plotting curves...', end='')
    fig = plot_advantage(data, name)
    fig.savefig(join(img_dir, name + '.jpg'))
    plt.close()

    time_end_global = time.time()
    print('\nAll simulations completed. ', end='')
    get_timer(time_start_global)
