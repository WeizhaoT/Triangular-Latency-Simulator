from utils import *
from network import Network
from argparse import ArgumentParser, Namespace

import multiprocessing as mp


NCPUCORES = mp.cpu_count()


def generate_name(args: Namespace):
    """Name the topology given hyperparameters

    Args:
        args (Namespace): hyperparameters of the topology

    Raises:
        NotImplementedError: when topology cannot be identified

    Returns:
        str: file name of the networks
    """
    if args.top == 'er':
        name = 'er_{}_{}_{:.4f}'.format(args.count, args.size, args.density)
    elif args.top == 'sf':
        name = 'sf_{}_{}_{}'.format(args.count, args.size, args.neighbors)
    elif args.top == 'sw':
        name = 'sw_{}_{}_{}_{:.2f}'.format(
            args.count, args.size, args.neighbors, args.rewire)
    elif args.top == 'tree':
        name = 'tree_{}_{}'.format(args.count, args.size)
    elif args.top == 'reg':
        name = 'reg_{}_{}_{}'.format(args.count, args.size, args.neighbors)
    else:
        raise NotImplementedError

    return name


def generate_graph(args: Namespace) -> Network:
    """Synthesize a graph given hyperparameters

    Args:
        args (Namespace): hyperparameters of the topology

    Raises:
        NotImplementedError: when topology cannot be identified

    Returns:
        Network: a resulting Network instance
    """
    np.random.seed()

    if args.top == 'er':
        net = nx.erdos_renyi_graph(args.size, args.density, seed=np.random)
    elif args.top == 'sf':
        net = nx.barabasi_albert_graph(
            args.size, args.neighbors, seed=np.random)
    elif args.top == 'sw':
        net = nx.watts_strogatz_graph(
            args.size, args.neighbors, args.rewire, seed=np.random)
    elif args.top == 'tree':
        net = nx.random_tree(args.size, seed=np.random)
    elif args.top == 'reg':
        net = nx.random_regular_graph(args.neighbors, args.size)
    else:
        raise NotImplementedError

    net = Network(net)
    net.set_dest(args.miner)

    return net


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--top', type=str, choices=['er',
                        'sf', 'sw', 'tree', 'reg'], required=True, help='graph topology')
    parser.add_argument('-c', '--count', type=int, default=50, help='number of graph instances')
    parser.add_argument('-s', '--size', type=int, required=True, help='number of nodes')
    parser.add_argument('-d', '--neighbors', type=int,
                        help='number of neighbors (for all topologies except Erdos-Renyi)')
    parser.add_argument('-D', '--density', type=float, help='graph density (for Erdos-Renyi only)')
    parser.add_argument('-w', '--rewire', type=float, help='rewire probability (for small-world only)')
    parser.add_argument('-r', '--miner', type=float, help='proportion of destination nodes to sample')
    parser.add_argument('-p', '--dir', type=str, default='nets/', help='directory to save the pickled nets')

    args = parser.parse_args()

    name = generate_name(args)

    pool = mp.Pool(processes=NCPUCORES)
    bar = tqdm(pool.imap(generate_graph, [args for _ in range(args.count)]),
               total=args.count, desc='Generating graph', ascii=FLAG_ASCII)
    nets = list(bar)
    pool.close()
    pool.join()

    savepath = join(args.dir, name + '.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(nets, f)

    # Show the stats of synthesized topologies
    degs, diameters, distances = [], [], []

    for net in nets:
        assert isinstance(net, Network)
        degs.append(net.get_avg_deg())
        diameters.append(net.get_diameter())
        distances.append(nx.average_shortest_path_length(net))

    print(
        f'Networks saved under {savepath}. Degree = {np.mean(degs):.4f} +/- {np.std(degs):.4f}. '
        f'Diameter = {np.mean(diameters):.2f} +/- {np.std(diameters):.2f} '
        f'Distance = {np.mean(distances):.2f} +/- {np.std(distances):.2f} \n')
