from utils import *
from pathlib import Path
from network import Network
from argparse import ArgumentParser, Namespace

import multiprocessing as mp


NCPUCORES = mp.cpu_count()


def modify_net(G: Network, k, r):
    """Centralize a synthesized Network instance, turning it hub-enriched

    Args:
        G (Network): a Network instance
        k (_type_): number of hubs
        r (_type_): ratio of degree of each hub to the size of the original G

    Returns:
        Network: hub-enriched Network instance
    """
    n = G.number_of_nodes()
    t = len(G.T)

    for i in range(n, n+k):
        G.add_node(i)
        neighbors = np.random.choice(np.arange(i), size=min(max(1, int(n * r)), n), replace=False)

        for u in neighbors:
            G.add_edge(i, u)

    # Reset sources and destinations
    G.S = G.nodes
    G.set_dest(t / n)

    return G


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='path to the original topologies (output of netgen.py)')
    parser.add_argument('-n', '--num', type=int, required=True, help='number of hubs')
    parser.add_argument('-r', '--ratio', type=float, default=.5, required=False,
                        help='ratio of degree of each hub to the size of the original G')
    parser.add_argument('--dir', type=str, default='nets/', help='directory to save the pickled nets')

    args = parser.parse_args()

    with open(args.path, 'rb') as f:
        nets = pickle.load(f)

    inputs = [(net, args.num, args.ratio) for net in nets]

    pool = mp.Pool(processes=NCPUCORES)
    bar = tqdm(pool.starmap(modify_net, inputs),
               total=len(nets), desc='Centralizing graph', ascii=FLAG_ASCII)
    nets = list(bar)
    pool.close()
    pool.join()

    stem = Path(args.path).stem
    savepath = join(args.dir, 'cent', stem + '_c{}_r{:6.4f}.pkl'.format(args.num, args.ratio))
    with open(savepath, 'wb') as f:
        pickle.dump(nets, f)

    print(f'Networks saved to {savepath}.\n')
