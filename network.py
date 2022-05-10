import numpy as np
import networkx as nx
from itertools import combinations


class Network(nx.Graph):
    """A class for P2P networks under simulation
    """

    def __init__(self, graph):
        """Create an instance of class Network

        Args:
            graph (nx.Graph): the backbone graph topology
        """
        self.__dict__.update(graph.__dict__)
        self.S = self.nodes  # Set of sources
        self.T = self.nodes  # Set of destinations
        self.adv_peer = []

    def get_density(self):
        return self.number_of_edges() * 2 / self.number_of_nodes() / (self.number_of_nodes()-1)

    def get_avg_deg(self):
        return self.number_of_edges() * 2 / self.number_of_nodes()

    def get_diameter(self):
        return nx.diameter(self)

    def get_avg_dist(self):
        return nx.average_shortest_path_length(self)

    def set_dest(self, miner):
        """Reset set of destinations given the proportion of them

        Args:
            miner (float): proportion of miners in [0, 1]
        """
        self.T = np.random.choice(self.nodes, size=max(1, int(self.number_of_nodes() * miner)), replace=False)

    def compute_advantage(self, adv_peer=None, tau=0):
        """Compute the advantage metric given a chosen set of peers of the adversarial agent

        Args:
            adv_peer (List[int], optional): Set of agent's peers. Defaults to None.
            tau (int, optional): The distance penalty of the agent in placing shortcuts between pairs of nodes. Defaults to 0.

        Returns:
            float: advantage metric
        """
        if adv_peer is None:
            adv_peer = self.adv_peer

        adv_peer = list(adv_peer)

        advantage = 0
        short_dist = nx.multi_source_dijkstra_path_length(self, adv_peer)

        for s in self.S:
            direct_dist = nx.single_source_dijkstra_path_length(self, s)

            for t in self.T:
                try:
                    direct = direct_dist[t]
                    shortcut = tau + short_dist[s] + short_dist[t]
                except:
                    print('Graph connectivity: ' + str(nx.is_connected(self)))
                    raise

                if direct > shortcut:
                    advantage += 1
                elif direct == shortcut:
                    advantage += 1/2

        return advantage

    # ------------------------------------------- Simulations on the network -------------------------------------------

    def choose_peer_greedy(self, victims, tau=0, bar=None):
        """Simulate the greedy algorithm and get the advantage

        Args:
            victims (List[int]): a list of the agent's peer budgets
            tau (int, optional): the distance penalty of the agent in placing shortcuts between pairs of nodes. Defaults to 0.
            bar (tqdm, optional): shows progress bar if fed not to be None. Defaults to None.

        Returns:
            List[float]: list of resulting advantages for the peer budgets
            List[int]: resulting peer set for debugging 
        """
        advantage, advantages = 0, []
        self.adv_peer = []

        for u, v in combinations(self.nodes, 2):
            adv = self.compute_advantage([u, v], tau)
            if adv > advantage:
                advantage = adv
                self.adv_peer = [u, v]

            if bar is not None:
                bar.update()

        if 2 in victims:
            advantages.append(advantage)

        for i in range(3, max(victims)+1):
            advantage = 0
            best_choice = None
            for u in self.nodes:
                if u in self.adv_peer:
                    continue

                adv = self.compute_advantage(self.adv_peer + [u], tau)
                if adv > advantage:
                    advantage, best_choice = adv, u

                if bar is not None:
                    bar.update()

            self.adv_peer.append(best_choice)
            if i in victims:
                advantages.append(advantage)

        return advantages, sorted(self.adv_peer)[::-1]

    def choose_peer_random(self, victims, tau=0, bar=None):
        """Simulate the random baseline and get the advantage

        Args:
            victims (List[int]): a list of the agent's peer budgets
            tau (int, optional): the distance penalty of the agent in placing shortcuts between pairs of nodes. Defaults to 0.
            bar (tqdm, optional): shows progress bar if fed not to be None. Defaults to None.

        Returns:
            List[float]: list of resulting advantages for the peer budgets
        """
        self.adv_peer = np.random.choice(self.nodes, size=max(victims), replace=False)

        result = []
        for p in victims:
            peers = np.random.choice(self.adv_peer, size=p, replace=False)
            result.append(self.compute_advantage(adv_peer=peers, tau=tau))
            if bar is not None:
                bar.update()

        return result

    def monte_carlo(self, victims, repeat, tau=0, bar=None):
        """Simulate the Monte-Carlo method and get the advantage

        Args:
            victims (List[int]): a list of the agent's peer budgets
            repeat (int): number of Monte-Carlo trials to take maximum from
            tau (int, optional): the distance penalty of the agent in placing shortcuts between pairs of nodes. Defaults to 0.
            bar (tqdm, optional): shows progress bar if fed not to be None. Defaults to None.

        Returns:
            List[float]: list of resulting advantages for the peer budgets
        """
        self.adv_peer = np.random.choice(self.nodes, size=max(victims), replace=False)

        result = []
        for p in victims:
            max_adv = 0
            for _ in range(repeat):
                peers = np.random.choice(self.adv_peer, size=p, replace=False)
                max_adv = max(max_adv, self.compute_advantage(adv_peer=peers, tau=tau))
                if bar is not None:
                    bar.update()

            result.append(max_adv)

        return result

    def enumerate_peer(self, maxvic, tau=0, bar=None):
        """Simulate the brute force method and get the advantage

        Args:
            maxvic (int): maximum peer budget; advantage will be computed over all budgets ranging from 3 to maxvic 
            tau (int, optional): the distance penalty of the agent in placing shortcuts between pairs of nodes. Defaults to 0.
            bar (tqdm, optional): shows progress bar if fed not to be None. Defaults to None.

        Returns:
            List[float]: list of resulting advantages for the peer budgets
        """
        max_advantages = []
        for vic in range(3, maxvic+1):
            max_advantage = 0
            for possibility in combinations(self.nodes, vic):
                max_advantage = max(max_advantage, self.compute_advantage(list(possibility), tau))
                if bar is not None:
                    bar.update()

            max_advantages.append(max_advantage)

        return max_advantages

    def perigee(self, victims, tau=0, ratio_replace=.1, epoch=[100], seed=None, bar=None):
        """Simulate Peri and get the advantage

        Args:
            victims (List[int]): a list of the agent's peer budgets
            tau (int, optional): the distance penalty of the agent in placing shortcuts between pairs of nodes. Defaults to 0.
            ratio_replace (float, optional): ratio of peers to replace. Defaults to .1.
            epoch (List[int], optional): numbers of epochs to simulate; advantages will be output at each number of epoch. Defaults to [100].
            seed (int|None, optional): seed for randomness. Defaults to None.
            bar (tqdm, optional): shows progress bar if fed not to be None. Defaults to None.

        Returns:
            List[float]: list of resulting advantages for the peer budgets
            List[int]: resulting peer set for debugging 
        """
        np.random.seed(seed)

        dists = dict(nx.all_pairs_dijkstra_path_length(self))
        sum_dists_source = np.array([sum(value for key, value in dists[i].items() if key in self.S) for i in dists])

        adv_peer = None
        advantages = []

        for vic in victims:
            advantage_vic = []
            # Number of dummy peers that are replaced and refilled
            dummy = int(np.ceil(vic * ratio_replace / (1 - ratio_replace)))
            num_replace, num_s_keep = dummy, vic

            # Init a random set of peers
            replacement = np.array([], dtype=int)
            adv_peer = np.random.choice(self.nodes, size=min(vic + dummy, self.number_of_nodes()), replace=False)

            for i in range(max(epoch)):
                adv_peer = np.append(adv_peer, replacement)
                # Keep the peers with aggregate shortest distances
                idx_s_rank = np.argsort(sum_dists_source[adv_peer])
                idx_keep = idx_s_rank[:num_s_keep]

                # Sample the peers from the remaining
                available = list(set(self.nodes).difference(adv_peer[idx_keep]))
                replacement = np.random.choice(available, size=num_replace, replace=False)

                adv_peer = adv_peer[idx_keep]
                if i+1 in epoch:
                    advantage_vic.append(self.compute_advantage(adv_peer, tau))

                if bar is not None:
                    bar.update()

            advantages.append(advantage_vic)

        return advantages, sorted(adv_peer)[::-1]
