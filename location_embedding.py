import scipy.io as sio
import numpy as np
from gensim.models import Word2Vec

EMBEDSIZE = 16

class Graph(object):
    def __init__(self, node_num):
        self._node_num = node_num
        self._edges = np.zeros((self._node_num, self._node_num))

    def add_weight(self, src_node, dst_node, weight=1):

        self._edges[src_node, dst_node] += weight

    def _next_node(self, node, walk, p=2, q=2):
        """ node2vec, return param: p, in-out param: q """

        if np.sum(self._edges[node]) == 0:
            return node
        if len(walk) < 2:
            probs = list(self._edges[node]/np.sum(self._edges[node]))
            probs[node] = probs[node]/10
            probs = probs/np.sum(probs)
            return np.random.choice(np.arange(self._node_num), p=probs)
        else:
            last_node = int(walk[-2])
            probs = self._edges[node]/np.sum(self._edges[node])
            for i in range(self._node_num):
                if i == last_node:
                    probs[i] = 1/p*probs[i]
                elif self._edges[last_node, i]/np.sum(self._edges[last_node]) < 0.1:
                    probs[i] = 1/q*probs[i]
            probs[node] = probs[node]/10
            probs = probs/np.sum(probs)
            return np.random.choice(np.arange(self._node_num), p=probs)

    def random_walk(self, n=16, length=32):
        walks = []
        for i in range(self._node_num):
            for _ in range(n):
                walk = [str(i)]
                start_node = i
                for __ in range(length-1):
                    start_node = self._next_node(start_node, walk)
                    walk.append(str(start_node))
                walks.append(walk)
        return walks

def geo_embed(flow, embed_size=EMBEDSIZE):
    flow = np.mean(flow, axis=2)
    _, n = flow.shape
    graph = Graph(n)
    for i in range(n):
        for j in range(n):
            weight = np.mean(np.abs(flow[:, i] - flow[:, j]))
            graph.add_weight(i, j, weight)
    walks = graph.random_walk()
    # print("Start embedding")
    model = Word2Vec(walks,
                     sg=1,
                     size=embed_size,
                     window=3,
                     min_count=2,
                     negative=3,
                     sample=0.001,
                     hs=1,
                     workers=8)
    features = np.zeros((n, embed_size))
    for i in range(n):
        features[i, :] = model[str(i)]
    return features
