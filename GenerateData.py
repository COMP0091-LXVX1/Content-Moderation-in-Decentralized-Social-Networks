import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import ast

DATASET = 10
SEED = 10 * DATASET
random.seed(10)
UNDIRECTED = True  # True for Facebook, False for Twitter


class User:
    def __init__(self, id, prob_retweet, prob_report):
        self.id = id
        self.prob_retweet = prob_retweet
        self.prob_report = prob_report
        self.following = set()
        self.followers = set()
        self.ego_network = set()


def readNetwork(path):
    nodes, edges = set(), set()

    with open(path) as f:
        lines = f.readlines()
        for l in lines:
            ls = l.split()
            a, b = int(ls[0]), int(ls[1])

            nodes.add(a)
            nodes.add(b)

            edges.add((a, b))

    return nodes, edges


def getEgoNetworks(id_to_user):

    for node in id_to_user.keys():
        connections = id_to_user[node].following.union(id_to_user[node].followers)
        connections.add(node)

        for u in connections:
            for v in connections:
                if v in id_to_user[u].following:
                    id_to_user[node].ego_network.add((u, v))


def plotNetwork(edges):
    G = nx.Graph()

    for (a, b) in edges:
        G.add_edge(a, b)

    nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_size=500, node_color="skyblue", node_shape="s", alpha=0.5, linewidths=40)
    plt.savefig("random_ego_network.png")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def propagatePost(id_to_user, author, post_quality):
    seen = set()
    seen.add(author)
    sharers = [(author, 0)]
    reports = set()

    while len(sharers) > 0:
        node, step = sharers[0]
        del sharers[0]

        for v in id_to_user[node].followers:
            if v not in seen:
                seen.add(v)
                # if random.uniform(0, 1) < id_to_user[v].prob_retweet * post_quality:
                #     sharers.append((v, step+1))

                if random.uniform(0, 1) < id_to_user[v].prob_retweet * (1 + post_quality) / 2:
                    sharers.append((v, step+1))

                if random.uniform(0, 1) < id_to_user[v].prob_report * (1 - post_quality) / 2:
                    reports.add(v)

    return len(seen)


def maxPropagate(id_to_user, author):
    seen = set()
    seen.add(author)
    sharers = [(author, 0)]

    while len(sharers) > 0:
        node, step = sharers[0]
        del sharers[0]

        for v in id_to_user[node].followers:
            if v not in seen:
                seen.add(v)
                sharers.append((v, step+1))

    return len(seen)


def get_no_of_connections(id_to_user, author):
    ego_nodes = id_to_user[author].followers
    ego_nodes.add(author)

    conn = 0
    for f in ego_nodes:
        for v in id_to_user[f].followers:
            if f < v and v in ego_nodes:
                conn += 1

    return conn


def get_no_of_triangles(id_to_user, author):
    ego_nodes = id_to_user[author].followers
    ego_nodes.add(author)

    triangles = 0
    for f in ego_nodes:
        for v in id_to_user[f].followers:
            if f < v and v in ego_nodes:
                for u in id_to_user[v].followers:
                    if v < u and u in ego_nodes:
                        triangles += 1

    return triangles


def gettingDataPoint(id_to_user, author, noisy_post_quality):
    no_of_followers = len(id_to_user[author].followers)
    ds.append(no_of_followers)

    no_of_connections = get_no_of_connections(id_to_user, author)
    ds.append(no_of_connections)

    no_of_triangles = get_no_of_triangles(id_to_user, author)
    ds.append(no_of_triangles)

    ds.append(no_of_followers / len(nodes))

    ds.append(no_of_connections / len(edges))

    ds.append(noisy_post_quality)

    true_propagation = propagatePost(id_to_user, author, noisy_post_quality)
    ds.append(true_propagation)


nodes, edges = readNetwork("dataset/facebook_combined.txt")

with open("dataset/prob_retweet-" + str(DATASET) + ".txt", "r") as f:
    contents = f.read()
    prob_retweet = ast.literal_eval(contents)

with open("dataset/prob_report-" + str(DATASET) + ".txt", "r") as f:
    contents = f.read()
    prob_report = ast.literal_eval(contents)

id_to_user = dict()
for node in nodes:
    id_to_user[node] = User(node, prob_retweet[node], prob_report[node])

for (u, v) in edges:
    id_to_user[u].following.add(v)
    id_to_user[v].followers.add(u)

    if UNDIRECTED:
        id_to_user[v].following.add(u)
        id_to_user[u].followers.add(v)


dataPoints = 200000
no_of_featuers = 6
ds = []

for step in range(dataPoints):
    u = random.choice(list(nodes))
    post_quality = random.uniform(-1, 1)
    gettingDataPoint(id_to_user, u, post_quality)
    if step % 100 == 0:
        print(step)


pd_ds = pd.DataFrame()
pd_ds['no_of_followers'] = ds[::(no_of_featuers+1)]
pd_ds['no_of_connections'] = ds[1::(no_of_featuers+1)]
pd_ds['no_of_triangles'] = ds[2::(no_of_featuers+1)]
pd_ds['user_coverage'] = ds[3::(no_of_featuers+1)]
pd_ds['conn_coverage'] = ds[4::(no_of_featuers+1)]
pd_ds['post_quality'] = ds[5::(no_of_featuers+1)]
pd_ds['true_propagation'] = ds[6::(no_of_featuers+1)]

pd_ds.to_csv("dataset/DATASET-" + str(DATASET) + ".csv")
