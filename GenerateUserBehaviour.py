import random

DATASET = 10
SEED = 10 * DATASET
random.seed(SEED)


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


nodes, edges = readNetwork("dataset/facebook_combined.txt")

prob_retweet = dict()
prob_report = dict()

for u in nodes:
    prob_retweet[u] = random.uniform(0, 1)
    prob_report[u] = random.uniform(0, 1)


with open("dataset/prob_retweet-" + str(DATASET) + ".txt", "w") as f:
    f.write(str(prob_retweet))

with open("dataset/prob_report-" + str(DATASET) + ".txt", "w") as f:
    f.write(str(prob_report))
