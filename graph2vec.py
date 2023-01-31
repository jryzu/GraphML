import os
import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from param_parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class WeisfeilerLehmanMachine:

    def __init__(self, graph, features, iterations):

        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):

        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):

        for _ in range(self.iterations):
            self.features = self.do_a_recursion()

def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def dataset_reader(path):

    name = path2name(path)
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])

    if "features" in data.keys():
        features = data["features"]
        features = {int(k): v for k, v in features.items()}
    else:
        features = nx.degree(graph)
        features = {int(k): v for k, v in features}
       
    return graph, features, name

def feature_extractor(path, rounds):

    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc

def save_embedding(output_path, model, files, dimensions):

    out = []
    for f in files:
        identifier = path2name(f)
        out.append([identifier] + list(model.docvecs["g_"+identifier]))
    column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(["type"])
    out.to_csv(output_path, index=None)

def main(args):

    graphs = glob.glob(os.path.join(args.input_path, "*.json"))
    print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=args.workers)(delayed(feature_extractor)(g, args.wl_iterations) for g in tqdm(graphs))
    print("\nOptimization started.\n")

    model = Doc2Vec(document_collections,
                    vector_size=args.dimensions,
                    window=0,
                    min_count=args.min_count,
                    dm=0,
                    sample=args.down_sampling,
                    workers=args.workers,
                    epochs=args.epochs,
                    alpha=args.learning_rate)

    save_embedding(args.output_path, model, graphs, args.dimensions)

if __name__ == "__main__":
    args = parameter_parser()
    main(args)
