import os
import pickle
import sys

import torch

sys.path.append("./")
from model.function import embedding_lookup, init_embedding


def create_embeddings(
    embeddings_pkl_path,
    embedding_num=100,
    embedding_dim=128,
    stddev=0.01,
):
    embeddings = init_embedding(embedding_num, embedding_dim, stddev)
    embeddings = embeddings.cuda()
    torch.save(embeddings, os.path.join(embeddings_pkl_path, "EMBEDDINGS.pkl"))


# Paths to the train.obj and output pickle files.
embeddings_pkl_path = "./data"

# Create and save the embeddings.
if __name__ == "__main__":
    create_embeddings(embeddings_pkl_path)
