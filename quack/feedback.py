import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


class PreferenceGenerator:
    def __init__(self,
                 embeddings,
                 entity_to_row,
                 id2ent,
                 id2rel,
                 max_num_partitions,
                 plot=False,
                 ):
        self.embeddings = embeddings
        self.entity_to_row = entity_to_row
        self.id2ent = id2ent
        self.id2rel = id2rel
        self.max_num_partitions = max_num_partitions
        self.plot = plot

    def generate(self, query, target_ids, descriptions):
        subject, predicate = query[0], query[1][0]
        if self.plot:
            print(query)
            print(f"Subject: [{self.id2ent[subject]}] {descriptions[self.entity_to_row[self.id2ent[subject]]][:150]}")
            print(f"Predicate: {self.id2rel[predicate]}")

        # Embed targets
        entities = [self.id2ent[t] for t in target_ids]
        entity_rows = [self.entity_to_row[e] for e in entities]
        target_embeddings = self.embeddings[entity_rows]
        labels = [descriptions[r][:150] for r in entity_rows]
        if self.plot:
            for ent_id, l in zip(entities, labels):
                print(f"\t [{ent_id}] {l}")

        # Find clusters within answers to each query in batch
        z = linkage(target_embeddings, method="average", metric="cosine")

        # Plot the dendrogram to visually inspect possible clusters
        if self.plot:
            plt.figure(figsize=(20, 7))
            dendrogram(z, labels=labels, orientation="left")
            plt.xlabel("Query targets")
            plt.ylabel("Euclidean Distance")
            plt.subplots_adjust(left=0.0, right=0.3)
            plt.show()

        # Compute a mapping from cluster number to data points in it
        n = target_embeddings.shape[0]
        cluster_map = {i: [i] for i in range(n)}
        for i, row in enumerate(z):
            cluster_1, cluster_2 = int(row[0]), int(row[1])
            new_cluster = cluster_map[cluster_1] + cluster_map[cluster_2]
            cluster_map[n + i] = new_cluster  # Update cluster map

        # Traverse the hierarchy from top to bottom collecting clusters
        cluster_threshold = int(0.2 * n)

        all_indices_set = {i for i in range(n)}
        partitions = []
        done = False
        for i in range(len(z) - 1, -1, -1):
            u, v, *_ = z[i].astype(int)

            for cluster_id in (u, v):
                pos_idx = cluster_map[cluster_id]
                if len(pos_idx) >= cluster_threshold:
                    neg_idx = all_indices_set.difference(set(pos_idx))

                    positives = [target_ids[i] for i in pos_idx]
                    negatives = [target_ids[i] for i in neg_idx]

                    partitions.append((positives, negatives))

                    if len(partitions) >= self.max_num_partitions:
                        done = True
                        break
            if done:
                break

        return partitions
