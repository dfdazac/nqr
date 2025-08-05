import torch
import torch.nn as nn
import time
import numpy as np
import gc
import matplotlib.pyplot as plt


class AdjustNet(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.self_attention_1 = nn.MultiheadAttention(embed_dim=embedding_dim + 1, num_heads=1,
                                                      batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(embedding_dim + 1)
        self.fc1 = nn.Sequential(nn.Linear(embedding_dim + 1, embedding_dim), nn.ReLU())
        self.self_attention_2 = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        fc2 = nn.Linear(embedding_dim * 2 + 1, embedding_dim)  # Input: [preference_embedding, entity_embedding, score]

        self.adjust_net = nn.Sequential(fc2,
                                        nn.ReLU(),
                                        nn.Linear(embedding_dim, 1),
                                        nn.Tanh())

    def embed_preferences(self, preferences, labels):
        """Retrieve embeddings of entities specified as preferences"""
        preferences_mask = preferences < 0
        preferences[preferences_mask] = 0

        # == Part 1: Compute preference embeddings m ==
        m = self.embedding(preferences)
        # (batch_size, num_preferences, embedding_dim)
        m = torch.cat([m, labels.unsqueeze(-1)], dim=-1)
        # (batch_size, num_preferences, embedding_dim + 1)

        # Alternative 2: self attention and mean pooling
        m = self.self_attention_1(m, m, m, key_padding_mask=preferences_mask)[0]
        m = self.layer_norm_1(m)
        m = self.fc1(m)
        m = self.self_attention_2(m, m, m, key_padding_mask=preferences_mask)[0]
        m = self.layer_norm_2(m)
        # (batch_size, num_preferences, embedding_dim)
        m = torch.mean(m, dim=1)
        # (batch_size, embedding_dim)

        return m

    def adjust_scores(self, pref_embeddings, candidates, scores, return_deltas=False):
        inputs = torch.cat([pref_embeddings, candidates, scores.unsqueeze(-1)], dim=1)
        # (p * batch_size, 2 * embedding_dim)
        score_deltas = self.adjust_net(inputs).squeeze(-1)
        # (p * batch_size, 1)
        new_scores = scores + score_deltas
        # (p * batch_size)
        if return_deltas:
            return new_scores, score_deltas
        else:
            return new_scores

    def rerank_nqr(self, scores, preferences, labels):
        # == Part 1: Embed preferences ==
        m = self.embed_preferences(preferences.unsqueeze(0), labels.unsqueeze(0))
        embeddings = self.embedding.weight
        m = m.expand(embeddings.shape[0], -1)
        scores = self.adjust_scores(m, embeddings, scores)
        return scores

with torch.inference_mode():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_preferences = 10
    embedding_dim = 1000
    print(f"{'V':>10}{'Time':>20}{'VRAM(MB)':>20}")
    num_entities_list = [20000, 40000, 60000, 80000, 100000]
    all_mean_times = []
    all_std_times = []
    all_mean_vrams = []
    all_std_vrams = []
    for num_entities in num_entities_list:
        times = []
        vrams = []
        for _ in range(10):
            start_time = time.time()
            model = AdjustNet(num_entities, embedding_dim).to(device)
            scores = torch.rand(num_entities).to(device)
            preferences = torch.randint(0, num_entities, (num_preferences,)).to(device)
            labels = torch.randint(0, 2, (num_preferences,)).to(device)
            new_scores = model.rerank_nqr(scores, preferences, labels)
            elapsed = time.time() - start_time
            if torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            else:
                vram = 0.0
            times.append(elapsed)
            vrams.append(vram)
            del model, scores, preferences, labels, new_scores
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_vram = np.mean(vrams)
        std_vram = np.std(vrams)
        all_mean_times.append(mean_time)
        all_std_times.append(std_time)
        all_mean_vrams.append(mean_vram)
        all_std_vrams.append(std_vram)
        print(f"{num_entities:>10,}{mean_time:>10.4f} ± {std_time:<7.4f}{mean_vram:>10.1f} ± {std_vram:<7.1f}")

    # Plotting
    fig, ax1 = plt.subplots()
    ax1.errorbar(num_entities_list, all_mean_times, yerr=all_std_times, fmt='-o', color='tab:blue', label='Time (s)')
    ax1.set_xlabel('Number of Entities (V)')
    ax1.set_ylabel('Time (s)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.errorbar(num_entities_list, all_mean_vrams, yerr=all_std_vrams, fmt='-s', color='tab:red', label='VRAM (MB)')
    ax2.set_ylabel('VRAM (MB)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Scaling of Time and VRAM with Number of Entities')
    fig.tight_layout()
    plt.show()
