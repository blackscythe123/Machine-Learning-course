# Experiment 8
### Clustering Human Activity Recognition Data using K-Means, DBSCAN, and Hierarchical Clustering

## Aim
Cluster the 10,299 smartphone-sensor windows of the UCI HAR dataset (561 standardized features, 6 activities) with K-Means, DBSCAN and hierarchical (Ward) clustering, without using the labels, and compare the results with internal metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) and external metrics (ARI, NMI).

## K-Means Elbow Method Results

| k | WCSS (x10^6) | Silhouette | ARI | NMI |
|---|---|---|---|---|
| **2** | **3.2729** | **0.3937** | 0.3296 | 0.5455 |
| 3 | 2.9211 | 0.3155 | 0.3317 | 0.5183 |
| 4 | 2.7816 | 0.1500 | 0.2990 | 0.4694 |
| 5 | 2.6548 | 0.1286 | 0.2912 | 0.4607 |
| 6 | 2.5772 | 0.1099 | 0.4196 | 0.5593 |
| 7 | 2.5198 | 0.0816 | **0.4332** | 0.5546 |
| 8 | 2.4654 | 0.0749 | 0.4266 | 0.5516 |

*Elbow rule (chord distance) gave k=4, silhouette gave k=2; silhouette decided. k=6 is reported alongside because there are 6 true activities.*

## DBSCAN Selected Settings (label-free selection: 2 to 12 clusters, at most 30% noise, best silhouette)

| Space | eps | minPts | Clusters | Noise (%) | Silhouette | DB | CH | ARI | NMI |
|---|---|---|---|---|---|---|---|---|---|
| Full 561-D | 17.07 | 10 | 2 | 11.5 | 0.3396 | 0.8729 | 103.2 | 0.0298 | 0.0882 |
| PCA-50 | 14.68 | 20 | 2 | 6.3 | 0.3271 | 0.9138 | 120.7 | 0.0119 | 0.0540 |

*Full-space cluster sizes: 9079 and 31 windows, plus 1189 noise windows.*

## Hierarchical Clustering: Linkage Comparison (cut at 6 clusters)

| Linkage | Cophenetic | Largest Cluster (%) | Silhouette | DB | CH | ARI | NMI |
|---|---|---|---|---|---|---|---|
| Single | 0.8139 | 99.9 | 0.5601 | 0.2733 | 28.9 | 0.0001 | 0.0013 |
| Complete | 0.7523 | 54.6 | 0.3792 | 1.3004 | 1898.4 | 0.3294 | 0.5421 |
| Average | 0.8557 | 98.5 | 0.4431 | 1.1020 | 161.1 | 0.0024 | 0.0220 |
| **Ward** | 0.6767 | 31.1 | 0.1170 | 2.4820 | 2349.7 | **0.4599** | **0.6015** |

## Algorithm Comparison

| Model | Clusters | Noise (%) | Silhouette | DB | CH | ARI | NMI |
|---|---|---|---|---|---|---|---|
| K-Means (k=2) | 2 | 0.0 | **0.3937** | 1.0707 | **7880.8** | 0.3296 | 0.5455 |
| K-Means (k=6) | 6 | 0.0 | 0.1099 | 2.3836 | 2556.5 | 0.4196 | 0.5593 |
| DBSCAN (561-D) | 2 | 11.5 | 0.3396 | **0.8729** | 103.2 | 0.0298 | 0.0882 |
| DBSCAN (PCA-50) | 2 | 6.3 | 0.3271 | 0.9138 | 120.7 | 0.0119 | 0.0540 |
| HAC Ward (k=6) | 6 | 0.0 | 0.1170 | 2.4820 | 2349.7 | **0.4599** | **0.6015** |

## Learning Outcomes
- The strongest structure in the data is static versus dynamic activity: K-Means with k=2 separates them with only 23 of 10,299 windows misplaced, and cutting Ward's tree into two clusters separates them with none misplaced.
- Internal metrics and the labels disagree on k: silhouette peaks at k=2 and the elbow at k=4, but ARI only rises at k=6 to 7, so K-Means is very sensitive to k. No algorithm separated sitting from standing, and Ward at k=6 (ARI 0.4599) was the best at recovering the six activities.
- DBSCAN mostly failed as a clusterer (one cluster of 9079 windows, one of 31, 11.5% noise) but worked as an outlier detector: 49% of walking-downstairs windows were flagged as noise, against under 5% for the static activities.
- Linkage choice decided everything: single linkage put 99.9% and average linkage 98.5% of the windows in one cluster, while complete (ARI 0.3294) and Ward (0.4599) gave usable clusters.
- Calinski-Harabasz matched the visual picture best (Spearman +0.81 with ARI across 8 partitions), while silhouette (-0.74) and Davies-Bouldin (-0.79) rated the one-giant-cluster partitions as the best ones; eight partitions is a small sample, so this is indicative only.
