# Experiment 8
### Clustering Human Activity Recognition Data using K-Means, DBSCAN, and Hierarchical Clustering

## Aim
Cluster the 10,299 smartphone-sensor windows of the UCI HAR dataset (561 standardized features, 6 activities) with K-Means, DBSCAN and hierarchical (Ward) clustering, and compare the results with internal metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) and external metrics (ARI, NMI).

## Algorithm Comparison (Final)

| Model | Clusters | Noise (%) | Silhouette | ARI | NMI |
|---|---|---|---|---|---|
| K-Means (k=2) | 2 | 0.0 | **0.3937** | 0.3296 | 0.5455 |
| K-Means (k=6) | 6 | 0.0 | 0.1099 | 0.4196 | 0.5593 |
| DBSCAN (561-D) | 2 | 11.5 | 0.3396 | 0.0298 | 0.0882 |
| HAC Ward (k=6) | 6 | 0.0 | 0.1170 | **0.4599** | **0.6015** |

*K-Means k chosen by silhouette (peak at k=2; elbow gave k=4); k=6 reported alongside since there are 6 true activities. Single and average linkage collapsed to one giant cluster (ARI under 0.003) and are omitted above.*

## Learning Outcomes
- Learned that clustering follows whatever structure the data actually has, not the labels you expect: K-Means (k=2) and a two-cluster Ward cut both found static vs dynamic activity, not the six named activities.
- Learned that more clusters can't separate two classes that look the same in the data: no method, not even Ward at k=6, ever told sitting from standing apart.
- Learned that DBSCAN's noise label can be useful on its own, flagging 49% of walking-downstairs windows as outliers while barely touching the static activities.
- Learned that internal metrics can point the wrong way: silhouette and Davies-Bouldin rated single-linkage's one-giant-cluster result as best, while Calinski-Harabasz got it right.
