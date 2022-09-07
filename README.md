# Experiments-on-Graph-Generative-Models
We aim to evaluate how well some popular graph generative models perform on several graph datasets.

## Progress:
- Currently working on "GDSS" proposed in "Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations" (https://arxiv.org/pdf/2202.02514.pdf). The baseline model is tested on **3** datasets (Grid, Protein, 3D Point Cloud) and measured under **4** metrics (degree, clustering, orbit, graph spectrum).

## Implementation
- It should be noted that we adopt the same datasets presets as in "Efficient Graph Generation with Graph Recurrent Attention Networks" (https://arxiv.org/pdf/1910.00760.pdf), where:
  - Grid: 100 graphs are generated with $100\leq |V| \leq 400$;
  - Protein: 918 graphs are generated with $100\leq |V| \leq 500$;
  - 3D Point-Cloud (FirstMM-DB): 41 graphs are generated with $\bar{|V|}+\bar{|E|} > 1000$

- Following the experimental setting as in "GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models" (https://arxiv.org/abs/1802.08773), we conduct a 80\%-20\% split of the graph samples in each dataset. Then we generate the same size of graph samples as the test dataset and harness the maximum mean discrepancy (MMD) to evaluate the generative graph distribution.
