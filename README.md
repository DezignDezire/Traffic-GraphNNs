# Traffic-GraphNNs
Traffic Forecasting on Highways using Graph Neural Networks

installing Pytorch Lightning Temporal, which depends on pytorch Scatter & pytroch Sparse can be quite cumbersome, but it's worth it.

run training:

```python TLS_GNN_mytune.py```

run evaluation:

```python TLS_GNN_eval.py```

data exploration is done in:

```TLS_GNN_A01.ipynb```

and the results are analysed in:

```analysis.ipynb```

results:

|   minutes | direction   |   baseline 0 |   GCN_LSTM 50 |   GCN_LSTM 100 |   A3TGCN 50 |   A3TGCN 100 |
|----------:|:------------|-------------:|--------------:|---------------:|------------:|-------------:|
|         5 | 1           |     1.31067  |      0.456955 |       1.0536   |    1.05083  |     1.04685  |
|         5 | 1,2         |     1.28284  |      1.01506  |       1.02451  |    1.03516  |     1.03057  |
|         5 | 2           |     1.2576   |      1.0034   |       0.590801 |    1.02898  |     1.02508  |
|        30 | 1           |     0.425698 |      1.02005  |       0.233556 |    1.00553  |     0.517924 |
|        30 | 1,2         |     0.413225 |      0.983687 |       0.267276 |    0.994697 |     0.992599 |
|        30 | 2           |     0.404753 |      1.01366  |       0.255877 |    0.523816 |     0.993287 |
|        60 | 1           |     0.591821 |      0.297151 |       0.290184 |    1.01491  |     1.00851  |
|        60 | 1,2         |     0.566442 |      0.317912 |       0.301587 |    0.991007 |     0.988065 |
|        60 | 2           |     0.544157 |      0.988103 |       0.285369 |    0.984486 |     0.979044 |
