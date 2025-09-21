Work in progress here.

Sample of config.json

```
{
  "output_path": "results",
  "scalar": "double",
  "experiments": [

    {
      "name": "big_gaussian",
      "enabled": true,
      "algorithms": [
        "frobenius removal",
        "spectral removal",
        "rank-revealing QR",
        "volume removal"
      ],
      "matrix": {
        "type": "gaussian matrix",
        "rows": 50,
        "cols": 100,
        "seed": 42
      },
      "k_values": [50, 60],
      "trials_per_k": 50
    },

    {
      "name": "small_graph_incidence",
      "enabled": true,
      "algorithms": [
        "spectral selection",
        "dual set"
      ],
      "matrix": {
        "type": "graph incidence matrix",
        "rows": 10,
        "cols": 50
      },
      "k_values_range": {
        "start": 10,
        "stop": 50,
        "step": 2
      },
      "trials_per_k": 10
    },

    {
      "name": "big_orthonormal_disabled",
      "enabled": false,
      "algorithms": [
        "frobenius removal"
      ],
      "matrix": {
        "type": "matrix with orthonormal rows or columns",
        "rows": 50,
        "cols": 100
      },
      "k_values": [50, 75, 100],
      "trials_per_k": 20
    }
  ]
}
```