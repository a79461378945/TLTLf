### Dataset

We use the dataset in _Bridging ltlf inference to gnn inference for learning ltlf formulae_.

The dataset is available in https://github.com/a79461378945/Bridging-LTLf-Inference-to-GNN-Inference-for-Learning-LTLf-Formulae/tree/main/datasets

### Code

Start learning from the traces in ```train.json```

```
python train.py
```

Test the learnt neural model using ```test.json``` as test data:

```
python test.py
```

Interpret the parameters and use the ```test.json``` as test data:

```
python interpret.py
```

