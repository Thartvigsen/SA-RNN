# Selective-Activation Recurrent Neural Network
Code for "Learning to Selectively Update State Neurons in Recurrent Networks",
published at <b>CIKM, 2020</b>. If you have any questions, please feel free to contact [Tom Hartvigsen](https://thartvigsen.github.io) at twhartvigsen@wpi.edu


In this repository, you can find a working example of our proposed SA-RNN method.
It can be used in leiu of a general pytorch RNN as follows:
```
from model import SARNN

SARNN = SARNN(ninp, nhid, nclasses, nepoch)

logits, update_decisions = SARNN(X, current_training_epoch)
```

For further details, please take a look at the [example](synthetic_example.py).

If you find this code or our work useful, please cite our paper:
```
@inproceedings{hartvigsen2020learning,
  title={Learning to Selectively Update State Neurons in Recurrent Networks},
  author={Hartvigsen, Thomas and Sen, Cansu and Kong, Xiangnan and Rundensteiner, Elke},
  booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
  pages={485--494},
  year={2020}
}
```
