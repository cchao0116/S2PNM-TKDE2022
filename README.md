# S2PNM-TKDE2022
Code for TKDE22 paper "Modeling Dynamic User Preference via Dictionary Learning for Sequential Recommendation"

## Data Format

The implementation is desiged for top-N recommendations on implicit data, and thus it takes user-item pairs as input:

```
uid,sid,time
1,1,98765
```

## Installation

The program requires Python 3.7+ with NumPy, Pandas and Tensorflow 1.x.

## Train and Test

Let us assume the original user-item-timestamp triplets (for example, MovieLens datasets) are stored in <ins>*user_train.csv*</ins>,
then it is quite simple to produce the train/validation/testing data and evaluate our S2PNM model by running
```
bash runme.sh
```


## Citation

If you find our code useful for your research, please consider cite.

```
@article{chen2022dyna,
    author = {Chen, Chao and Li, Dongsheng and Yan, Junchi and Yang, Xiaokang},
    journal = {IEEE Transactions on Knowledge and Data Engineering},
    title = {Modeling Dynamic User Preference via Dictionary Learning for Sequential Recommendation},
    year={2022},
    volume={34},
    number={11},
    pages={5446-5458}
}
```
