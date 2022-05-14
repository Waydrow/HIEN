
# Hierarchical Intention Embedding Network
This is our Tensorflow implementation for the paper:
>Zuowu Zheng, Changwang Zhang, Xiaofeng Gao, and Guihai Chen. 2022. 
HIEN: Hierarchical Intention Embedding Network for Click-Through Rate Prediction. 
In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’22), July 11–15, 2022, Madrid, Spain.

## Environment Requirement
 The code has been tested running under Python 3.5. The required packages are as follows:
 * tensorflow == 1.4
 * numpy == 1.14.3
 * scipy == 1.1.0
 * sklearn == 0.19.1
 * pandas == 0.22.0
Note that we refer to the implementation of [NGCF](https://github.com/xiangwang1223/neural_graph_collaborative_filtering).

## Dataset
1. [Alimama Data](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56) and [Tmall Data](https://tianchi.aliyun.com/dataset/dataDetail?dataId=42)
2. Extract the files into the `data/raw_data` directory
3. Follow the code of the data preprocessing in [DSIN](https://github.com/shenweichen/DSIN) to preprocess data
4. Then we need to extract structure information in item attributes. Take `(adgroup_id, campaign_id, customer)` as example, we build a dictionary for
`(campaign_id: adgroup_id)` and `(customer: campaign_id)` respectively, which contains the relation dependencies of these attributes.
 

## Example to Run the Codes
```
python main.py --regs [1e-5]
               --embed_size 128
               --lr 0.001
               --save_flag 0
               --pretrain 0
               --batch_size 4096
               --epoch 50
               --verbose 50
               --node_dropout [0.1]
               --mess_dropout [0.1,0.1,0.1]
               --tree_type gcn
```

Some important arguments:
* `alg_type`
  * It specifies the type of graph convolutional layer.
  * Here we provide three options:
    * `ngcf` , proposed in [Neural Graph Collaborative Filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/sigir19-NGCF.pdf), SIGIR2019. Usage: `--alg_type ngcf`.
    * `gcn`, proposed in [Semi-Supervised Classification with Graph Convolutional Networks](https://openreview.net/pdf?id=SJU4ayYgl), ICLR2018. Usage: `--alg_type gcn`.
    * `gcmc`, propsed in [Graph Convolutional Matrix Completion](https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_32.pdf), KDD2018. Usage: `--alg_type gcmc`.

* `adj_type`
  * It specifies the type of laplacian matrix where each entry defines the decay factor between two connected nodes.
  * Here we provide four options:
    * `ngcf` (by default), where each decay factor between two connected nodes is set as 1(out degree of the node), while each node is also assigned with 1 for self-connections. Usage: `--adj_type ngcf`.
    * `plain`, where each decay factor between two connected nodes is set as 1. No self-connections are considered. Usage: `--adj_type plain`.
    * `norm`, where each decay factor bewteen two connected nodes is set as 1/(out degree of the node + self-conncetion). Usage: `--adj_type norm`.
    * `gcmc`, where each decay factor between two connected nodes is set as 1/(out degree of the node). No self-connections are considered. Usage: `--adj_type gcmc`.

* `node_dropout`
  * It indicates the node dropout ratio, which randomly blocks a particular node and discard all its outgoing messages. Usage: `--node_dropout [0.1] --node_dropout_flag 1`
  * Note that the arguement `node_dropout_flag` also needs to be set as 1, since the node dropout could lead to higher computational cost compared to message dropout.

* `mess_dropout`
  * It indicates the message dropout ratio, which randomly drops out the outgoing messages. Usage `--mess_dropout [0.1,0.1,0.1]`.

* `tree_type`
  * It indicates the aggregator we used in attribute graph aggregation
  * Here we provide four options:
    * `GCN`
    * `NGCF`
    * `LightGCN`
    * `Concat & Product (CP)`
    