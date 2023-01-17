import torch, math
print("PyTorch has version {}".format(torch.__version__))
print("CUDA has version {}".format(torch.version.cuda))
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn.parameter import Parameter
from torch_geometric.data import Data, download_url, extract_gz, HeteroData
from torch_geometric.nn import GAE, GCNConv, VGAE
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import train_test_split_edges, negative_sampling, degree


def load_node_mapping(datafile, index_col, offset=0):
    """
    将每个独特的节点映射到一个唯一的整数索引.

    Args
    ----
    datafile: 包含图数据的文件
    index_col: 包含关注节点的列的名字
    offset: 将生成的索引移动的数量

    Returns
    -------
    从节点名到整数索引的映射
    """
    df = pd.DataFrame(datafile)
    df.set_index(index_col, inplace=True)
    mapping = {index_id: i + offset for i, index_id in enumerate(df.index.unique())}
    return mapping


def load_edge_list(datafile, src_col, src_mapping, dst_col, dst_mapping):
    """
    给定节点映射, 返回按照节点整数索引的边列表.

    Args
    ----
    datafile: 包含图数据的文件
    src_col: 与源节点相关的列的名字
    src_mapping: 从源节点名到整数索引的映射
    dst_col: 与目标节点相关的列的名字
    dst_mapping: 从目标节点名到整数索引的映射

    Returns
    -------
    从节点名到整数索引的映射
    """
    df = pd.DataFrame(datafile)
    src_nodes = [src_mapping[index] for index in df[src_col]]
    dst_nodes = [dst_mapping[index] for index in df[dst_col]]
    edge_index = torch.tensor([src_nodes, dst_nodes])
    return edge_index


def construct_feature(drug_dis_matrix):
    drug_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj


def load_feature(feature, datafile, index_col, offset=0):
    df = pd.DataFrame(datafile)
    df.set_index(index_col, inplace=True)
    index = df.index.unique() + offset - 1
    index = np.array(index, dtype='int')
    x = feature[index]
    return torch.tensor(x, dtype=torch.float)


def load_similarity_file(sim_matrix, mapping):
    # 将相似性最高的前5%添加为边
    sim_matrix = sim_matrix - np.eye(sim_matrix.shape[0])
    a = math.ceil(sim_matrix.shape[0] * sim_matrix.shape[0] * 0.05)
    b = np.partition(sim_matrix.reshape(-1), -a)[-a]
    sim_matrix = np.where(sim_matrix >= b, 1, 0)

    # 转成DataFrame，筛选并添加边索引
    sim = pd.DataFrame(sim_matrix)
    sim = pd.melt(sim.reset_index(), id_vars=['index'], value_vars=sim.columns.values)
    sim.columns = ['d1', 'd2', 'value']
    sim = sim.drop(index=sim[(sim.value == 0)].index.tolist()).reset_index(drop=True)
    sim[['d1', 'd2']] = sim[['d1', 'd2']].astype('float')
    src_col, dst_col = 'd1', 'd2'
    edge_index = load_edge_list(sim, src_col, mapping, dst_col, mapping)
    return edge_index


def initialize_data(datafile_path1, datafile_path2, datafile_path3, datafile_path4):
    """
    给定tsv文件, 指明药物-lncRNA的相互作用, 索引节点, 并构建一 HeteroData 对象.
    """
    # 获得疾病节点映射和基因节点映射.
    # 每个节点类型都有自己的整数id的集合.
    df = np.loadtxt(datafile_path1)
    df = df[np.where(df[:, 2] == 1)]
    drug_col, lnc_col = 0, 1
    drug_mapping = load_node_mapping(df, drug_col, offset=0)
    lnc_mapping = load_node_mapping(df, lnc_col, offset=0)
    print("Number of drugs:", len(drug_mapping))
    print("Number of lncRNAs:", len(lnc_mapping))

    # 根据分配给节点的整数索引来获取边索引.
    edge_index = load_edge_list(
        df, drug_col, drug_mapping, lnc_col, lnc_mapping)

    # 添加反向边索引
    rev_edge_index = load_edge_list(
        df, lnc_col, lnc_mapping, drug_col, drug_mapping)

    # 创建节点特征
    x = np.loadtxt(datafile_path2, delimiter=',')
    x = construct_feature(x)

    # 构建一个 HeteroData 对象
    data = HeteroData()

    # 不同节点，不同特征
    data['drug'].x = load_feature(x, df, drug_col, offset=0)
    data['lnc'].x = load_feature(x, df, lnc_col, offset=len(drug_mapping))
    data['drug'].num_nodes = len(drug_mapping)
    data['lnc'].num_nodes = len(lnc_mapping)

    # 创建异质节点间的有向边
    data['drug', 'regulate', 'lnc'].edge_index = edge_index
    data['lnc', 'regulate_by', 'drug'].edge_index = rev_edge_index

    # 创建同质节点间的有向边
    sim_drug = np.loadtxt(datafile_path3, delimiter=',')
    drug_edge_index = load_similarity_file(sim_drug, drug_mapping)
    data['drug', 'sim', 'drug'].edge_index = drug_edge_index
    sim_lnc = np.loadtxt(datafile_path4, delimiter='\t')
    lnc_edge_index = load_similarity_file(sim_lnc, lnc_mapping)
    data['lnc', 'sim', 'lnc'].edge_index = lnc_edge_index

    return data, drug_mapping, lnc_mapping


data_path1 = r'C:/Users/朱济村/Desktop/DLA-GCN/data/all_drug_lnc_pairs2.txt'
data_path2 = r'C:/Users/朱济村/Desktop/DLA-GCN/data/all_drug_lnc_pairs_matrix2.txt'
data_path3 = r'C:/Users/朱济村/Desktop/DLA-GCN/data/drugsim_matrix2.txt'
data_path4 = r'C:/Users/朱济村/Desktop/DLA-GCN/data/lnc_sim3.txt'
df = np.loadtxt(data_path1)
df = df[np.where(df[:, 2] == 1)]
df = pd.DataFrame(df)
print(df.head(), '\n')

# Read data and construct HeteroData object.
data_object, drug_mapping, lnc_mapping = initialize_data(data_path1, data_path2, data_path3, data_path4)
print(data_object)
node_type, edge_type = data_object.metadata()
print(node_type)
print(edge_type)
print(data_object[edge_type[0]].edge_index)
# homogeneous_data = data_object.to_homogeneous()
# print(homogeneous_data)
# print(homogeneous_data.x)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fold = 5

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, dropout):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_size, cached=True)
        self.conv2 = GCNConv(hidden_size, hidden_size, cached=True)
        self.conv3 = GCNConv(hidden_size, out_channels, cached=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x_temp1 = self.conv1(x, edge_index).relu()
        x_temp2 = self.dropout(x_temp1)
        x_temp2 = self.conv2(x_temp2, edge_index)
        x_temp3 = self.dropout(x_temp2)
        return self.conv3(x_temp3, edge_index)


def gae_train(train_data, gae_model, optimizer):
    gae_model.train()
    optimizer.zero_grad()
    z = gae_model.encode(train_data.x, train_data.edge_index)

    pos_y = torch.ones(train_data.pos_edge_index.size(1), dtype=torch.long)
    neg_y = torch.zeros(train_data.neg_edge_index.size(1), dtype=torch.long)
    y = torch.cat([pos_y, neg_y], dim=0).to(device)
    pos_pred = gae_model.decoder(z, train_data.pos_edge_index, sigmoid=True)
    neg_pred = gae_model.decoder(z, train_data.neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0).reshape(-1, 1)
    pred = torch.cat((1 - pred, pred), dim=1)

    loss = F.cross_entropy(pred, y)
    loss.backward(retain_graph=True)
    optimizer.step()
    return float(loss)


def gae_test(test_data, gae_model):
    gae_model.eval()
    z = gae_model.encode(test_data.x, test_data.edge_index)
    return gae_model.test(z, test_data.pos_edge_label, test_data.neg_edge_label)


def split_edges(data, k, d, l):
    node_types, edge_types = data.metadata()
    edge_index_1 = data[edge_types[0]].edge_index
    edge_index_1[1] += d
    edge_index_2 = data[edge_types[1]].edge_index
    edge_index_2[0] += d
    edge_index_3 = data[edge_types[2]].edge_index
    edge_index_4 = data[edge_types[3]].edge_index
    edge_index_4 += d

    train = Data()
    test = Data()
    train.x = torch.cat((data[node_types[0]].x, data[node_types[1]].x), dim=0)
    test.x = torch.cat((data[node_types[0]].x, data[node_types[1]].x), dim=0)
    train.num_nodes = data[node_types[0]].num_nodes + data[node_types[1]].num_nodes
    test.num_nodes = data[node_types[0]].num_nodes + data[node_types[1]].num_nodes

    eids = np.arange(edge_index_1.shape[1])
    eids = np.random.permutation(eids)  # 将顺序打乱
    pos_size = int(len(eids) * 0.2)
    train.edge_index = torch.cat((edge_index_1[:, pos_size * (k+1):],
                                  edge_index_2[:, pos_size * (k+1):],
                                  edge_index_3,
                                  edge_index_4), dim=1)
    test.edge_index = torch.cat((edge_index_1[:, pos_size * k:pos_size * (k+1)],
                                 edge_index_2[:, pos_size * k:pos_size * (k+1)],
                                 edge_index_3,
                                 edge_index_4), dim=1)

    train.pos_edge_index = edge_index_1[:, pos_size * (k+1):]
    train.pos_edge_label = torch.ones(size=[(edge_index_1.shape[1] - pos_size) * 2, 1], dtype=torch.long)

    test.pos_edge_index = edge_index_1[:, pos_size * k:pos_size * (k+1)]
    test.pos_edge_label = torch.ones(size=[pos_size * 2, 1], dtype=torch.long)

    adj = torch.cat((torch.ones(size=[d, d]), torch.zeros(size=[d, l])), dim=1)
    for j in range(edge_index_1.shape[1]):
        adj[edge_index_1[0, j], edge_index_1[1, j]] = 1
    src_nodes, dst_nodes = torch.where(adj == 0)

    neg_edge_index = torch.stack([src_nodes, dst_nodes])
    rev_neg_edge_index = torch.stack([dst_nodes, src_nodes])
    print(neg_edge_index)
    eids = np.arange(neg_edge_index.shape[1])
    eids = np.random.permutation(eids)
    neg_size = int(len(eids) * 0.2)
    train.neg_edge_index = neg_edge_index[:, neg_size * (k+1):]
    train.neg_edge_label = torch.ones(size=[(neg_edge_index.shape[1] - neg_size) * 2, 1], dtype=torch.long)
    test.neg_edge_index = neg_edge_index[:, neg_size * k:neg_size * (k+1)]
    test.neg_edge_label = torch.ones(size=[neg_size *2, 1], dtype=torch.long)

    return train, test


for i in range(fold):
    if i > 0:
        break
    print("------this is %dth cross validation------" % (i + 1))
    train_set, test_set = split_edges(data_object, i, len(drug_mapping), len(lnc_mapping))
    train_set.to(device)
    test_set.to(device)
    print('train_set', train_set)
    print('test_set', test_set)

    HIDDEN_SIZE = 64
    OUT_CHANNELS = 64
    EPOCHS = 4000

    gae_model = GAE(GCNEncoder(data_object.num_nodes, HIDDEN_SIZE, OUT_CHANNELS, 0.4))
    gae_model = gae_model.to(device)
    print(gae_model)

    losses = []
    test_auc = []
    test_ap = []
    train_aucs = []
    train_aps = []

    optimizer = torch.optim.Adam(gae_model.parameters(), lr=0.01)

    for epoch in range(1, EPOCHS + 1):
        loss = gae_train(train_set, gae_model, optimizer)
        losses.append(loss)
        auc, ap = gae_test(test_set, gae_model)
        test_auc.append(auc)
        test_ap.append(ap)

        train_auc, train_ap = gae_test(train_set, gae_model)

        train_aucs.append(train_auc)
        train_aps.append(train_ap)
        if epoch % 100 == 0:
            print('Epoch: {:03d}, test AUC: {:.4f}, test AP: {:.4f}, train AUC: {:.4f}, train AP: {:.4f}, loss:{:.4f}'.format(epoch, auc, ap, train_auc, train_ap, loss))
    print(gae_model.encoder.conv1.lin.weight)
    print(gae_model.encoder.conv2.lin.weight)
    print(gae_model.encoder.conv3.lin.weight)
