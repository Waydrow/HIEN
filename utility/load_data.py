
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import pandas as pd

from pandas.compat.pickle_compat import _class_locations_map

_class_locations_map.update({
	('pandas.core.internals.managers', 'BlockManager'): ('pandas.core.internals', 'BlockManager')
})

class Data(object):
	def __init__(self, path, batch_size):
		self.path = path
		self.batch_size = batch_size

		self.data = pd.read_pickle(path + 'sampled_data_fit.pkl')
		self.train = self.data.loc[self.data.time_stamp < 1494633600]
		self.test = self.data.loc[self.data.time_stamp >= 1494633600]

		#get number of users and items
		self.n_users, self.n_items = self.data['userid'].unique().shape[0], self.data['adgroup_id'].unique().shape[0]
		self.n_train, self.n_test = self.train.shape[0], self.test.shape[0]


		self.print_statistics()

		self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

		pos_samples = self.data.loc[self.data.clk == 1]
		for i, row in pos_samples.iterrows():
			self.R[row['userid'], row['adgroup_id']] = 1


	def get_adj_mat(self):
		try:
			t1 = time()
			adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
			norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
			mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
			print('already load adj matrix', adj_mat.shape, time() - t1)

		except Exception:
			adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
			sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
			sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
			sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
		return adj_mat, norm_adj_mat, mean_adj_mat

	def create_adj_mat(self):
		t1 = time()
		adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
		adj_mat = adj_mat.tolil()
		R = self.R.tolil()

		adj_mat[:self.n_users, self.n_users:] = R
		adj_mat[self.n_users:, :self.n_users] = R.T
		adj_mat = adj_mat.todok()
		print('already create adjacency matrix', adj_mat.shape, time() - t1)

		t2 = time()

		def normalized_adj_single(adj):
			rowsum = np.array(adj.sum(1))

			d_inv = np.power(rowsum, -1).flatten()
			d_inv[np.isinf(d_inv)] = 0.
			d_mat_inv = sp.diags(d_inv)

			norm_adj = d_mat_inv.dot(adj)
			# norm_adj = adj.dot(d_mat_inv)
			print('generate single-normalized adjacency matrix.')
			return norm_adj.tocoo()

		def check_adj_if_equal(adj):
			dense_A = np.array(adj.todense())
			degree = np.sum(dense_A, axis=1, keepdims=False)

			temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
			print('check normalized adjacency matrix whether equal to this laplacian matrix.')
			return temp

		norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		mean_adj_mat = normalized_adj_single(adj_mat)

		print('already normalize adjacency matrix', time() - t2)
		return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()



	def get_num_users_items(self):
		return self.n_users, self.n_items

	def print_statistics(self):
		print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
		print('n_interactions=%d' % (self.n_train + self.n_test))
		print('n_train=%d, n_test=%d' % (self.n_train, self.n_test))


# id_feats = ['user_lid', 'adid']
# user_feats = ['age', 'gender', 'provinceid', 'cityid', 'degree']
# item_feats = ['advertiser_id', 'creative_id', 'product_id', 'product_type', 'campaign_id', 'ams_industry_level_one_id', 'ams_industry_level_two_id']
# context_feats = ['adpos', 'device_brand_id', 'device_alias_id']

class DataLoader:
	def __init__(self, batch_size, data, path):
		self.batch_size = batch_size
		self.data = data
		self.epoch_size = len(self.data) // self.batch_size
		if self.epoch_size * self.batch_size < len(self.data):
			self.epoch_size += 1
		self.i = 0

		self.campaign_ad_dic = pd.read_pickle(path + 'campaign_ad_dic.pkl')
		self.advertiser_cam_dic = pd.read_pickle(path + 'advertiser_cam_dic.pkl')

	def __iter__(self):
		self.i = 0
		return self

	def __next__(self):
		if self.i == self.epoch_size:
			raise StopIteration
		ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
													  len(self.data))]
		self.i += 1

		y = []
		campaign_ad_list = []
		advertiser_cam_list = []
		for i, row in ts.iterrows():
			y.append([float(row['clk']), 1-float(row['clk'])])
			campaign_ad_list.append(self.campaign_ad_dic[row['campaign_id']])
			advertiser_cam_list.append(self.advertiser_cam_dic[row['customer']])


		return np.array(y), ts['userid'].values, ts['adgroup_id'].values, ts['cms_segid'].values, ts['cms_group_id'].values, ts['final_gender_code'].values, \
			   ts['age_level'].values, ts['pvalue_level'].values, ts['shopping_level'].values, \
			   ts['occupation'].values, ts['new_user_class_level'].values, ts['campaign_id'].values, \
			   ts['customer'].values, ts['cate_id'].values, ts['brand'].values, \
			   ts['pid'].values, \
			   np.array(campaign_ad_list), np.array(advertiser_cam_list)
