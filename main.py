
import tensorflow as tf
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utility.helper import *
from utility.batch_test import *

class NGCF(object):
	def __init__(self, fd, data_config, pretrain_data):
		# argument settings
		self.model_type = 'ngcf'
		self.adj_type = args.adj_type
		self.alg_type = args.alg_type
		self.tree_type = args.tree_type

		self.pretrain_data = pretrain_data

		self.fd = fd

		self.n_users = data_config['n_users']
		self.n_items = data_config['n_items']

		self.n_fold = 100

		self.norm_adj = data_config['norm_adj']
		self.n_nonzero_elems = self.norm_adj.count_nonzero()

		self.lr = args.lr

		self.emb_dim = args.embed_size
		self.batch_size = args.batch_size

		self.weight_size = eval(args.layer_size)
		self.n_layers = len(self.weight_size)

		self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

		self.regs = eval(args.regs)
		self.decay = self.regs[0]

		self.verbose = args.verbose

		'''
		*********************************************************
		Create Placeholder for Input Data & Dropout.
		'''
		# sparse feature
		# self.users = tf.placeholder(tf.int32, shape=(None,))
		# self.items = tf.placeholder(tf.int32, shape=(None,))

		# user_feats = ['age', 'gender', 'provinceid', 'cityid', 'degree']
		# item_feats = ['advertiser_id', 'creative_id', 'product_id', 'product_type', 'campaign_id', 'ams_industry_level_one_id', 'ams_industry_level_two_id']
		# context_feats = ['adpos', 'device_brand_id', 'device_alias_id']
		# id_feats = ['user_lid', 'adid']

		# self.sp_feat_ph_dic ={feat.name: tf.placeholder(tf.int32, shape=(None,))
		#                  for i, feat in enumerate(self.fd)}
		self.id_feats_ph_dic = {feat.name: tf.placeholder(tf.int32, shape=(None,))
								for i, feat in enumerate(self.fd['id'])}
		self.user_feats_ph_dic = {feat.name: tf.placeholder(tf.int32, shape=(None,))
								  for i, feat in enumerate(self.fd['user'])}
		self.item_feats_ph_dic = {feat.name: tf.placeholder(tf.int32, shape=(None,))
								  for i, feat in enumerate(self.fd['item'])}
		self.context_feats_ph_dic = {feat.name: tf.placeholder(tf.int32, shape=(None,))
									 for i, feat in enumerate(self.fd['context'])}

		self.target_ph = tf.placeholder(tf.float32, [None, None])

		# structure feature
		self.campaign_ad_ph = tf.placeholder(tf.int32, shape=[None, None], name='campaign_ad_ph')
		self.advertiser_campaign_ph = tf.placeholder(tf.int32, shape=[None, None], name='advertiser_campaign_ph')



		# dropout: node dropout (adopted on the ego-networks);
		#          ... since the usage of node dropout have higher computational cost,
		#          ... please use the 'node_dropout_flag' to indicate whether use such technique.
		#          message dropout (adopted on the convolution operations).
		self.node_dropout_flag = args.node_dropout_flag
		self.node_dropout = tf.placeholder(tf.float32, shape=[None])
		self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

		"""
		*********************************************************
		Create Model Parameters (i.e., Initialize Weights).
		"""
		# initialization of model parameters
		self.weights = self._init_weights()


		"""
		*********************************************************
		Initialize Attribute Embeddings
		"""
		self.user_feats_emb_dic = {feat.name: tf.nn.embedding_lookup(self.weights[feat.name], self.user_feats_ph_dic[feat.name])
								   for i, feat in enumerate(self.fd['user'])}

		self.item_feats_emb_dic = {feat.name: tf.nn.embedding_lookup(self.weights[feat.name], self.item_feats_ph_dic[feat.name])
								   for i, feat in enumerate(self.fd['item'])}

		self.context_feats_emb_dic = {feat.name: tf.nn.embedding_lookup(self.weights[feat.name], self.context_feats_ph_dic[feat.name])
									  for i, feat in enumerate(self.fd['context'])}

		self.campaign_ad_emb = tf.nn.embedding_lookup(self.weights['adgroup_id'], self.campaign_ad_ph)
		self.advertiser_campaign_emb = tf.nn.embedding_lookup(self.weights['campaign_id'], self.advertiser_campaign_ph)


		"""
		*********************************************************
		Attribute Tree Aggregation
		"""
		if self.tree_type in ['gcn']:
			self.item_feats_emb_dic['campaign_id'] = self._tree_gcn_emb(self.item_feats_emb_dic['campaign_id'], self.campaign_ad_emb)
			self.item_feats_emb_dic['customer'] = self._tree_gcn_emb(self.item_feats_emb_dic['customer'], self.advertiser_campaign_emb)

		elif self.tree_type in ['ngcf']:
			self.item_feats_emb_dic['campaign_id'] = self._tree_ngcf_emb(self.item_feats_emb_dic['campaign_id'], self.campaign_ad_emb)
			self.item_feats_emb_dic['customer'] = self._tree_ngcf_emb(self.item_feats_emb_dic['customer'], self.advertiser_campaign_emb)

		elif self.tree_type in ['lightgcn']:
			self.item_feats_emb_dic['campaign_id'] = self._tree_lightgcn_emb(self.item_feats_emb_dic['campaign_id'], self.campaign_ad_emb)
			self.item_feats_emb_dic['customer'] = self._tree_lightgcn_emb(self.item_feats_emb_dic['customer'], self.advertiser_campaign_emb)

		elif self.tree_type in ['cp']:
			self.item_feats_emb_dic['campaign_id'] = self._tree_cp_emb(self.item_feats_emb_dic['campaign_id'], self.campaign_ad_emb)
			self.item_feats_emb_dic['customer'] = self._tree_cp_emb(self.item_feats_emb_dic['customer'], self.advertiser_campaign_emb)


		"""
		*********************************************************
		Compute Graph-based Representations of all users & items
		"""
		if self.alg_type in ['ngcf']:
			self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

		elif self.alg_type in ['gcn']:
			self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

		elif self.alg_type in ['gcmc']:
			self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

		"""
		*********************************************************
		Establish the final representations for user-item pairs in batch.
		"""
		self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.id_feats_ph_dic['userid'])
		self.i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.id_feats_ph_dic['adgroup_id'])

		"""
		*********************************************************
		User Intent and Item Intent modeling
		"""

		uid_emb = self._intent_modeling(self.u_g_embeddings, self.i_g_embeddings, self.user_feats_emb_dic)
		iid_emb = self._intent_modeling(self.i_g_embeddings, self.u_g_embeddings, self.item_feats_emb_dic)
		self.u_g_embeddings = uid_emb
		self.i_g_embeddings = iid_emb

		"""
		*********************************************************
		Inference for the testing phase.
		"""
		# self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)
		all_emb_list = []
		all_emb_list.append(tf.expand_dims(self.u_g_embeddings, 1))
		all_emb_list.append(tf.expand_dims(self.i_g_embeddings, 1))
		id_emb = tf.concat([self.u_g_embeddings, self.i_g_embeddings], 1)
		# user_emb = tf.concat([self.user_feats_emb_dic[key] for key in self.user_feats_emb_dic], 1)
		# item_emb = tf.concat([self.item_feats_emb_dic[key] for key in self.item_feats_emb_dic], 1)
		# context_emb = tf.concat([self.context_feats_emb_dic[key] for key in self.context_feats_emb_dic], 1)
		shape1 = tf.shape(self.user_feats_emb_dic['age_level'])[0]
		shape2 = tf.shape(self.user_feats_emb_dic['age_level'])[1]
		user_emb = tf.zeros([shape1, shape2])
		item_emb = tf.zeros([shape1, shape2])
		context_emb = tf.zeros([shape1, shape2])
		for value in self.user_feats_emb_dic.values():
			user_emb += value
			all_emb_list.append(tf.expand_dims(value, 1))
		for value in self.item_feats_emb_dic.values():
			item_emb += value
			all_emb_list.append(tf.expand_dims(value, 1))
		for value in self.context_feats_emb_dic.values():
			context_emb += value
			all_emb_list.append(tf.expand_dims(value, 1))

		# FM part
		all_sum_emb = self.u_g_embeddings + self.i_g_embeddings + user_emb + item_emb + context_emb
		all_sum_emb_square = tf.square(all_sum_emb)

		squared_emb = tf.square(tf.concat(all_emb_list, 1))
		squared_sum_emb = tf.reduce_sum(squared_emb, 1)

		FM_emb = 0.5 * tf.subtract(all_sum_emb_square, squared_sum_emb)
		FM_out = tf.reduce_sum(FM_emb, 1, keep_dims=True)

		# Deep
		all_emb = tf.concat([id_emb, user_emb, item_emb, context_emb], 1)
		dnn1 = tf.layers.dense(all_emb, 200, activation=tf.nn.relu, name='f1')
		dnn2 = tf.layers.dense(dnn1, 80, activation=tf.nn.relu, name='f2')
		dnn3 = tf.layers.dense(dnn2, 1, activation=tf.nn.relu, name='f3')

		out = FM_out + dnn3
		dnn_o = tf.layers.dense(out, 2, activation=tf.nn.sigmoid, name='f4')

		self.y_hat = tf.nn.softmax(dnn_o) + 0.00000001

		"""
		*********************************************************
		Generate Predictions & Optimize via cross entropy loss.
		"""
		# self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
		# 																  self.pos_i_g_embeddings,
		# 																  self.neg_i_g_embeddings)
		# self.loss = self.mf_loss + self.emb_loss + self.reg_loss
		self.loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)

		# self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
		self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.loss)

	def train(self, sess, inps):
		probs, loss, _ = sess.run([self.y_hat, self.loss, self.opt], feed_dict={
			self.target_ph: inps[0],
			self.id_feats_ph_dic['userid']: inps[1],
			self.id_feats_ph_dic['adgroup_id']: inps[2],
			self.user_feats_ph_dic['cms_segid']: inps[3],
			self.user_feats_ph_dic['cms_group_id']: inps[4],
			self.user_feats_ph_dic['final_gender_code']: inps[5],
			self.user_feats_ph_dic['age_level']: inps[6],
			self.user_feats_ph_dic['pvalue_level']: inps[7],
			self.user_feats_ph_dic['shopping_level']: inps[8],
			self.user_feats_ph_dic['occupation']: inps[9],
			self.user_feats_ph_dic['new_user_class_level']: inps[10],
			self.item_feats_ph_dic['campaign_id']: inps[11],
			self.item_feats_ph_dic['customer']: inps[12],
			self.item_feats_ph_dic['cate_id']: inps[13],
			self.item_feats_ph_dic['brand']: inps[14],
			self.context_feats_ph_dic['pid']: inps[15],
			self.campaign_ad_ph: inps[16],
			self.advertiser_campaign_ph: inps[17],
			self.node_dropout: inps[18],
			self.mess_dropout: inps[19]
		})
		return probs, loss

	def calculate(self, sess, inps):
		probs, loss = sess.run([self.y_hat, self.loss], feed_dict={
			self.target_ph: inps[0],
			self.id_feats_ph_dic['userid']: inps[1],
			self.id_feats_ph_dic['adgroup_id']: inps[2],
			self.user_feats_ph_dic['cms_segid']: inps[3],
			self.user_feats_ph_dic['cms_group_id']: inps[4],
			self.user_feats_ph_dic['final_gender_code']: inps[5],
			self.user_feats_ph_dic['age_level']: inps[6],
			self.user_feats_ph_dic['pvalue_level']: inps[7],
			self.user_feats_ph_dic['shopping_level']: inps[8],
			self.user_feats_ph_dic['occupation']: inps[9],
			self.user_feats_ph_dic['new_user_class_level']: inps[10],
			self.item_feats_ph_dic['campaign_id']: inps[11],
			self.item_feats_ph_dic['customer']: inps[12],
			self.item_feats_ph_dic['cate_id']: inps[13],
			self.item_feats_ph_dic['brand']: inps[14],
			self.context_feats_ph_dic['pid']: inps[15],
			self.campaign_ad_ph: inps[16],
			self.advertiser_campaign_ph: inps[17],
			self.node_dropout: inps[18],
			self.mess_dropout: inps[19]
		})
		return probs, loss

	def _init_weights(self):
		all_weights = dict()

		initializer = tf.contrib.layers.xavier_initializer()

		if self.pretrain_data is None:
			for key in self.fd:
				for i, feat in enumerate(self.fd[key]):
					all_weights[feat.name] = tf.Variable(initializer([feat.dimension, self.emb_dim]), name=feat.name)
			# all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
			# all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
			print('using xavier initialization')
		# else:
		#     all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
		#                                                 name='user_embedding', dtype=tf.float32)
		#     all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
		#                                                 name='item_embedding', dtype=tf.float32)
		#     print('using pretrained initialization')



		self.weight_size_list = [self.emb_dim] + self.weight_size

		for k in range(self.n_layers):
			all_weights['W_gc_%d' %k] = tf.Variable(
				initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
			all_weights['b_gc_%d' %k] = tf.Variable(
				initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

			all_weights['W_bi_%d' % k] = tf.Variable(
				initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
			all_weights['b_bi_%d' % k] = tf.Variable(
				initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

			all_weights['W_mlp_%d' % k] = tf.Variable(
				initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
			all_weights['b_mlp_%d' % k] = tf.Variable(
				initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

		return all_weights

	def _tree_gcn_emb(self, h, ch):
		# h (None, K), ch (None, None, K)
		# f(h,ch) = \sigma(W(h+ch))
		initializer = tf.contrib.layers.xavier_initializer()
		w1 = tf.Variable(initializer([self.emb_dim, self.emb_dim]), name='gcn_w')
		sum_emb = h + tf.reduce_sum(ch, axis=1)
		res = tf.nn.leaky_relu(tf.matmul(sum_emb, w1))
		return res

	def _tree_ngcf_emb(self, h, ch):
		# h (None, K), ch (None, None, K)
		# f(h, ch) = \sigma(W3*h + \sum_i\in ch(W1*i + W2(h \odot i)))
		initializer = tf.contrib.layers.xavier_initializer()
		w1 = tf.Variable(initializer([self.emb_dim, self.emb_dim]), name='ngcf_w1')
		w2 = tf.Variable(initializer([self.emb_dim, self.emb_dim]), name='ngcf_w2')
		w3 = tf.Variable(initializer([self.emb_dim, self.emb_dim]), name='ngcf_w3')
		h1 = tf.expand_dims(h, 1)
		h1 = tf.tile(h1, multiples=[1, tf.shape(ch)[1], 1])
		mul1 = tf.tensordot(h1*ch, w2, axes=1)
		s1 = tf.reduce_sum(tf.tensordot(ch, w1, axes=1) + mul1, axis=1)
		res = tf.nn.leaky_relu(tf.matmul(h, w3) + s1)
		return res

	def _tree_lightgcn_emb(self, h, ch):
		# h (None, K), ch (None, None, K)
		# f(h, ch) = \sum ch
		res = tf.reduce_sum(ch, axis=1)
		return res

	def _tree_cp_emb(self, h, ch):
		# h (None, K), ch (None, None, K)
		# f(h, ch) = \sigma(W3*h + \sum_i\in ch(W1*i + W2(h \odot i) + W4(h \oplus i)))
		initializer = tf.contrib.layers.xavier_initializer()
		w1 = tf.Variable(initializer([self.emb_dim, self.emb_dim]), name='ngcf_w1')
		w2 = tf.Variable(initializer([self.emb_dim, self.emb_dim]), name='ngcf_w2')
		w3 = tf.Variable(initializer([self.emb_dim, self.emb_dim]), name='ngcf_w3')
		w4 = tf.Variable(initializer([self.emb_dim*2, self.emb_dim]), name='ngcf_w4')
		h1 = tf.expand_dims(h, 1)
		h1 = tf.tile(h1, multiples=[1, tf.shape(ch)[1], 1])
		mul1 = tf.tensordot(h1*ch, w2, axes=1)
		h2 = tf.concat([h1, ch], axis=-1)
		mul2 = tf.tensordot(h2, w4, axes=1)
		s1 = tf.reduce_sum(tf.tensordot(ch, w1, axes=1) + mul1 + mul2, axis=1)
		res = tf.nn.leaky_relu(tf.matmul(h, w3) + s1)
		return res

	# calculate target intent for attributes of base
	def _intent_modeling(self, base, target, attributes):
		# base/target (None, K), attributes: dict (None, K)
		att_emb = tf.concat([tf.expand_dims(value, 1) for value in attributes.values()], 1) #(None, m, K)
		target = tf.tile(tf.expand_dims(target, 1), multiples=[1, tf.shape(att_emb)[1], 1]) #(None, m, K)
		weight_emb = tf.reduce_sum(target*att_emb, 2) # (None, m)
		weight_emb = tf.nn.softmax(logits=weight_emb)
		weight_emb = tf.expand_dims(weight_emb, 1) # (None, 1, m)

		res_emb = tf.matmul(weight_emb, att_emb) # (None, 1, K)
		res_emb = tf.squeeze(res_emb, 1)

		return res_emb + base



	def _split_A_hat(self, X):
		A_fold_hat = []

		fold_len = (self.n_users + self.n_items) // self.n_fold
		for i_fold in range(self.n_fold):
			start = i_fold * fold_len
			if i_fold == self.n_fold -1:
				end = self.n_users + self.n_items
			else:
				end = (i_fold + 1) * fold_len

			A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
		return A_fold_hat

	def _split_A_hat_node_dropout(self, X):
		A_fold_hat = []

		fold_len = (self.n_users + self.n_items) // self.n_fold
		for i_fold in range(self.n_fold):
			start = i_fold * fold_len
			if i_fold == self.n_fold -1:
				end = self.n_users + self.n_items
			else:
				end = (i_fold + 1) * fold_len

			# A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
			temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
			n_nonzero_temp = X[start:end].count_nonzero()
			A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

		return A_fold_hat

	def _create_ngcf_embed(self):
		# Generate a set of adjacency sub-matrix.
		if self.node_dropout_flag:
			# node dropout.
			A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
		else:
			A_fold_hat = self._split_A_hat(self.norm_adj)

		ego_embeddings = tf.concat([self.weights['user_lid'], self.weights['adid']], axis=0)

		all_embeddings = [ego_embeddings]

		for k in range(0, self.n_layers):

			temp_embed = []
			for f in range(self.n_fold):
				temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

			# sum messages of neighbors.
			side_embeddings = tf.concat(temp_embed, 0)
			# transformed sum messages of neighbors.
			sum_embeddings = tf.nn.leaky_relu(
				tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

			# bi messages of neighbors.
			bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
			# transformed bi messages of neighbors.
			bi_embeddings = tf.nn.leaky_relu(
				tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

			# non-linear activation.
			ego_embeddings = sum_embeddings + bi_embeddings

			# message dropout.
			ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

			# normalize the distribution of embeddings.
			norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)
			all_embeddings += [norm_embeddings]

		# all_embeddings = tf.concat(all_embeddings, 1)
		all_embeddings = tf.reduce_sum(tf.concat([tf.expand_dims(e, 1) for e in all_embeddings], 1), 1)
		u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
		return u_g_embeddings, i_g_embeddings

	def _create_gcn_embed(self):
		A_fold_hat = self._split_A_hat(self.norm_adj)
		embeddings = tf.concat([self.weights['user_lid'], self.weights['adid']], axis=0)


		all_embeddings = [embeddings]

		for k in range(0, self.n_layers):
			temp_embed = []
			for f in range(self.n_fold):
				temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

			embeddings = tf.concat(temp_embed, 0)
			embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])
			embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

			all_embeddings += [embeddings]

		# all_embeddings = tf.concat(all_embeddings, 1)
		all_embeddings = tf.reduce_sum(tf.concat([tf.expand_dims(e, 1) for e in all_embeddings], 1), 1)
		u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
		return u_g_embeddings, i_g_embeddings

	def _create_gcmc_embed(self):
		A_fold_hat = self._split_A_hat(self.norm_adj)

		embeddings = tf.concat([self.weights['user_lid'], self.weights['adid']], axis=0)

		all_embeddings = []

		for k in range(0, self.n_layers):
			temp_embed = []
			for f in range(self.n_fold):
				temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
			embeddings = tf.concat(temp_embed, 0)
			# convolutional layer.
			embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
			# dense layer.
			mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' %k]) + self.weights['b_mlp_%d' %k]
			mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

			all_embeddings += [mlp_embeddings]

		# all_embeddings = tf.concat(all_embeddings, 1)
		all_embeddings = tf.reduce_sum(tf.concat([tf.expand_dims(e, 1) for e in all_embeddings], 1), 1)
		u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
		return u_g_embeddings, i_g_embeddings


	def create_bpr_loss(self, users, pos_items, neg_items):
		pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
		neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

		regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
		regularizer = regularizer/self.batch_size

		# In the first version, we implement the bpr loss via the following codes:
		# We report the performance in our paper using this implementation.
		maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
		mf_loss = tf.negative(tf.reduce_mean(maxi))

		## In the second version, we implement the bpr loss via the following codes to avoid 'NAN' loss during training:
		## However, it will change the training performance and training performance.
		## Please retrain the model and do a grid search for the best experimental setting.
		# mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_scores - neg_scores)))


		emb_loss = self.decay * regularizer

		reg_loss = tf.constant(0.0, tf.float32, [1])

		return mf_loss, emb_loss, reg_loss

	def _convert_sp_mat_to_sp_tensor(self, X):
		coo = X.tocoo().astype(np.float32)
		indices = np.mat([coo.row, coo.col]).transpose()
		return tf.SparseTensor(indices, coo.data, coo.shape)

	def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
		"""
		Dropout for sparse tensors.
		"""
		noise_shape = [n_nonzero_elems]
		random_tensor = keep_prob
		random_tensor += tf.random_uniform(noise_shape)
		dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
		pre_out = tf.sparse_retain(X, dropout_mask)

		return pre_out * tf.div(1., keep_prob)

def load_pretrained_data():
	pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
	try:
		pretrain_data = np.load(pretrain_path)
		print('load the pretrained embeddings.')
	except Exception:
		pretrain_data = None
	return pretrain_data

def eval_model(sess, test_data, model):
	loss_sum = 0.
	nums = 0
	stored_arr = []

	for step, (y, uid, adid, cms_segid, cms_group_id, gender, age, pvalue, shopping, occup, class_level, campaign_id,
			   customer, cate_id, brand, pid, campaign_ad, advertiser_cam) in enumerate(test_data, start=1):
		nums += 1
		prob, loss = model.calculate(sess, [y, uid, adid, cms_segid, cms_group_id, gender, age, pvalue, shopping, occup, class_level, campaign_id,
											customer, cate_id, brand, pid, campaign_ad, advertiser_cam,
											[0.]*len(eval(args.layer_size)), [0.]*len(eval(args.layer_size))])
		loss_sum += loss

		prob_1 = prob[:, 0].tolist()
		target_1 = y[:, 0].tolist()
		for p, t in zip(prob_1, target_1):
			stored_arr.append([p, t])
	test_auc = calc_auc(stored_arr)
	loss_sum = loss_sum / nums

	return test_auc, loss_sum

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

	config = dict()
	config['n_users'] = data_generator.n_users
	config['n_items'] = data_generator.n_items

	"""
	*********************************************************
	Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
	"""
	plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

	if args.adj_type == 'plain':
		config['norm_adj'] = plain_adj
		print('use the plain adjacency matrix')

	elif args.adj_type == 'norm':
		config['norm_adj'] = norm_adj
		print('use the normalized adjacency matrix')

	elif args.adj_type == 'gcmc':
		config['norm_adj'] = mean_adj
		print('use the gcmc adjacency matrix')

	else:
		config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
		print('use the mean adjacency matrix')

	t0 = time()

	if args.pretrain == -1:
		pretrain_data = load_pretrained_data()
	else:
		pretrain_data = None

	model = NGCF(fd, data_config=config, pretrain_data=pretrain_data)

	"""
	*********************************************************
	Save the model parameters.
	"""
	saver = tf.train.Saver()

	if args.save_flag == 1:
		layer = '-'.join([str(l) for l in eval(args.layer_size)])
		weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
															str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
		ensureDir(weights_save_path)
		save_saver = tf.train.Saver(max_to_keep=1)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)

	"""
	*********************************************************
	Reload the pretrained model parameters.
	"""
	if args.pretrain == 1:
		layer = '-'.join([str(l) for l in eval(args.layer_size)])

		pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
														str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))


		ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
		if ckpt and ckpt.model_checkpoint_path:
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, ckpt.model_checkpoint_path)
			print('load the pretrained model parameters from: ', pretrain_path)

			# *********************************************************
			# get the performance from pretrained model.
			if args.report != 1:
				users_to_test = list(data_generator.test_set.keys())
				ret = test(sess, model, users_to_test, drop_flag=True)
				cur_best_pre_0 = ret['recall'][0]

				pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
							   'ndcg=[%.5f, %.5f]' % \
							   (ret['recall'][0], ret['recall'][-1],
								ret['precision'][0], ret['precision'][-1],
								ret['hit_ratio'][0], ret['hit_ratio'][-1],
								ret['ndcg'][0], ret['ndcg'][-1])
				print(pretrain_ret)
		else:
			sess.run(tf.global_variables_initializer())
			cur_best_pre_0 = 0.
			print('without pretraining.')

	else:
		sess.run(tf.global_variables_initializer())
		cur_best_pre_0 = 0.
		print('without pretraining.')


	"""
	*********************************************************
	Train.
	"""
	test_iter = 100
	best_auc = 0.0
	best_auc_loss = 0.0
	no_change = 0 # 记录best_auc无变化的次数
	stop_flag = False # early stop flag
	for epoch in range(args.epoch):
		start_time = time()
		loss_sum = 0.0
		stored_arr = []
		for step, (y, uid, adid, cms_segid, cms_group_id, gender, age, pvalue, shopping, occup, class_level, campaign_id,
				   customer, cate_id, brand, pid, campaign_ad, advertiser_cam) in enumerate(train_data, start=1):
			probs, loss = model.train(sess, [y, uid, adid, cms_segid, cms_group_id, gender, age, pvalue, shopping, occup, class_level, campaign_id,
											 customer, cate_id, brand, pid, campaign_ad, advertiser_cam,
											 eval(args.node_dropout), eval(args.mess_dropout)])
			for p, t in zip(probs[:, 0].tolist(), y[:, 0].tolist()):
				stored_arr.append([p,t])

			loss_sum += loss
			sys.stdout.flush()
			if step % test_iter == 0:
				test_auc, test_loss = eval_model(sess, test_data, model)
				train_auc = calc_auc(stored_arr)
				print('Epoch: %d\tStep: %d\ttrain_loss: %.4f\ttrain_auc: %.4f\ttest_loss: %.4f\ttest_auc: %.4f' % \
					  (epoch, step, loss_sum / test_iter, train_auc, test_loss, test_auc))

				if best_auc < test_auc:
					best_auc = test_auc
					best_auc_loss = test_loss

				stored_arr = []
				loss_sum = 0.0

		print('Epoch %d DONE\tCost time: %.2f' % (epoch, time()-start_time))

	print('Best Test Result (Loss: %.4f, AUC: %.4f)' % (best_auc_loss, best_auc))

