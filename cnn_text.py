
import numpy as np
import tensorflow as tf

class CNN_Relation(object):		 
			 
   def __init__(self, seq_len, num_classes, vocab_size, pos_dict_size, p1_dict_size, p2_dict_size, type_dict_size, w_emb_size, p1_emb_size, p2_emb_size, pos_emb_size, type_emb_size, filter_sizes, num_filters, l2_reg_lambda = 0.0):

	emb_size = w_emb_size + p1_emb_size + p2_emb_size + pos_emb_size + type_emb_size
 
	self.x  = tf.placeholder(tf.int32, [None, seq_len], name="x")
	self.x1 = tf.placeholder(tf.int32, [None, seq_len], name="x1")
	self.x2 = tf.placeholder(tf.int32, [None, seq_len], name="x2")
	self.x3 = tf.placeholder(tf.int32, [None, seq_len], name="x3")
	self.x4 = tf.placeholder(tf.int32, [None, seq_len], name='x4')

	self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
	self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

	W_wemb =  tf.Variable(tf.random_uniform([vocab_size, w_emb_size], -1.0, +1.0))
	W_p1emb = tf.Variable(tf.random_uniform([p1_dict_size, p1_emb_size], -1.0, +1.0))
	W_p2emb = tf.Variable(tf.random_uniform([p2_dict_size, p2_emb_size], -1.0, +1.0))
	W_posemb = tf.Variable(tf.random_uniform([pos_dict_size, pos_emb_size], -1.0, +1.0))
	W_temb = tf.Variable(tf.random_uniform([type_dict_size, type_emb_size], -1.0, +1.0))	

	#Embedding layer
	emb = tf.nn.embedding_lookup(W_wemb, self.x)			#word embedding
	emb1 = tf.nn.embedding_lookup(W_p1emb, self.x1)			#position from first entity embedding
	emb2 = tf.nn.embedding_lookup(W_p2emb, self.x2)			#position from second entity embedding
	emb3 = tf.nn.embedding_lookup(W_posemb, self.x3)		#POS embedding
	emb4 = tf.nn.embedding_lookup(W_temb, self.x4)		#POS embedding

	X = tf.concat(2, [emb, emb1, emb2, emb3, emb4])			#shape(?, 21, 100)

	X_expanded = tf.expand_dims(X, -1) 				#shape (?, 21, 100, 1)

	l2_loss = tf.constant(0.0)
	
	#CNN+Maxpooling Layer
	pooled_outputs = []
	for i, filter_size in enumerate(filter_sizes):
		filter_shape = [filter_size, emb_size, 1, num_filters]
		W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
		conv = tf.nn.conv2d(
                    X_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
        	# Apply nonlinearity
		h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") 	#shape (?, 19, 1, 70)
		# print "h ", h.get_shape

		# Maxpooling over the outputs
		pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, seq_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
		# print "pooled", pooled.get_shape			#shape=(?, 1, 1, 70)

		pooled_outputs.append(pooled)

	print "pooled_outputs", len(pooled_outputs)

	# Combine all the pooled features
	num_filters_total = num_filters * len(filter_sizes)
	h_pool = tf.concat(3, pooled_outputs)				#shape= (?, 1, 1, 210)
	print "h_pool", h_pool.get_shape
	h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])	#shape =(?, 210)
	print "h_pool_flate", h_pool_flat.get_shape

	#dropout layer	 
	h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

	#Fully connetecte layer
	W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
	b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
	l2_loss += tf.nn.l2_loss(W)
	l2_loss += tf.nn.l2_loss(b)
	scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
	
	#prediction and loss function
	self.predictions = tf.argmax(scores, 1, name="predictions")
	self.losses = tf.nn.softmax_cross_entropy_with_logits(scores, self.input_y)
        self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss

        # Accuracy
        self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")



