import tensorflow as tf

class CNN_Relation(object):

	def __init__(self, num_classes, seq_len, word_dict_size, pos_dict_size, chunk_dict_size, d1_dict_size, d2_dict_size, type_dict_size, wv, batch_size = 50, w_emb_size=50, d1_emb_size=5, d2_emb_size=5, pos_emb_size=5, chunk_emb_size=5, dep_emb_size=5, type_emb_size=5, filter_sizes=[4,6], num_filters=100, l2_reg_lambda = 0.0):

		emb_size = w_emb_size + pos_emb_size + chunk_emb_size + d1_emb_size + d2_emb_size + type_emb_size  
#		emb_size = w_emb_size + d1_emb_size + d2_emb_size + type_emb_size  
#		emb_size = w_emb_size + type_emb_size  

		self.w  = tf.placeholder(tf.int32, [None, seq_len], name="x")
		self.pos = tf.placeholder(tf.int32, [None, seq_len], name="x1")
		self.chunk = tf.placeholder(tf.int32, [None, seq_len], name="x2")
		self.d1 = tf.placeholder(tf.int32, [None, seq_len], name="x3")
		self.d2 = tf.placeholder(tf.int32, [None, seq_len], name='x4')
		self.type = tf.placeholder(tf.int32, [None, seq_len], name='x5')
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		
		# Initialization
		#W_wemb =    tf.Variable(tf.random_uniform([word_dict_size, w_emb_size], -1.0, +1.0))
		W_wemb = tf.Variable(wv)
		W_posemb =  tf.Variable(tf.random_uniform([pos_dict_size, pos_emb_size], -1.0, +1.0))
		W_chunkemb =tf.Variable(tf.random_uniform([chunk_dict_size, chunk_emb_size], -1.0, +1.0))
		W_d1emb =   tf.Variable(tf.random_uniform([d1_dict_size, d1_emb_size], -1.0, +1.0))
		W_d2emb =   tf.Variable(tf.random_uniform([d2_dict_size, d2_emb_size], -1.0, +1.0))
		W_typeemb = tf.Variable(tf.random_uniform([type_dict_size, type_emb_size], -1.0, +1.0))
 
		
		# Embedding layer
		emb0 = tf.nn.embedding_lookup(W_wemb, self.w)				#word embedding
		emb1 = tf.nn.embedding_lookup(W_posemb, self.pos)			#position from first entity embedding
		emb2 = tf.nn.embedding_lookup(W_chunkemb, self.chunk)			#position from second entity embedding
		emb3 = tf.nn.embedding_lookup(W_d1emb, self.d1)				#POS embedding
		emb4 = tf.nn.embedding_lookup(W_d2emb, self.d2)				#POS embedding
		emb5 = tf.nn.embedding_lookup(W_typeemb, self.type)			#POS embedding
 

		X = tf.concat(2, [emb0, emb1, emb2, emb3, emb4, emb5])			#shape(?, 21, 80)
#		X = tf.concat(2, [emb0, emb3, emb4, emb5])
#		X = tf.concat(2, [emb0, emb5])
		X_expanded = tf.expand_dims(X, -1) 					#shape (?, 21, 80, 1)

		l2_loss = tf.constant(0.0)
		
		# CNN+Maxpooling Layer
		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			filter_shape = [filter_size, emb_size, 1, num_filters]
			W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
			b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
			conv = tf.nn.conv2d(X_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
        		# Apply nonlinearity
			h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") 		#shape (?, 19, 1, 70)
			# print "h ", h.get_shape

			# Maxpooling over the outputs
			pooled = tf.nn.max_pool(h, ksize=[1, seq_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
			# print "pooled", pooled.get_shape				#shape=(?, 1, 1, 70)

			pooled_outputs.append(pooled)

		print "pooled_outputs", len(pooled_outputs)

		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		h_pool = tf.concat(3, pooled_outputs)					#shape= (?, 1, 1, 210)
		print "h_pool", h_pool.get_shape
		h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])		#shape =(?, 210)
		print "h_pool_flate", h_pool_flat.get_shape

		# dropout layer	 
		h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

		# Fully connetected layer
		W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
		l2_loss += tf.nn.l2_loss(W)
		l2_loss += tf.nn.l2_loss(b)
		scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
	
		# prediction and loss function
		self.predictions = tf.argmax(scores, 1, name="predictions")
		self.losses = tf.nn.softmax_cross_entropy_with_logits(scores, self.input_y)
        	self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss

        	# Accuracy
        	self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        	self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")	
 
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		self.sess = tf.Session(config=session_conf)  

		self.optimizer = tf.train.AdamOptimizer(1e-2)

		self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

		self.sess.run(tf.initialize_all_variables())

	def train_step(self, W_batch, pos_batch, chunk_batch, d1_batch, d2_batch, t_batch, y_batch):
    		feed_dict = {
				self.w 		:W_batch,
				self.pos	:pos_batch, 
				self.chunk	:chunk_batch,
				self.d1		:d1_batch,
				self.d2		:d2_batch,
				self.type	:t_batch,
				self.dropout_keep_prob: 0.5,
				self.input_y 	:y_batch
	    			}
   		_, step, loss, accuracy, predictions = self.sess.run([self.train_op, self.global_step, self.loss, self.accuracy, self.predictions], feed_dict)
    		print ("step "+str(step) + " loss "+str(loss) +" accuracy "+str(accuracy))


	def test_step(self, W_batch, pos_batch, chunk_batch, d1_batch, d2_batch, t_batch, y_batch):
    		feed_dict = {
				self.w 		:W_batch,
				self.pos	:pos_batch, 
				self.chunk	:chunk_batch,
				self.d1		:d1_batch,
				self.d2 	:d2_batch,
				self.type	:t_batch,
				self.dropout_keep_prob:1.0,
				self.input_y 	:y_batch
	    		}
   		step, loss, accuracy, predictions = self.sess.run([self.global_step, self.loss, self.accuracy, self.predictions], feed_dict)
    		print "Accuracy in test data", accuracy
		return accuracy, predictions

	

