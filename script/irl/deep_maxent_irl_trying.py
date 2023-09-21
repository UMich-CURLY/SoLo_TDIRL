from random import randint
import numpy as np
import tensorflow as tf
import mdp.gridworld as gridworld
import mdp.value_iteration as value_iteration
import img_utils
import tf_utils
import tensorflow as tf
from utils import *
import matplotlib.pyplot as plt
import time
from IPython import embed
try:
  from StringIO import StringIO
except:
  from io import StringIO
import collections

class DeepIRLFC:
  def __init__(self, n_input, feature_map_size, lr, n_h1=400, n_h2=300, l2=10, name='deep_irl_fc'):
    self.n_input = n_input
    self.lr = lr
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.name = name
    self.feature_map_size = feature_map_size
    print("Feature map size is ", self.feature_map_size)
    self.sess = tf.Session()
    self.input_s, self.reward, self.theta = self._build_network(self.name)
    
    step_rate = 1000
    decay = 0.95

    global_step = tf.Variable(0, trainable=False)
    increment_global_step = tf.assign(global_step, global_step + 1)
    self.lr = tf.train.exponential_decay(lr, global_step, step_rate, decay, staircase=True)

    # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
    self.optimizer = tf.train.GradientDescentOptimizer(lr)
    
    self.grad_r = tf.placeholder(tf.float32, [None, 1])
    a = [1,2,3]
    self.dummy_name = tf.convert_to_tensor(a)
    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
    self.grad_l2 = tf.gradients(self.l2_loss, self.theta)

    self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)
    # apply l2 loss gradients
    self.grad_theta = [tf.add(l2*self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
    self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)

    self.grad_norms = tf.global_norm(self.grad_theta)
    self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))
    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver()
    self.weight_path = ""
    self.weight_folder = ""


  def _build_network(self, name):
    input_s = tf.placeholder(tf.float32, [self.n_input, self.feature_map_size])
    with tf.variable_scope(name):
      fc1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.relu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      fc2 = tf_utils.fc(fc1, self.n_h2, scope="fc2", activation_fn=tf.nn.relu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      reward = tf_utils.fc(fc2, 1, scope="reward")
    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    print(reward)
    embed()
    return input_s, reward, theta


  def get_theta(self):
    theta = self.sess.run(self.theta)
    self.saver.save(self.sess, "../weights2/saved_weights")
    return theta

  def get_theta_no_loss(self):
    theta = self.sess.run(self.theta)
    self.saver.save(self.sess, self.weight_path)
    return theta


  def get_rewards(self, states):
    rewards = self.sess.run(self.reward, feed_dict={self.input_s: states})
    return rewards


  def apply_grads(self, feat_map, grad_r):
    grad_r = np.reshape(grad_r, [-1, 1])
    feat_map = np.reshape(feat_map, [-1, self.feature_map_size])
    _, grad_theta, l2_loss, grad_norms, name = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms, self.dummy_name], 
      feed_dict={self.grad_r: grad_r, self.input_s: feat_map})
    return grad_theta, l2_loss, grad_norms

  def load_weights(self):
    # with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(self.weight_path+'.meta')
    new_saver.restore(self.sess, tf.train.latest_checkpoint(self.weight_folder))

  # def save_weights(self):
  #   self.theta.

class DeepIRLConv:
  def __init__(self, n_input, feature_map_size, lr, n_h1=400, n_h2=300, l2=10, name='deep_irl_conv'):
    self.n_input = n_input
    self.lr = lr
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.name = name
    self.width = np.sqrt(self.n_input)
    self.height = np.sqrt(self.n_input)
    
    self.feature_map_size = feature_map_size
    print("Feature map size is ", self.feature_map_size)
    self.sess = tf.Session()
    self.input_s, self.reward, self.theta = self._build_network(self.name)
    # self.optimizer = tf.train.GradientDescentOptimizer(lr)
    self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # self.grad_r = tf.placeholder(tf.float32, [None, self.height, self.width, 1])
    self.grad_r = tf.placeholder(tf.float32, [None, self.n_input])
    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
    self.grad_l2 = tf.gradients(self.l2_loss, self.theta)
    self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)
    # apply l2 loss gradients
    self.grad_theta = [tf.add(l2*self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
    self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)

    self.grad_norms = tf.global_norm(self.grad_theta)
    self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))
    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver()
    self.weight_path = ""
    self.weight_folder = ""


  def conv_layer(self, x, input_channel, hidden_size, name="Conv"):
    with tf.variable_scope(name):
      conv1 = tf.layers.conv2d(inputs=x, filters=hidden_size, kernel_size=(3,3),padding="same")
      batch_norm1 = tf.layers.batch_normalization(conv1)
    return tf.nn.relu(batch_norm1)

  def _build_network(self, name):

    input_s = tf.placeholder(tf.float32, [None, self.width, self.height, self.feature_map_size])
    with tf.variable_scope(name):
      conv1 = tf_utils.conv2d(input_s, self.n_h1, (2, 2), 1)
      conv2 = tf_utils.conv2d(conv1, self.n_h2, (1, 1), 1)
      conv3 = tf_utils.conv2d(conv2, self.n_h2, (1, 1), 1)
      reward = tf_utils.conv2d(conv3, 1, (1, 1), 1)
    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    reward1 = tf.reshape(reward, (-1, self.n_input))
    # input_s = tf.placeholder(tf.float32, [1, self.feature_map_size, self.feature_map_size , self.n_input])
    # print(input_s)
    # # input_s = tf.placeholder(tf.float32, [None, self.n_input])
    # hidden_size1 = 32
    # hidden_size2 = 64
    # with tf.variable_scope(name):
    #   # Block1
    #   conv1 = self.conv_layer(input_s, self.n_input, hidden_size1, name="Conv1")
    #   conv2 = self.conv_layer(conv1, hidden_size1, hidden_size1, name="Conv2")
    #   conv3 = self.conv_layer(conv2, hidden_size1, hidden_size1, name="Conv3")
    #   max_pool1 = tf.layers.max_pooling2d(inputs=conv3,pool_size=2,strides=2,name="Pool1")
    #   print(max_pool1)
    #   # Block2
    #   conv4 = self.conv_layer(max_pool1, hidden_size1, hidden_size2, name="Conv4")
    #   conv5 = self.conv_layer(conv4, hidden_size2, hidden_size2, name="Conv5")
    #   conv6 = self.conv_layer(conv5, hidden_size2, hidden_size2, name="Conv6")
    #   max_pool2 = tf.layers.max_pooling2d(inputs=conv6,pool_size=2,strides=2,name="Pool2")
    #   print(max_pool2)
    #   # Block3
    #   unpooling1 = tf.contrib.layers.conv2d_transpose(max_pool2,num_outputs=hidden_size2,kernel_size=(2,2),stride=2,padding="valid")
    #   conv7 = self.conv_layer(unpooling1, hidden_size2, hidden_size2, name="Conv7")
    #   conv8 = self.conv_layer(conv7, hidden_size2, hidden_size2, name="Conv8")
    #   conv9 = self.conv_layer(conv8, hidden_size2, hidden_size2, name="Conv9")
    #   print(conv9)
    #   # Block4
    #   unpooling2 = tf.contrib.layers.conv2d_transpose(conv9, num_outputs=hidden_size2,kernel_size=(2,2),stride=2,padding="valid")
    #   conv10 = self.conv_layer(unpooling2, hidden_size1, hidden_size1, name="Conv10")
    #   conv11 = self.conv_layer(conv10, hidden_size1, hidden_size1, name="Conv11")
    #   conv12 = self.conv_layer(conv11, hidden_size1, hidden_size1, name="Conv12")
    #   print(conv12)

    #   # Output
    #   reward = self.conv_layer(conv12, 1, 1, name="Output")
    print(reward)
    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    return input_s, reward1, theta


  def get_theta(self):
    theta = self.sess.run(self.theta)
    self.saver.save(self.sess, "../weights2/saved_weights")
    return theta

  def get_theta_no_loss(self):
    theta = self.sess.run(self.theta)
    self.saver.save(self.sess, self.weight_path)
    return theta


  def get_rewards(self, states):
    rewards = self.sess.run(self.reward, feed_dict={self.input_s: states})
    return rewards


  def apply_grads(self, feat_map, grad_r):
    grad_r = np.reshape(grad_r, [-1, self.n_input])
    feat_map = np.reshape(feat_map, [-1, int(self.height), int(self.width), self.feature_map_size])
    _, grad_theta, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms], 
      feed_dict={self.grad_r: grad_r, self.input_s: feat_map})
    return grad_theta, l2_loss, grad_norms

  def load_weights(self):
    # with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(self.weight_path+'.meta')
    new_saver.restore(self.sess, tf.train.latest_checkpoint(self.weight_folder))
  

def compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True):
  """compute the expected states visition frequency p(s| theta, T) 
  using dynamic programming

  inputs:
    P_a     NxNxN_ACTIONS matrix - transition dynamics
    gamma   float - discount factor
    trajs   list of list of Steps - collected from expert
    policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

  returns:
    p       Nx1 vector - state visitation frequencies
  """
  N_STATES, _, N_ACTIONS = np.shape(P_a)

  T = min([len(trajs[i]) for i in range(len(trajs))])
  # mu[s, t] is the prob of visiting state s at time t
  mu = np.zeros([N_STATES, T])

  for traj in trajs:
    mu[traj[0].cur_state, 0] += 1
  mu[:,0] = mu[:,0]/len(trajs)
  
  for t in range(T-1):
    for s in range(N_STATES):
      if deterministic:
        mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
      else:
        mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES)])
  p = np.sum(mu, 1)
  # print(p.shape)
  return p

def demo_svf(trajs, n_states):
  """
  compute state visitation frequences from demonstrations
  
  input:
    trajs   list of list of Steps - collected from expert
  returns:
    p       Nx1 vector - state visitation frequences   
  """
  # [[Step1, Step2]]
  p = np.zeros(n_states)
  for traj in trajs:
    for step in traj:
      # print(step)
      p[step.cur_state] += 1
  p = p/len(trajs)
  return p



def get_reward_sum_from_policy(reward, policy, gridsize):
  total_reward = 0

  # policy_dict = {0: 'r', 1: 'l', 2: 'u', 3: 's'}

  # policy = [policy_dict[int(i)] for i in policy]

  policy = np.array(policy).reshape(gridsize[1],gridsize[0])

  direct = {'r':np.array([0, 1]), 'l':np.array([0, -1]), 'u':np.array([-1, 0]), 'd':np.array([1, 0]), 's':np.array([0, 0])}
  
  current_pose = np.array([2,1])

  first_act = policy[current_pose[0]][current_pose[1]]

  total_reward += reward[current_pose[0]*gridsize[0] + current_pose[1]][0]

  count = 0

  while(count < gridsize[0] * gridsize[1] - 1):
      
      next_goal = current_pose + direct[first_act]
      if next_goal[0] >= gridsize[0] or next_goal[1] >= gridsize[1] or \
            next_goal[0] < 0 or next_goal[1] < 0:
            break

      total_reward += reward[current_pose[0]*gridsize[0] + current_pose[1]][0]

      current_pose = next_goal

      first_act = policy[current_pose[0]][current_pose[1]]
      # Open loop maybe closed loop later
      count += 1
  
  return total_reward
  # self.path_pub.publish(self.irl_path)



def deep_maxent_irl_no_traj_loss_tribhi(feat_maps, P_a, gamma, trajs,  lr, n_iters, weight_path = None):
  """
  Maximum Entropy Inverse Reinforcement Learning (Maxent IRL) 
  
  Add trajectory ranking loss batchsize=2

  inputs:
    feat_map    NxD matrix - the features for each state
    P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of 
                                       landing at state s1 when taking action 
                                       a at state s0
    gamma       float - RL discount factor
    trajs       a list of demonstrations
    lr          float - learning rate
    n_iters     int - number of optimization steps

  returns
    rewards     Nx1 vector - recoverred state rewards
  """


  '''
  trajs = [[Step(cur_state=7.0, next_state=4.0), Step(cur_state=4.0, next_state=1.0)], 
            [Step(cur_state=7.0, next_state=4.0), Step(cur_state=4.0, next_state=5.0)]]
  fms = [array([[0, 0, 1, 0, 0, 1, 0, 0, 0],
      [1, 1, 0, 1, 1, 0, 0, 1, 1],
      [0, 0, 0, 0, 0, 0, 1, 0, 0],
      [1, 1, 1, 1, 1, 1, 0, 1, 1],
      [0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 0, 0]]), 
      
      array([[0, 0, 1, 0, 0, 1, 0, 0, 0],
      [1, 1, 0, 0, 1, 0, 0, 1, 1],
      [0, 0, 0, 1, 0, 0, 1, 0, 0],
      [1, 1, 1, 1, 1, 1, 1, 1, 1],
      [0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0]])]
'''
  # tf.set_random_seed(1)
  # Just for testing
  # feat_maps = feat_maps[:2]
  # trajs = trajs[:2]
  # feat_maps[:] = feat_maps[:][:2]
  # feat_maps[:][0] = [-e for e in feat_maps[:][0]]
  # print(np.array(feat_maps[1]).T)


  N_STATES, _, N_ACTIONS = np.shape(P_a)
  # print(N_STATES, N_ACTIONS)

  # init nn model
  print("Number of feature: ", feat_maps[0].shape[0])
  # nn_r = DeepIRLFC(feat_maps[0].shape[0], feat_maps[0].shape[1], lr, 3, 2)
  nn_r = DeepIRLConv(feat_maps[0].shape[1], feat_maps[0].shape[0], lr, 64, 32)
  if(weight_path):
    nn_r.weight_path = weight_path
  # Hight and width of the feature map. (Assume the grid is a square)
  hight = int(np.sqrt(feat_maps[0].shape[1]))
  width = hight
  n_feats = feat_maps[0].shape[0]

  # find state visitation frequencies using demonstrations
  
  # training 
  # print(trajs)

  train_summary_writer = tf.summary.FileWriter(weight_path+"/logs1")

  train_summary_writer.flush()
  
  prev_l2_loss = 1e30
  l2_loss = 0
  grad_r1 = []
  batch = [50,50,50,44]
  n_epochs = 1000
  full_grad_rolling = collections.deque(maxlen=10)
  for epoch in range(n_epochs):
    # for i in range(n_iters):

    for batch_num in range(len(batch)):
      if (batch_num == 0):
        feat_maps_batch = np.array(feat_maps[0:batch[batch_num]])
        traj_batch = np.array(trajs[0:batch[batch_num]])
      else:
        end_index = np.sum(batch[0:batch_num+1])
        start_index = np.sum(batch[0:batch_num])
        feat_maps_batch = np.array(feat_maps[start_index:end_index])
        traj_batch = np.array(trajs[start_index:end_index])
      mu_D1_batch = []
      value1_batch = []
      policy1_batch = []
      mu_exp1_batch = []
      grad_r1_batch = []
      rewards1 = nn_r.get_rewards(np.reshape(feat_maps_batch.T, [batch[batch_num], hight, width, n_feats]))
      for j in range(len(feat_maps_batch)):
        traj1 = [traj_batch[j]]
        mu_D1 = demo_svf(traj1, N_STATES)
        mu_D1_batch.append(demo_svf(traj1, N_STATES))
      # while(l2_loss>1):
        # if iteration % (n_iters/10) == 0:
        #   print 'iteration: {}'.format(iteration)
        
        # compute the reward matrix
        np.set_printoptions(suppress=True)
        # print("Learning rate is", nn_r.sess.run(nn_r.lr))
        # print(rewards)
        # compute policy 
        
        
        value1, policy1 = value_iteration.value_iteration(P_a, rewards1[j], gamma, error=1e-5, deterministic=True)
        value1_batch.append(value1)
        policy1_batch.append(policy1)
        dict = {0: 'r', 1: 'l', 2: 'u', 3: 's', 4: 'ru', 5: 'lu'}
        policy = [dict[i] for i in policy1]
        # print ("policy is ", policy)
        # print("demo is", mu_D1)
        # compute expected svf
        mu_exp1 = compute_state_visition_freq(P_a, gamma, traj1, policy1, deterministic=True)
        mu_exp1_batch.append(mu_exp1)
        # compute gradients on rewards:
        grad_r1 = mu_D1 - mu_exp1
        grad_r1_batch.append(grad_r1)
          


          # apply gradients to the neural network
      grad_theta, l2_loss, grad_norm = nn_r.apply_grads(np.reshape(feat_maps_batch.T, [batch[batch_num], hight, width, n_feats]), grad_r1_batch)
      # theta_summary = tf.Summary.Image(nn_r.theta)
      # train_summary_writer.add_summary(theta_summary, global_step=i + j*n_iters)
      loss_summary = tf.Summary(value=[tf.Summary.Value(tag="loss",
                                                    simple_value=l2_loss)])
      # train_summary_writer.add_summary(loss_summary, global_step=j*len(trajs)*n_iters + i*n_iters + iteration)
      train_summary_writer.add_summary(loss_summary, global_step=epoch)
      # grad_summary = tf.Summary(value=[tf.Summary.Value(tag="grad_theta",
                                                    #  simple_value=grad_theta)])
      # train_summary_writer.add_summary(loss_summary, global_step=j*len(trajs)*n_iters + i*n_iters + iteration)
      # train_summary_writer.add_summary(grad_summary, global_step=i + j*n_iters)
      if (epoch %10 == 0):
        print("gradient for feature %i is %d", epoch*len(batch)+batch_num, np.linalg.norm(np.array(grad_r1_batch), axis = 0))
      grad_batch_every_state = np.linalg.norm(np.array(grad_r1_batch), axis = 0)
      grad_total_summary = tf.Summary(value =[tf.Summary.Value(tag = "global_grad_norm", simple_value = np.linalg.norm(grad_batch_every_state))])
      train_summary_writer.add_summary(grad_total_summary, global_step = epoch*len(batch)+batch_num)
      for k in range(9):
        grad_summary = tf.Summary(value =[tf.Summary.Value(tag = "state"+str(k), simple_value = grad_batch_every_state[k])])
        train_summary_writer.add_summary(grad_summary, global_step = epoch*len(batch)+batch_num)
      full_grad_rolling.append(np.linalg.norm(grad_batch_every_state))
      if (epoch >20 and ( np.std(full_grad_rolling) < 0.1)):
        print("Increasing gradient ", full_grad_rolling[-1] > np.average(full_grad_rolling))
        print("Stganant gradient", np.std(full_grad_rolling) < 0.1)
        break
      # print ("Normalized grad is ", np.linalg.norm(grad_r1))
      #   if (np.linalg.norm(grad_r1) <1.0):
      #     break
      # if (np.linalg.norm(grad_r1) > 1.0):
      #   embed()
    if (epoch >20 and (np.std(full_grad_rolling) < 0.1)):
        print("Reinitializing  :( ")
        nn_r.sess.run(tf.global_variables_initializer())
        prev_l2_loss = 1e30
        l2_loss = 0
        full_grad_rolling = collections.deque(maxlen=10)
    if (abs((prev_l2_loss - l2_loss)) > 0.1):
      prev_l2_loss = l2_loss
    else:
      print("Loss Converged!, prev, now is ", prev_l2_loss, l2_loss)
      break
    if (l2_loss < 5):
      if (np.linalg.norm(grad_batch_every_state)>5):
        print("Reinitializing :( ")
        nn_r.sess.run(tf.global_variables_initializer())
        prev_l2_loss = 1e30
        l2_loss = 0
      else:
        print ("Going on zero now")
        break
        


  rewards =nn_r.get_rewards(np.reshape(feat_maps[0].T, [1, hight, width, n_feats]))
  weight = nn_r.get_theta_no_loss()
  rewards = normalize(rewards[0])
  _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
  
  for j in range(0,len(feat_maps), 10):
    traj = trajs[j]
    rewards =nn_r.get_rewards(np.reshape(feat_maps[j].T, [1, hight, width, n_feats]))
    weight = nn_r.get_theta_no_loss()
    rewards = normalize(rewards[0])
    _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
  
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    plt.subplot(2, 2, 1)
    ax1 = img_utils.heatmap2d(np.reshape(feat_maps[j][0], (hight, width)), 'Distance Feature', block=False)
    plt.subplot(2, 2, 2)
    if (feat_maps[j].shape[0] > 1):
      ax2 = img_utils.heatmap2d(np.reshape(feat_maps[j][1], (hight, width)), 'SDF Feature', block=False)
    plt.subplot(2, 2, 3)
    if (rewards.shape[0] > 1):
      ax3 = img_utils.heatmap2d(np.reshape(rewards, (hight, width)), 'Reward', block=False)
    
    s = StringIO()
    traj_viz = np.zeros(hight*width)
    maxval = 1.0
    i = 0
    for index in traj:
      traj_viz[index.cur_state] = maxval - i*0.1
      i+=1
    traj_viz[index.next_state] = maxval - i*0.1

    plt.subplot(2, 2, 4)
    ax4 = img_utils.heatmap2d(np.reshape(traj_viz, (hight, width)), 'Observed Traj', block=False)
    plt.savefig(s, format='png')
    img_sum = tf.Summary.Image(encoded_image_string=s.getvalue())
    s.close()
    im_summaries = []
    im_summaries.append(tf.Summary.Value(tag='%s/%d' % ("train", j), image=img_sum))
    summary = tf.Summary(value=im_summaries)
    train_summary_writer.add_summary(summary, j)


  

  return rewards, nn_r



def get_irl_reward_policy(nn_r,feat_maps, P_a, gamma=0.9,lr=0.001):
  # print(N_STATES, N_ACTIONS)

  # init nn model

  rewards =nn_r.get_rewards(feat_maps.T)

  _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
  # return sigmoid(normalize(rewards))
  dict = {0: 'r', 1: 'l', 2: 'u', 3: 's', 4: 'ru', 5: 'lu'}
  policy = [dict[i] for i in policy]

  # print(np.array(policy).reshape(3,3))

  # print(np.array(rewards).reshape(3,3))

  # weight = nn_r.get_theta()

  # print("weight is ", weight)

  return normalize(rewards), policy

def get_policy(rewards, P_a, gamma=0.9,lr=0.001):
  # print(N_STATES, N_ACTIONS)


  value, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
  # return sigmoid(normalize(rewards))
  dict = {0: 'r', 1: 'l', 2: 'u', 3: 's', 4: 'ru', 5: 'lu'}
  policy = [dict[i] for i in policy]

  # print(np.array(policy).reshape(3,3))

  # print(np.array(rewards).reshape(3,3))

  # weight = nn_r.get_theta()

  # print("weight is ", weight)

  return value, policy