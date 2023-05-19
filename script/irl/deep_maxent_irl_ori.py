from random import randint
import numpy as np
import tensorflow as tf
import mdp.gridworld as gridworld
import mdp.value_iteration as value_iteration
import img_utils
import rospy
import tf_utils
import tensorflow as tf
from utils import *
import matplotlib.pyplot as plt
import time

class DeepIRLFC:
  def __init__(self, n_input, lr, n_h1=400, n_h2=300, l2=10, name='deep_irl_fc'):
    self.n_input = n_input
    self.lr = lr
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.name = name
    self.sess = tf.Session()
    self.input_s, self.reward, self.theta = self._build_network(self.name)
    # self.optimizer = tf.train.GradientDescentOptimizer(lr)
    self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    
    self.grad_r = tf.placeholder(tf.float32, [None, 1])
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


  def _build_network(self, name):
    input_s = tf.placeholder(tf.float32, [None, self.n_input])
    with tf.variable_scope(name):
      fc1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.relu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      fc2 = tf_utils.fc(fc1, self.n_h2, scope="fc2", activation_fn=tf.nn.relu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      reward = tf_utils.fc(fc2, 1, scope="reward")
    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    return input_s, reward, theta


  def get_theta(self):
    theta = self.sess.run(self.theta)
    self.saver.save(self.sess, "../weights2/saved_weights")
    return theta

  def get_theta_no_loss(self):
    theta = self.sess.run(self.theta)
    self.saver.save(self.sess, "../weights6/saved_weights")
    return theta


  def get_rewards(self, states):
    rewards = self.sess.run(self.reward, feed_dict={self.input_s: states})
    return rewards


  def apply_grads(self, feat_map, grad_r):
    grad_r = np.reshape(grad_r, [-1, 1])
    feat_map = np.reshape(feat_map, [-1, self.n_input])
    _, grad_theta, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms], 
      feed_dict={self.grad_r: grad_r, self.input_s: feat_map})
    return grad_theta, l2_loss, grad_norms

  def load_weights(self):
    # with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('../weights6/saved_weights.meta')
    new_saver.restore(self.sess, tf.train.latest_checkpoint('../weights6/'))

  # def save_weights(self):
  #   self.theta.

class DeepIRLConv:
  def __init__(self, n_input, feature_map_size, lr, n_h1=400, n_h2=300, l2=10, name='deep_irl_conv'):
    self.n_input = n_input
    self.lr = lr
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.name = name
    self.feature_map_size = feature_map_size
    self.sess = tf.Session()
    self.input_s, self.reward, self.theta = self._build_network(self.name)
    # self.optimizer = tf.train.GradientDescentOptimizer(lr)
    self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    
    self.grad_r = tf.placeholder(tf.float32, [None, self.feature_map_size, self.feature_map_size, 1])
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

  def conv_layer(self, x, input_channel, hidden_size, name="Conv"):
    with tf.variable_scope(name):
      conv1 = tf.layers.conv2d(inputs=x, filters=hidden_size, kernel_size=(3,3),padding="same")
      batch_norm1 = tf.layers.batch_normalization(conv1)
    return tf.nn.relu(batch_norm1)

  def _build_network(self, name):
    # input_s = tf.placeholder(tf.float32, [None, self.n_input])

    # with tf.variable_scope(name):
    #   fc1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.relu,
    #     initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
    #   fc2 = tf_utils.fc(fc1, self.n_h2, scope="fc2", activation_fn=tf.nn.relu,
    #     initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
    #   reward = tf_utils.fc(fc2, 1, scope="reward")
    # theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    input_s = tf.placeholder(tf.float32, [1, self.feature_map_size, self.feature_map_size , self.n_input])
    hidden_size1 = 32
    hidden_size2 = 64
    with tf.variable_scope(name):
      # Block1
      conv1 = self.conv_layer(input_s, self.n_input, hidden_size1, name="Conv1")
      conv2 = self.conv_layer(conv1, hidden_size1, hidden_size1, name="Conv2")
      conv3 = self.conv_layer(conv2, hidden_size1, hidden_size1, name="Conv3")
      max_pool1 = tf.layers.max_pooling2d(inputs=conv3,pool_size=2,strides=2,name="Pool1")

      # Block2
      conv4 = self.conv_layer(max_pool1, hidden_size1, hidden_size2, name="Conv4")
      conv5 = self.conv_layer(conv4, hidden_size2, hidden_size2, name="Conv5")
      conv6 = self.conv_layer(conv5, hidden_size2, hidden_size2, name="Conv6")
      max_pool2 = tf.layers.max_pooling2d(inputs=conv6,pool_size=2,strides=2,name="Pool2")

      # Block3
      unpooling1 = tf.keras.layers.Conv2DTranspose(filters=hidden_size2,kernel_size=(3,3),strides=2,padding="valid")(max_pool2)
      conv7 = self.conv_layer(unpooling1, hidden_size2, hidden_size2, name="Conv7")
      conv8 = self.conv_layer(conv7, hidden_size2, hidden_size2, name="Conv8")
      conv9 = self.conv_layer(conv8, hidden_size2, hidden_size2, name="Conv9")

      # Block4
      unpooling2 = tf.keras.layers.Conv2DTranspose(filters=hidden_size2,kernel_size=(3,3),strides=2,padding="valid")(conv9)
      conv10 = self.conv_layer(unpooling2, hidden_size1, hidden_size1, name="Conv10")
      conv11 = self.conv_layer(conv10, hidden_size1, hidden_size1, name="Conv11")
      conv12 = self.conv_layer(conv11, hidden_size1, hidden_size1, name="Conv12")
      # print(conv12)

      # Output
      reward = self.conv_layer(conv12, 1, 1, name="Output")
    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    return input_s, reward, theta


  def get_theta(self):
    theta = self.sess.run(self.theta)
    self.saver.save(self.sess, "../weights2/saved_weights")
    return theta

  def get_theta_no_loss(self):
    theta = self.sess.run(self.theta)
    self.saver.save(self.sess, "../weights6/saved_weights")
    return theta


  def get_rewards(self, states):
    rewards = self.sess.run(self.reward, feed_dict={self.input_s: states})
    return rewards


  def apply_grads(self, feat_map, grad_r):
    grad_r = np.reshape(grad_r, [-1, self.feature_map_size, self.feature_map_size, 1])
    feat_map = np.reshape(feat_map, [-1, self.feature_map_size, self.feature_map_size, self.n_input])
    _, grad_theta, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms], 
      feed_dict={self.grad_r: grad_r, self.input_s: feat_map})
    return grad_theta, l2_loss, grad_norms

  def load_weights(self):
    # with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('../weights2/saved_weights.meta')
    new_saver.restore(self.sess, tf.train.latest_checkpoint('../weights2/'))

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

  for s in range(N_STATES):
    for t in range(T-1):
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

def deep_maxent_irl_mul_traj(feat_maps, P_a, gamma, trajs, lr, n_iters):
  """
  Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

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

  # tf.set_random_seed(1)

  resume_training = False
  
  N_STATES, _, N_ACTIONS = np.shape(P_a)
  # init nn model
  # print(feat_maps[0].shape)
  nn_r = DeepIRLConv(feat_maps[0].shape[3], feat_maps[0].shape[1], lr, 3, 3)

  if(resume_training):
    nn_r.load_weights()
  
  for i in range(len(feat_maps)):
    mu_D = demo_svf(trajs[i], N_STATES)
    for iteration in range(n_iters):
      
      # compute the reward matrix
      rewards = nn_r.get_rewards(feat_maps[i])
      rewards = np.reshape(rewards[0], rewards[0].shape[0]*rewards[0].shape[1])

      # compute policy 
      _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
      
      # compute expected svf
      mu_exp = compute_state_visition_freq(P_a, gamma, trajs[i], policy, deterministic=True)
      
      # compute gradients on rewards:
      grad_r = mu_D - mu_exp

      # apply gradients to the neural network
      grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_maps[i], grad_r)
      print("Loss is ",l2_loss)
    weight = nn_r.get_theta()
  return normalize(rewards)

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



def deep_maxent_irl_fetch(feat_maps, P_a, gamma, trajs, lr, n_iters):
  """
  Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

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
  # feat_maps = feat_maps[:1]
  # trajs = trajs[:1]
  # feat_maps[:] = feat_maps[:][:2]
  # feat_maps[:][0] = [-e for e in feat_maps[:][0]]
  # print(np.array(feat_maps[1]).T)


  N_STATES, _, N_ACTIONS = np.shape(P_a)
  # print(N_STATES, N_ACTIONS)

  # init nn model
  nn_r = DeepIRLFC(feat_maps[0].shape[0], lr, 3, 3)

  # find state visitation frequencies using demonstrations
  
  # training 
  # print(trajs)

  train_summary_writer = tf.summary.FileWriter("../logs1")

  loss_summary = tf.Summary()

  for j in range(2):

    for i in range(len(trajs)):
      traj = [trajs[i]]
      # print(traj)
      mu_D = demo_svf(traj, N_STATES)
      for iteration in range(n_iters):
      # while(l2_loss>1):
        # if iteration % (n_iters/10) == 0:
        #   print 'iteration: {}'.format(iteration)
        
        # compute the reward matrix
        
        rewards = nn_r.get_rewards(feat_maps[i].T)
        # print(rewards)
        # compute policy 
        _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
        
        # compute expected svf
        mu_exp = compute_state_visition_freq(P_a, gamma, traj, policy, deterministic=True)
        
        # compute gradients on rewards:
        grad_r = mu_D - mu_exp

        # apply gradients to the neural network
        grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_maps[i].T, grad_r)

        loss_summary.value.add(tag='loss', simple_value=l2_loss)
        train_summary_writer.add_summary(loss_summary, global_step=j*len(trajs)*n_iters + i*n_iters + iteration)
        # train_summary_writer.add_summary(l2_loss, global_step=i*n_iters + iteration)
        print(l2_loss)
        if(l2_loss < 0.1):
          break
        # with train_summary_writer.as_default():
        #   tf.summary.scalar('loss', l2_loss, step=i*n_iters + iteration)
        #   tf.summary.scalar('grad_theta', grad_theta, step=i*n_iters + iteration)

      # print("Training %d  done" % i)
  
  print(trajs)


  rewards =nn_r.get_rewards(feat_maps[0].T)

  _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
  # return sigmoid(normalize(rewards))
  dict = {0: 'r', 1: 'l', 2: 'u', 4: 's'}
  policy = [dict[i] for i in policy]
  # print(np.array(policy).reshape(3,3))

  # print(np.array(rewards).reshape(3,3))

  weight = nn_r.get_theta()


  return rewards


def deep_maxent_irl_no_traj_loss(feat_maps, P_a, gamma, trajs,  lr, n_iters):
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
  nn_r = DeepIRLFC(feat_maps[0].shape[0], lr, 3, 3)

  # Hight and width of the feature map. (Assume the grid is a square)
  hight = int(np.sqrt(feat_maps[0].shape[1]))
  width = hight

  # find state visitation frequencies using demonstrations
  
  # training 
  # print(trajs)

  train_summary_writer = tf.summary.FileWriter("../logs1")

  loss_summary = tf.Summary()

  for j in range(1):
    prev_loss = 5000
    while(True):
      
      num1 = randint(0, len(trajs)-1)
      num2 = randint(0, len(trajs)-1)
      while(num1 == num2):
          num2 = randint(0, len(trajs)-1)

      traj1 = [trajs[num1]]
      traj2 = [trajs[num2]]
      # print(traj)
      mu_D1 = demo_svf(traj1, N_STATES)
      mu_D2 = demo_svf(traj2, N_STATES)


      for iteration in range(n_iters):
      # while(l2_loss>1):
        # if iteration % (n_iters/10) == 0:
        #   print 'iteration: {}'.format(iteration)
        
        # compute the reward matrix
        
        rewards1 = nn_r.get_rewards(feat_maps[num1].T)
        rewards2 = nn_r.get_rewards(feat_maps[num2].T)
        # print(rewards)
        # compute policy 
        _, policy1 = value_iteration.value_iteration(P_a, rewards1, gamma, error=0.01, deterministic=True)
        _, policy2 = value_iteration.value_iteration(P_a, rewards2, gamma, error=0.01, deterministic=True)
        
        # compute expected svf
        mu_exp1 = compute_state_visition_freq(P_a, gamma, traj1, policy1, deterministic=True)
        mu_exp2 = compute_state_visition_freq(P_a, gamma, traj2, policy2, deterministic=True)
        
        # compute gradients on rewards:
        grad_r1 = mu_D1 - mu_exp1
        grad_r2 = mu_D2 - mu_exp2

        # compute the trajectory ranking loss
        # r1 = get_reward_sum_from_policy(rewards1, policy1, [width, hight])
        # r2 = get_reward_sum_from_policy(rewards2, policy2, [width, hight])
        # r1 = percent_change[num1]
        # r2 = percent_change[num2]
        # traj_loss = -np.log(np.exp(max(r1,r2)) / (np.exp(r1) + np.exp(r2)))
        # traj_loss = (r1 < r2)


        # apply gradients to the neural network
        grad_theta, l2_loss, grad_norm = nn_r.apply_grads(0.5 * (feat_maps[num2].T + feat_maps[num1].T), (grad_r1 + grad_r2)/2)

        loss_summary.value.add(tag='loss', simple_value=l2_loss)
        # train_summary_writer.add_summary(loss_summary, global_step=j*len(trajs)*n_iters + i*n_iters + iteration)
        # train_summary_writer.add_summary(l2_loss, global_step=i*n_iters + iteration)
      print(l2_loss)
      if(abs(l2_loss - prev_loss) < 0.0001):
        break
      prev_loss = l2_loss
        # with train_summary_writer.as_default():
        #   tf.summary.scalar('loss', l2_loss, step=i*n_iters + iteration)
        #   tf.summary.scalar('grad_theta', grad_theta, step=i*n_iters + iteration)

      # print("Training %d  done" % i)
  
  # print(trajs)


  rewards =nn_r.get_rewards(feat_maps[0].T)

  _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
  # print(policy)
  # print(rewards)
  # return sigmoid(normalize(rewards))
  dict = {0: 'r', 1: 'l', 2: 'u', 3: 's'}
  policy = [dict[i] for i in policy]
  print(np.array(policy).reshape(hight, width))

  # print(np.array(rewards).reshape(3,3))

  weight = nn_r.get_theta_no_loss()


  return rewards

def deep_maxent_irl_traj_loss(feat_maps, P_a, gamma, trajs,percent_change,  lr, n_iters):
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
  nn_r = DeepIRLFC(feat_maps[0].shape[0], lr, 3, 3)

  # Hight and width of the feature map. (Assume the grid is a square)
  hight = int(np.sqrt(feat_maps[0].shape[1]))
  width = hight

  # find state visitation frequencies using demonstrations
  
  # training 
  # print(trajs)

  train_summary_writer = tf.summary.FileWriter("../logs1")

  loss_summary = tf.Summary()

  for j in range(1):

    while(True):
      
      num1 = randint(0, len(trajs)-1)
      num2 = randint(0, len(trajs)-1)
      while(num1 == num2):
          num2 = randint(0, len(trajs)-1)

      traj1 = [trajs[num1]]
      traj2 = [trajs[num2]]
      # print(traj)
      mu_D1 = demo_svf(traj1, N_STATES)
      mu_D2 = demo_svf(traj2, N_STATES)


      for iteration in range(n_iters):
      # while(l2_loss>1):
        # if iteration % (n_iters/10) == 0:
        #   print 'iteration: {}'.format(iteration)
        
        # compute the reward matrix
        
        rewards1 = nn_r.get_rewards(feat_maps[num1].T)
        rewards2 = nn_r.get_rewards(feat_maps[num2].T)
        # print(rewards)
        # compute policy 
        _, policy1 = value_iteration.value_iteration(P_a, rewards1, gamma, error=0.01, deterministic=True)
        _, policy2 = value_iteration.value_iteration(P_a, rewards2, gamma, error=0.01, deterministic=True)
        
        # compute expected svf
        mu_exp1 = compute_state_visition_freq(P_a, gamma, traj1, policy1, deterministic=True)
        mu_exp2 = compute_state_visition_freq(P_a, gamma, traj2, policy2, deterministic=True)
        
        # compute gradients on rewards:
        grad_r1 = mu_D1 - mu_exp1
        grad_r2 = mu_D2 - mu_exp2

        # compute the trajectory ranking loss
        # r1 = get_reward_sum_from_policy(rewards1, policy1, [width, hight])
        # r2 = get_reward_sum_from_policy(rewards2, policy2, [width, hight])
        r1 = percent_change[num1]
        r2 = percent_change[num2]
        if(r1 != r2):
          traj_loss = -np.log(np.exp(max(r1,r2)) / (np.exp(r1) + np.exp(r2)))
        else:
          traj_loss = 0.0
        # traj_loss = -np.log(np.exp(max(r1,r2)) / (np.exp(r1) + np.exp(r2)))
        # traj_loss = (r1 < r2)

        # apply gradients to the neural network
        grad_theta, l2_loss, grad_norm = nn_r.apply_grads(0.5 * (feat_maps[num2].T + feat_maps[num1].T), (grad_r1 + grad_r2)/2 - traj_loss / 2.0)

        loss_summary.value.add(tag='loss', simple_value=l2_loss)
        # train_summary_writer.add_summary(loss_summary, global_step=j*len(trajs)*n_iters + i*n_iters + iteration)
        # train_summary_writer.add_summary(l2_loss, global_step=i*n_iters + iteration)
      print(l2_loss)
      if(l2_loss < 1):
        break
        # with train_summary_writer.as_default():
        #   tf.summary.scalar('loss', l2_loss, step=i*n_iters + iteration)
        #   tf.summary.scalar('grad_theta', grad_theta, step=i*n_iters + iteration)

      # print("Training %d  done" % i)
  
  # print(trajs)


  rewards =nn_r.get_rewards(feat_maps[0].T)

  _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
  # print(policy)
  # print(rewards)
  # return sigmoid(normalize(rewards))
  dict = {0: 'r', 1: 'l', 2: 'u', 3: 's'}
  policy = [dict[i] for i in policy]
  print(np.array(policy).reshape(3,3))

  # print(np.array(rewards).reshape(3,3))

  weight = nn_r.get_theta()


  return rewards

def get_irl_reward_policy(nn_r,feat_maps, P_a, gamma=0.9,lr=0.001):
  # print(N_STATES, N_ACTIONS)

  # init nn model

  rewards =nn_r.get_rewards(feat_maps.T)

  _, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)
  # return sigmoid(normalize(rewards))
  dict = {0: 'r', 1: 'l', 2: 'u', 3: 's'}
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
  dict = {0: 'r', 1: 'l', 2: 'u', 3: 's'}
  policy = [dict[i] for i in policy]

  # print(np.array(policy).reshape(3,3))

  # print(np.array(rewards).reshape(3,3))

  # weight = nn_r.get_theta()

  # print("weight is ", weight)

  return value, policy
