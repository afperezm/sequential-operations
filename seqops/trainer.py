import numpy as np
import tensorflow as tf
from data import generate_sequence

class AgentTrainer(object):
    
    def __init__(self, agent, training_iters, display_step, n_first_digit_length, n_second_digit_length):
        self.agent = agent
        self.training_iters = training_iters
        self.display_step = display_step
        self.n_first_digit_length = n_first_digit_length
        self.n_second_digit_length = n_second_digit_length
    
    def train(self, digits, digit_labels, symbols, symbol_labels):
        # Initialize variables
        init = tf.initialize_all_variables()
        # Launch the graph
        with tf.Session() as session:
            session.run(init)
            step = 1
            # Keep training until reach max iterations
            while step < self.training_iters:
                # Generate random sequence from trainig data
                batch_xs, batch_ys, batch_ops = generate_sequence(digits, digit_labels, symbols, symbol_labels, self.agent.batch_size, self.n_first_digit_length, self.n_second_digit_length)
                # Fit training using batch data
                session.run(self.agent.optimizer, feed_dict={self.agent.x: batch_xs, self.agent.y: batch_ys, self.agent.istate: np.zeros((self.agent.batch_size, 2 * self.agent.n_hidden))})
                if step % self.display_step == 0:
                    # Calculate batch accuracy, loss and prediction
                    loss, acc, prd = session.run([self.agent.cost, self.agent.accuracy, self.agent.pred], feed_dict={self.agent.x: batch_xs, self.agent.y: batch_ys, self.agent.istate: np.zeros((self.agent.batch_size, 2 * self.agent.n_hidden))})
                    print "  Iteration=" + str(step) + ", Minibatch Error=" + "{:.6f}".format(np.sqrt(loss)) + ", Training Accuracy=" + "{:.5f}".format(acc)
                    print " ", batch_ops[0], batch_ys[0], prd[0]
                # Increase step
                step += 1
    
    def test(self, digits, digit_labels, symbols, symbol_labels):
        # Initialize variables
        init = tf.initialize_all_variables()
        # Launch the graph
        with tf.Session() as session:
            session.run(init)
            # Generate random sequence from testing data
            batch_xs, batch_ys, batch_ops = generate_sequence(digits, digit_labels, symbols, symbol_labels, self.agent.batch_size, self.n_first_digit_length, self.n_second_digit_length)
            # Calculate batch accuracy
            pred = session.run(self.agent.pred, feed_dict={self.agent.x: batch_xs, self.agent.y: batch_ys, self.agent.istate: np.zeros((self.agent.batch_size, 2 * self.agent.n_hidden))})
            
            return [batch_ys, batch_ops, pred]

