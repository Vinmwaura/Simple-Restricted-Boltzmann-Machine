import numpy as np


class RBM:
    def __init__(self, visible_nodes=1, hidden_nodes=1, learning_rate=1e-3, max_epoch=100):
        self.weights = np.random.rand(visible_nodes, hidden_nodes)
        self.bias_hidden = np.random.rand(1, hidden_nodes)
        self.bias_visible = np.random.rand(1, visible_nodes)
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch

    """
    Train RBM Net using Contrastive Divergence learning procedure
    Given training data, v, the binary state of each hidden unit is set to 1
    with a given probability p(h|v). Then binary state of each visible unit
    is set to 1 with a given probability p(v|h),
    """
    def train(self, data):
        for epoch in range(self.max_epoch):
            # Positive Phase - Sets hidden units on with probability p(h|v)
            positive_hidden_out, positive_hidden_binary = self.forward_hidden(data)

            # <v_i * h_j>_data - Expectation of data distribition
            pos_assoc = np.dot(data.T, positive_hidden_out)

            # Negative Phase
            negative_visible_out, negative_visible_binary = self.forward_visible(positive_hidden_out)
            negative_hidden_out, negative_hidden_binary = self.forward_hidden(negative_visible_out)

            """
            <v_i * h_j>_model - Expectation of model distribition
            This is computed using Gibbs sampling executed above:
            One iteration of alternating Gibbs sampling consists of
            updating all of the hidden units, p(h|v), followed by
            updating all of the visible units, p(v|h) 
            """
            neg_assoc = np.dot(negative_visible_out.T, negative_hidden_out)

            # Updating parameters
            # Delta_weight = <v_i * h_j>_data - <v_i * h_j>_model
            weights_delta = pos_assoc - neg_assoc
            self.weights += self.learning_rate * (weights_delta / data.shape[0])
            
            # Delta_visible_bias = v_0 - v_1
            bias_visible_delta = np.sum(data - negative_visible_binary, axis=0)
            self.bias_visible += self.learning_rate * (bias_visible_delta / data.shape[0])

            # Delta_hidden_bias = p(h=1|v_0) - p(h=1|v_1)
            bias_hidden_delta = np.sum(positive_hidden_out - negative_hidden_out, keepdims=True, axis=0)
            self.bias_hidden += self.learning_rate * (bias_hidden_delta / data.shape[0])

    def sigmoid(self, out):
        return 1 / (1 + np.exp(-out))

    def forward_hidden(self, data):
        # Input of hidden units
        hidden_in = np.dot(data, self.weights) + self.bias_hidden
        # Activation of hidden units (Probability of being 1)
        hidden_out_prob = self.sigmoid(hidden_in)

        """
        Random values used in determing whether hidden unit
        is set to 1
        """
        shape1_hidden = hidden_out_prob.shape[0]
        shape2_hidden = hidden_out_prob.shape[1]
        random_prob = np.random.rand(shape1_hidden, shape2_hidden)

        # Determines whether hidden units are set to 1, otherwise 0
        hidden_out_binary = hidden_out_prob > random_prob
        return hidden_out_prob, hidden_out_binary

    def forward_visible(self, data):
        # Input of visible units
        visible_in = np.dot(data, self.weights.T) + self.bias_visible
        # Activation of visible units (Probability of being 1)
        visible_out_prob = self.sigmoid(visible_in)

        """
        Random values used in determing whether hidden unit
        is set to 1
        """
        shape1_visible = visible_out_prob.shape[0]
        shape2_visible = visible_out_prob.shape[1]
        random_prob = np.random.rand(shape1_visible, shape2_visible)

        # Determines whether visbible units are set to 1, otherwise 0
        visible_out_binary = visible_out_prob > random_prob
        return visible_out_prob, visible_out_binary

    # Hidden units of Net are used to regenerate data on Visible units.
    # This is done by giving an example to the net and see what units
    # in the hidden unit.
    def daydream(self, sample_size=1):
        visible_size, hidden_size = self.weights.shape

        data = np.random.rand(sample_size, visible_size)
        # Positive Phase
        positive_hidden_out, positive_hidden_binary = self.forward_hidden(data)

        # Negative Phase (aka Daydreaming phase)
        negative_visible_out, negative_visible_binary = self.forward_visible(positive_hidden_binary)

        print("Reconstructed Data:\n", np.multiply(negative_visible_binary, 1))
        print("-" * 100)

def main():
    training_data = np.array(
        [[1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0]])

    rbm = RBM(visible_nodes=6, hidden_nodes=3, learning_rate=1e-1, max_epoch=1000)
    rbm.train(training_data)
    rbm.daydream(10)

if __name__ == '__main__':
    main()
