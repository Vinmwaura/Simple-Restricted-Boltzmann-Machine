import numpy as np


class RBM:
    def __init__(self, visible_nodes=1, hidden_nodes=1, learning_rate=1e-3, max_epoch=100):
        self.weights = np.random.rand(visible_nodes, hidden_nodes)
        self.bias_hidden = np.random.rand(1, hidden_nodes)
        self.bias_visible = np.random.rand(1, visible_nodes)
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch

    def train(self, data):
        for epoch in range(self.max_epoch):
            # Positive Phase
            positive_hidden_out, positive_hidden_binary = self.forward_hidden(data)

            pos_assoc = np.dot(data.T, positive_hidden_out)

            # Negative Phase
            negative_visible_out, negative_visible_binary = self.forward_visible(positive_hidden_binary)
            negative_hidden_out, negative_hidden_binary = self.forward_hidden(negative_visible_out)

            neg_assoc = np.dot(negative_visible_out, negative_hidden_out)

            # Updating parameters
            self.weights += self.learning_rate * (np.sum((pos_assoc - neg_assoc), axis=0) / data.shape[0])
            self.bias_visible += self.learning_rate * (np.sum((data - negative_visible_binary), axis=0) / data.shape[0])
            self.bias_hidden += self.learning_rate * (np.sum((positive_hidden_out - negative_hidden_out), axis=0) / data.shape[0])

    def sigmoid(self, out):
        return 1 / (1 + np.exp(-out))

    def forward_hidden(self, data):
        hidden_in = np.dot(data, self.weights) + self.bias_hidden
        hidden_out_prob = self.sigmoid(hidden_in)

        shape1_hidden = hidden_out_prob.shape[0]
        shape2_hidden = hidden_out_prob.shape[1]
        random_prob = np.random.rand(shape1_hidden, shape2_hidden)

        hidden_out_binary = hidden_out_prob < random_prob
        return hidden_out_prob, hidden_out_binary

    def forward_visible(self, data):
        visible_in = np.dot(data, self.weights.T) + self.bias_visible
        visible_out_prob = self.sigmoid(visible_in)

        shape1_visible = visible_out_prob.shape[0]
        shape2_visible = visible_out_prob.shape[1]
        random_prob = np.random.rand(shape1_visible, shape2_visible)

        visible_out_binary = visible_out_prob < random_prob
        return visible_out_prob, visible_out_binary


def main():
    training_data = np.array(
        [[1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0]])

    rbm = RBM(visible_nodes=6, hidden_nodes=3)
    rbm.train(training_data)

if __name__ == '__main__':
    main()
