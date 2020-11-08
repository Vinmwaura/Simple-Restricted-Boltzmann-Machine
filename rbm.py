import cv2
import glob
import numpy as np

# Seed value
np.random.seed(1234)

class RBM:
    def __init__(self, visible_nodes=1, hidden_nodes=1, learning_rate=1e-3, max_epoch=100):
        self.weights = np.random.normal(loc=0, scale=0.01, size=(visible_nodes, hidden_nodes))
        self.bias_hidden = np.random.normal(loc=0, scale=0.01, size=(1, hidden_nodes))
        self.bias_visible = np.random.normal(loc=0, scale=0.01, size=(1, visible_nodes))
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch

    """
    Train RBM Net using Contrastive Divergence learning procedure
    Given training data, v, the binary state of each hidden unit is set to 1
    with a given probability p(h|v). Then binary state of each visible unit
    is set to 1 with a given probability p(v|h),
    """
    def train(self, data):
        losses = []
        for epoch in range(self.max_epoch):
            np.random.shuffle(data)

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

            losses = np.sum(0.5 * ((data - negative_visible_binary)**2))

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
            
            status = self.draw_images_training(data[:10,:], negative_visible_binary[:10,:])
            if status is False:
                break

            print("Epoch: ", epoch, " Reconstruction Error: ", losses / data.shape[0])

    # Visualize training images and reconstructed images
    def draw_images_training(self, orig_images, recon_images):
        orig_image = np.float32(orig_images[0].reshape(64,64))
        for img_ in orig_images[1:]:
            img_ = np.float32(img_.reshape(64,64))
            orig_image = np.hstack((orig_image, img_))

        recon_image = np.float32(recon_images[0].reshape(64,64))
        for img_ in recon_images[1:]:
            
            img_ = np.float32(img_.reshape(64,64))
            recon_image = np.hstack((recon_image, img_))

        image = np.vstack((orig_image, recon_image))
        cv2.imshow("Original/Reconstructed Images", image)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return False
        return True

    # Visualize sampled data from model (Daydreaming)
    def draw_images_daydreaming(self, daydreamed_images):
        combined_imgs = np.float32(daydreamed_images[0].reshape(64,64))
        for img_ in daydreamed_images[1:]:
            img_ = np.float32(img_.reshape(64,64))
            combined_imgs = np.hstack((combined_imgs, img_))

        print("Press Q to exit when window is selected")

        cv2.imshow("Daydreamed Images", combined_imgs)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

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

        #data = np.ones((sample_size, visible_size))
        data = np.random.randint(0, 1, (sample_size, visible_size))
        # Sampling from model distribution (Multiple iterations makes it work)
        for _ in range(100):
            # Positive Phase
            positive_hidden_out, positive_hidden_binary = self.forward_hidden(data)

            # Negative Phase (aka Daydreaming phase)
            negative_visible_out, negative_visible_binary = self.forward_visible(positive_hidden_binary)
            data = negative_visible_out
        self.draw_images_daydreaming(
            np.multiply(
                negative_visible_binary, 1))

def create_image_data(file_path):
    training_data = []
    for name in glob.glob(file_path):
        # Simple 64 * 64 Binary Vector Example
        # Converts Grayscale image to binary image( 0's and 1's only)
        img = 1 - (cv2.imread(name, 0) / 255)

        # Resizes image to easier to handle 64 *64 size: 4096 nodes
        img = cv2.resize(img, (64, 64))

        # Converts 64*64 image to vectors of size: (1, 1, 4096)
        input_vector = img.reshape(64 * 64)
        training_data.append(input_vector)

    training_data = np.array(training_data)
    return training_data

def main():
    """
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
    """
    training_data = create_image_data('./Example images/*')
    rbm = RBM(visible_nodes=4096, hidden_nodes=750, learning_rate=5e-3, max_epoch=1000)
    rbm.train(training_data)
    rbm.daydream(10)
    print("Finished")

if __name__ == '__main__':
    main()
