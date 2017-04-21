import numpy as np
import os
import scipy.io.wavfile as wavfile
from scipy.fftpack import fft


def get_data():
    X_data = np.ones(shape=(43,46),dtype=np.int8)
    Y_label = np.ones(shape=(43),dtype=np.int8)
    i = 0
    for fn in os.listdir('../../phonemes-wav/'):
         if (fn != '.DS_Store'):
             print("../../phonemes-wav/" + fn)
             wav1, wav2 = wavfile.read("../../phonemes-wav/" + fn)
             Y_label[i] = i
             curr_wav_fft = fft(wav2)
             X_data[i,0] = curr_wav_fft[75].real[0]
             X_data[i,1] = curr_wav_fft[75].real[1]
             X_data[i,2] = curr_wav_fft[100].real[0]
             X_data[i,3] = curr_wav_fft[100].real[1]
             X_data[i,4] = curr_wav_fft[200].real[0]
             X_data[i,5] = curr_wav_fft[200].real[1]
             X_data[i,6] = curr_wav_fft[300].real[0]
             X_data[i,7] = curr_wav_fft[300].real[1]
             X_data[i,8] = curr_wav_fft[400].real[0]
             X_data[i,9] = curr_wav_fft[400].real[1]
             X_data[i,10] = curr_wav_fft[600].real[0]
             X_data[i,11] = curr_wav_fft[600].real[1]
             X_data[i,12] = curr_wav_fft[800].real[0]
             X_data[i,13] = curr_wav_fft[800].real[1]
             X_data[i,14] = curr_wav_fft[1000].real[0]
             X_data[i,15] = curr_wav_fft[1000].real[1]
             X_data[i,16] = curr_wav_fft[1200].real[0]
             X_data[i,17] = curr_wav_fft[1200].real[1]
             X_data[i,18] = curr_wav_fft[1400].real[0]
             X_data[i,19] = curr_wav_fft[1400].real[1]
             X_data[i,20] = curr_wav_fft[1600].real[0]
             X_data[i,21] = curr_wav_fft[1600].real[1]
             X_data[i,22] = curr_wav_fft[1800].real[0]
             X_data[i,23] = curr_wav_fft[1800].real[1]
             X_data[i,24] = curr_wav_fft[2000].real[0]
             X_data[i,25] = curr_wav_fft[2000].real[1]
             X_data[i,26] = curr_wav_fft[2200].real[0]
             X_data[i,27] = curr_wav_fft[2200].real[1]
             X_data[i,28] = curr_wav_fft[2400].real[0]
             X_data[i,29] = curr_wav_fft[2400].real[1]
             X_data[i,30] = curr_wav_fft[2600].real[0]
             X_data[i,31] = curr_wav_fft[2600].real[1]
             X_data[i,32] = curr_wav_fft[2800].real[0]
             X_data[i,33] = curr_wav_fft[2800].real[1]
             X_data[i,34] = curr_wav_fft[3000].real[0]
             X_data[i,35] = curr_wav_fft[3000].real[1]
             X_data[i,36] = curr_wav_fft[3200].real[0]
             X_data[i,37] = curr_wav_fft[3200].real[1]
             X_data[i,38] = curr_wav_fft[3500].real[0]
             X_data[i,39] = curr_wav_fft[3500].real[1]
             X_data[i,40] = curr_wav_fft[4000].real[0]
             X_data[i,41] = curr_wav_fft[4000].real[1]
             X_data[i,42] = curr_wav_fft[4500].real[0]
             X_data[i,43] = curr_wav_fft[4500].real[1]
             X_data[i,44] = curr_wav_fft[5000].real[0]
             X_data[i,45] = curr_wav_fft[5000].real[1]
             i += 1
    print(X_data.shape)
    return X_data, Y_label

def main():
    # Number of classes
    C = 43
    # Number of dimensions
    D = 46
    # Size of hidden layer
    H = 10
    # Number of inputs(???)
    I = 0

    reg = 1e-5 # 0.05?
    # learning_rate = .02
    learning_rate = .1
    lrdecay = .95
    num_iters = 10000

    # Set initial parameters
    parameters = {}
    # parameters['W1'] = .01 * np.random.randn(D, H)
    # parameters['b1'] = np.zeros(H)
    # parameters['W2'] = .01 * np.random.randn(H, C)
    # parameters['b2'] = np.zeros(C)
    W1 = .01 * np.random.randn(D, H)
    b1 = np.zeros(H)
    W2 = .01 * np.random.randn(H, C)
    b2 = np.zeros(C)

    # Input data
    X, y = get_data()
    N = X.shape[0]

    ################## Training #######

    num_train = N
    batch_size = 20 # 43?
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      # Random minibatch
      index_batch = np.random.choice(np.arange(num_train), batch_size)
      X_batch = X[index_batch]
      y_batch = y[index_batch]


      # From current minibatch, get gradient and loss


      # ReLU activtation layer
      hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
      scores = np.dot(hidden_layer, W2) + b2

      # Softmax Regression
      # Compute probabilities of classes
      exp_scores = np.exp(scores)
      probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

      # compute the loss: average cross-entropy loss and regularization
      correct_logprobs = -np.log(probs[range(N),y])
      data_loss = np.sum(correct_logprobs)/N
      reg_loss = reg*np.sum(W1*W1) + reg*np.sum(W2*W2)
      loss = data_loss + reg_loss

      # Calculate gradients from backprop
      grads = {}
      dscores = probs
      dscores[range(N),y] -= 1
      dscores /= N

      # backpropate the gradient to the parameters
      # first backprop into parameters W2 and b2
      dW2 = np.dot(hidden_layer.T, dscores)
      db2 = np.sum(dscores, axis=0)
      # next backprop into hidden layer
      dhidden = np.dot(dscores, W2.T)
      # backprop the ReLU non-linearity
      dhidden[hidden_layer <= 0] = 0
      # finally into W,b
      dW1 = np.dot(X.T, dhidden)
      db1 = np.sum(dhidden, axis=0)

      # add regularization gradient contribution
      dW2 += 2 * reg * W2
      dW1 += 2 * reg * W1

      grads['W1'] = dW1
      grads['b1'] = db1
      grads['W2'] = dW2
      grads['b2'] = db2

      loss_history.append(loss)


      W1 += -learning_rate * grads['W1']
      b1 += -learning_rate * grads['b1']
      W2 += -learning_rate * grads['W2']
      b2 += -learning_rate * grads['b2']

      if it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        hidden_layer = np.maximum(0, np.dot(X, W1) + b1) # note, ReLU activation
        scores = np.dot(hidden_layer, W2) + b2
        y_pred = np.argmax(scores, axis=1)

        hidden_layer_batch = np.maximum(0, np.dot(X_batch, W1) + b1) # note, ReLU activation
        scores_batch = np.dot(hidden_layer_batch, W2) + b2
        y_pred_batch = np.argmax(scores_batch, axis=1)
        train_acc = (y_pred_batch == y_batch).mean()
        val_acc = (y_pred == y).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= lrdecay
    print(W1)
    # aah
    X_test = X[5]
    hidden_layer = np.maximum(0, np.dot(X_test, W1) + b1) # note, ReLU activation
    scores = np.dot(hidden_layer, W2) + b2
    y_pred = np.argmax(scores)
    print(y_pred)




if __name__ == '__main__': # Main function
    main()
