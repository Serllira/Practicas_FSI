import gzip
import _pickle as cPickle

import tensorflow as tf
import numpy as np

#import matplotlib.cm as cm
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f,encoding='latin1')
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

# ---------------- Visualizing some element of the MNIST dataset --------------


#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print (train_y[57])

# ---------------- END OF Visualizing some element of the MNIST dataset --------------

# TODO: the neural net!!
y_data_train = one_hot(train_y, 10)
y_data_valid = one_hot(valid_y,10)
y_data_test = one_hot(test_y,10)


x = tf.placeholder("float", [None, 28*28])  # samples
y_ = tf.placeholder("float", [None, 10], name ='etiquetas')  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)  # Capa de 10 neuronas con 28*28 entradas
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)  # nº de vias

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)  # Capa de 10 neuronas con 10 entradas
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)
##--->90.27% 27 epocas learning rate 0.005          89.09% 21 epocas learning rate 0.01         88.08% 17 epcoas learning rate 0.03

#W1 = tf.Variable(np.float32(np.random.rand(784, 5)) * 0.1)  # Capa de 5 neuronas con 28*28 entradas
#b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)  # nº de vias
#
#W2 = tf.Variable(np.float32(np.random.rand(5, 10)) * 0.1)  # Capa de 5 neuronas con  entradas
#b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)
##---->87.85% 35 epocas learning rate 0.01          87.93% 31 epocas learning rate 0.01         87.31% 22 epocas learning rate 0.03

#W1 = tf.Variable(np.float32(np.random.rand(784, 15)) * 0.1)
#b1 = tf.Variable(np.float32(np.random.rand(15)) * 0.1)
#
#W2 = tf.Variable(np.float32(np.random.rand(15, 10)) * 0.1)
#b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)
##---->91.27% 23 epocas learning rate 0.005         91.02% 20 epocas learning rate 0.01     92.84% 24 epocas learning rate 0.03

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

#train = tf.train.GradientDescentOptimizer(0.03).minimize(loss)  # Ratio de aprendizaje: 0.03  Tarda más en estabilizarse y a veces aumenta el error
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # Ratio de aprendizaje: 0.01
#train = tf.train.GradientDescentOptimizer(0.005).minimize(loss)  # Ratio de aprendizaje: 0.005 Aprende muy lentamente, tarda mucho en estabilizarse

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print("----------------------")
print("   Start training...  ")
print("----------------------")

batch_size = 20
errorPrevio = 0
hayEstabilidad = 0
epoch = 0
erroresValidacion=[]
erroresEntrenamiento=[]


while (hayEstabilidad < 15):
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs , y_: batch_ys})
    ErrorEntrena = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})/ batch_size
    errorActual = sess.run(loss, feed_dict={x: valid_x, y_: y_data_valid})/ len(y_data_valid)

    if (errorActual >= errorPrevio * 0.95):
        hayEstabilidad += 1
    else:
        hayEstabilidad = 0
    print("Epoch #:", epoch, "Error: ", errorActual, "Error Anterior: ", errorPrevio, "Estabilidad: ",hayEstabilidad)
    erroresValidacion.append(errorActual)
    erroresEntrenamiento.append(ErrorEntrena)
    errorPrevio = errorActual
    epoch += 1


print("----------------------")
print("   Start testing...  ")
print("----------------------")

result = sess.run(y, feed_dict={x: test_x})

Aciertos=0
Fallos=0

for b, r in zip(y_data_test, result):
    if (np.argmax(b) == np.argmax(r)):
        Aciertos += 1
    else:
        Fallos += 1
    print (b, "-->", r)
    print ("Aciertos: ", Aciertos)
    print ("Fallos: ",Fallos)
    Total = Aciertos + Fallos
    print("Porcentaje de aciertos: ", (float(Aciertos) / float(Total)) * 100, "%")
    print("----------------------------------------------------------------------------------")

plt.plot(erroresValidacion)
plt.plot(erroresEntrenamiento)

plt.legend(['Error Validacion', 'Error Entrenamiento'], loc='upper right')
plt.show()