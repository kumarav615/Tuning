# mlp for regression with mse loss function
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot


def create_dataset():
    # generate regression dataset
    x, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)

    # standardize dataset
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y.reshape(len(y), 1))[:, 0]

    # split into train and test
    n_train = 500
    trainx, testx = x[:n_train, :], x[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    return trainx, trainy, testx, testy


def create_model(loss_value):
    # define model
    model = Sequential()
    model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='linear'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss=loss_value, optimizer=opt)
    return model


def fit_model(trainX, trainy, testX, testy, model):
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
    return history


def eval_model(trainX, trainy, testX, testy, model):
    # evaluate the model
    train_mse = model.evaluate(trainX, trainy, verbose=0)
    test_mse = model.evaluate(testX, testy, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))


def plot_mse(hist):
    pyplot.subplot(212)
    pyplot.title('Mean Squared Error', pad=-20)
    pyplot.plot(hist.history['mean_squared_error'], label='train')
    pyplot.plot(hist.history['val_mean_squared_error'], label='test')
    pyplot.legend()


def plot_loss(hist, losstype):
    # plot loss during training
    pyplot.subplot(211)
    pyplot.title(losstype, pad=-20)
    pyplot.plot(hist.history['loss'], label='train')
    pyplot.plot(hist.history['val_loss'], label='test')
    pyplot.legend()


trainX, trainy, testX, testy = create_dataset()
loss_value = ['mean_squared_error', 'mean_squared_logarithmic_error', 'mean_absolute_error']
for loss in loss_value:
    print(loss)
    model = create_model(loss)
    history = fit_model(trainX, trainy, testX, testy, model)
    eval_model(trainX, trainy, testX, testy, model)
    if loss != 'mean_squared_error':
        plot_mse(history)
    plot_loss(history, loss)

pyplot.show()
