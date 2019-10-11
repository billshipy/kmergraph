nloop=1
tacc_array = []
vacc_array = []
while (nloop <= 200):
    print("train with",nloop,"hidden units.\n")
    nn = NeuralNetMLP(n_hidden=nloop,
                  l2=0.01,
                  epochs=n_epochs,
                  eta=0.0005,
                  minibatch_size=100,
                  shuffle=True,
                  seed=1)
    nn.fit(X_train=X_train[:55000],
       y_train=y_train[:55000],
       X_valid=X_train[55000:],
       y_valid=y_train[55000:])
    arrt = nn.eval_['train_acc'][99]
    tacc_array.append(arrt)
    arrv =nn.eval_['valid_acc'][99]
    vacc_array.append(arrv)
    nloop = nloop+1
import numpy as np

a = np.arange(5)
b = a
print('a & b', np.may_share_memory(a, b))


a = np.arange(5)
print('a & b', np.may_share_memory(a, b))
import matplotlib.pyplot as plt

plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
#plt.savefig('images/12_07.png', dpi=300)
plt.show()
plt.plot(range(nn.n_hidden), tacc_array,
         label='training')
plt.plot(range(nn.n_hidden), vacc_array,
         label='validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('hidden units')
plt.legend()
#plt.savefig('images/12_08.png', dpi=300)
plt.show()