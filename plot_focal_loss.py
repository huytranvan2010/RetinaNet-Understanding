import numpy as np
import matplotlib.pyplot as plt

def focal_loss(alpha, gamma, predicted_prob):
    return -alpha * (1- predicted_prob)** gamma * np.log(predicted_prob)

predicted_prob = np.linspace(start=0, stop=1, num=50)
gamma_list = [0, 0.5, 1, 2, 5]
alpha = 1

for gamma in gamma_list:
    output = focal_loss(alpha=alpha, gamma=gamma, predicted_prob=predicted_prob)
    plt.plot(predicted_prob, output, label="gamma = {}".format(gamma))
    plt.legend()

plt.xlabel("predicted probability")
plt.ylabel("Focal loss")
plt.savefig("plot_focal_loss.png")
plt.show()