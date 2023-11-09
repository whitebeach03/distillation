import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    # setting history path
    student_path = 'history/student/0'

    # loading history
    _, _, _, student_acc = load_hist(student_path, 1)

    # plot result
    x = np.arange(50)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')

    plt.plot(x, student_acc, label='Student', linewidth=0.5, color='red')

    plt.xticks(np.arange(0, 55, 5))
    plt.yticks(np.arange(0.60, 1.00, 0.05))
    plt.xlim(0, 51)
    plt.ylim(0.60, 0.91)

    plt.legend()
    plt.savefig('./result/result00.png')


def load_hist(path, iteration):
    dic = {}
    for i in range(iteraiton):
        with open(path + str(i) + '.pickle', mode='rb') as f:
            dic[i] = pickle.load(f)
    
    val_acc = np.zeros(len(dic[i]['val_accuracy']))

    for i in range(iteration):
        val_acc += np.array(dic[i]['val_accuracy'])
    
    val_acc = val_acc / iteration

    return val_acc

