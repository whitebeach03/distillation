import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    st_path = './history/resnet/st/10.pickle'
    cam_path = './history/resnet/cam/10.pickle'
    
    st_acc = load_hist(st_path)
    cam_acc = load_hist(cam_path)
    
    x = np.arange(100)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel('epoch')
    plt.ylabel('validation accuracy')
    
    plt.plot(x, st_acc, label='Distillation', linewidth=0.5, color='orange')
    plt.plot(x, cam_acc, label='Proposed', linewidth=0.5, color='green')
    
    plt.xticks(np.arange(0, 110, 10))
    plt.yticks(np.arange(0.55, 0.95, 0.05))
    plt.xlim(0, 101)
    plt.ylim(0.60, 0.90)
    
    plt.legend()
    plt.savefig('./result/resnet_result_100.png')

def load_hist(path):
    dic = {}
    with open(path, mode='rb') as f:
        dic = pickle.load(f)
    val_acc = np.array(dic['val_accuracy'])
    return val_acc

if __name__ == '__main__':
    main()