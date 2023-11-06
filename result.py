import pickle
import numpy as np
import matplotlib.pyplot as plt

data_name = 'cifar10'
epoch = 100
batch_size = 128
num_class = 10
iteration = 5

def load_hist(path):
    dic = {}
    
    # load history
    for i in range(iteration):
        with open(path + str(i) + '.pickle', mode='rb') as f:
            dic[i] = pickle.load(f)   
            
    train_loss = np.zeros(len(dic[i]['loss']))
    train_acc = np.zeros(len(dic[i]['accuracy']))
    val_loss = np.zeros(len(dic[i]['val_loss']))
    val_acc = np.zeros(len(dic[i]['val_accuracy']))
    
    for i in range(iteration):
        train_loss += np.array(dic[i]['loss'])
        train_acc += np.array(dic[i]['accuracy'])
        val_loss += np.array(dic[i]['val_loss'])
        val_acc += np.array(dic[i]['val_accuracy'])

    train_loss = train_loss / iteration
    train_acc = train_acc / iteration
    val_loss = val_loss / iteration
    val_acc = val_acc / iteration
    
    return train_loss, train_acc, val_loss, val_acc


student_path = './history/student/'
teacher_path = './history/teacher/'
student_st_path = './history/student_st/'
student_cam_path = './history/student_cam/'

s_train_loss, s_train_acc, s_val_loss, s_val_acc = load_hist(student_path)
# t_train_loss, t_train_acc, t_val_loss, t_val_acc = load_hist(teacher_path)
# st_train_loss, st_train_acc, st_val_loss, st_val_acc = load_hist(student_st_path)
# cam_train_loss, cam_train_acc, cam_val_loss, cam_val_acc = load_hist(student_cam_path)

x = np.arange(100)
fig = plt.figure()
fig.patch.set_facecolor('white')
plt.xlabel('epoch')
plt.ylabel('Accuracy')

plt.plot(x, s_val_acc, label='Student', linewidth=0.5, color='red')
# plt.plot(x, t_val_acc, label='Teacher', linewidth=0.5, color='blue')
# plt.plot(x, st_val_acc, label='Distillation', linewidth=0.5, color='orange')
# plt.plot(x, cam_val_acc, label='CAM-Distillation', linewidth=0.5, color='green')

plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0.60, 0.90, 0.05))
plt.xlim(0, 100)
plt.ylim(0.65, 0.865)

plt.legend()
plt.savefig('./result.png')