import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    # setting history path
    student_path = './history/student/'
    teacher_path = './history/teacher/'
    st_path = './history/student_st/'
    student_cam_path = './history/student_cam/sample'
    
    # loading history
    s_train_loss, s_train_acc, s_val_loss, s_val_acc = load_hist(student_path, 5)
    t_train_loss, t_train_acc, t_val_loss, t_val_acc = load_hist(teacher_path, 5)
    st_train_loss, st_train_acc, st_val_loss, st_val_acc = load_hist(st_path, 5)
    cam_train_loss, cam_train_acc, cam_val_loss, cam_val_acc = load_hist(student_cam_path, 1)
    
    # print best test accuracy
    student_avg = load_avg_test(student_path, 5)
    student_best = load_best_test(student_path, 5)
    teacher_avg = load_avg_test(teacher_path, 5)
    teacher_best = load_best_test(teacher_path, 5)
    st_avg = load_avg_test(st_path, 5)
    st_best = load_best_test(st_path, 5)
    print('Student avg: ' + str(student_avg) + ' best_acc: ' + str(student_best))
    print('Teacher avg: ' + str(teacher_avg) + ' best_acc: ' + str(teacher_best))
    print('Distillation avg: ' + str(st_avg) + ' best_acc: ' + str(st_best))

    # plot result
    x = np.arange(100)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')

    plt.plot(x, s_val_acc, label='Student', linewidth=0.5, color='red')
    plt.plot(x, t_val_acc, label='Teacher', linewidth=0.5, color='blue')
    plt.plot(x, st_val_acc, label='Distillation', linewidth=0.5, color='orange')
    plt.plot(x, cam_val_acc, label='CAM-Distillation', linewidth=0.5, color='green')

    plt.xticks(np.arange(0, 110, 10))
    plt.yticks(np.arange(0.60, 0.90, 0.05))
    plt.xlim(0, 103)
    plt.ylim(0.60, 0.855)

    plt.legend()
    plt.savefig('./result/result.png')

def load_hist(path, iteration):
    dic = {}
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

def load_best_test(path, iteration):
    dic = {}
    for i in range(iteration):
        with open(path + 'test' + str(i) + '.pickle', mode='rb') as f:
            dic[i] = pickle.load(f)
    
    best_acc = 0
    for i in range(iteration):
        if dic[i]['acc'][0] >= best_acc:
            best_acc = dic[i]['acc'][0]
    best_acc *= 100
    best_acc = round(best_acc, 2)
    return best_acc

def load_avg_test(path, iteration):
    dic = {}
    for i in range(iteration):
        with open(path + 'test' + str(i) + '.pickle', mode='rb') as f:
            dic[i] = pickle.load(f)
    avg_acc = 0
    for i in range(iteration):
        avg_acc += dic[i]['acc'][0]
    avg_acc = avg_acc / iteration
    avg_acc *= 100
    avg_acc = round(avg_acc, 2)
    return avg_acc

if __name__ == '__main__':
    main()