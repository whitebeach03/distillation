import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    # setting path
    student_path = './history/resnet/student/'
    teacher_path = './history/resnet/teacher/'
    st_path = './history/resnet/st/'
    cam_path = './history/resnet/cam/'
    
    # loading history
    student_acc = load_hist(student_path, 1)
    teacher_acc = load_hist(teacher_path, 1)
    st_acc = load_hist(st_path, 1)
    cam_acc = load_hist(cam_path, 1)
    
    # print test accuracy
    student_avg = load_avg_test(student_path, 1)
    student_best = load_best_test(student_path, 1)
    teacher_avg = load_avg_test(teacher_path, 1)
    teacher_best = load_best_test(teacher_path, 1)
    st_avg = load_avg_test(st_path, 1)
    st_best = load_best_test(st_path, 1)
    cam_avg = load_avg_test(cam_path, 1)
    cam_best = load_best_test(cam_path, 1)
    print('Student avg: ' + str(student_avg) + ' best: ' + str(student_best))
    print('Teacher avg: ' + str(teacher_avg) + ' best: ' + str(teacher_best))
    print('Distillation avg: ' + str(st_avg) + ' best: ' + str(st_best))
    print('CAM-Distillation avg: ' + str(cam_avg) + ' best: ' + str(cam_best))
    
    # plot result
    x = np.arange(50)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel('epoch')
    plt.ylabel('validation accuracy')
    
    plt.plot(x, student_acc, label='Student', linewidth=0.5, color='red')
    plt.plot(x, teacher_acc, label='Teacher', linewidth=0.5, color='blue')
    plt.plot(x, st_acc, label='Distillation', linewidth=0.5, color='orange')
    plt.plot(x, cam_acc, label='Proposed', linewidth=0.5, color='green')
    
    plt.xticks(np.arange(0, 55, 5))
    plt.yticks(np.arange(0.60, 0.90, 0.05))
    plt.xlim(0, 51)
    plt.ylim(0.60, 0.90)
    
    plt.legend()
    plt.savefig('./result/resnet_result.png')

def load_hist(path, iteration):
    dic = {}
    for i in range(iteration):
        with open(path + str(i) + '.pickle', mode='rb') as f:
            dic[i] = pickle.load(f)
    val_acc = np.zeros(len(dic[i]['val_accuracy']))
    for i in range(iteration):
        val_acc += np.array(dic[i]['val_accuracy'])
    val_acc = val_acc / iteration
    return val_acc

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