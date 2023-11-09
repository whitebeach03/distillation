import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    # setting history path
    student_path = 'history/student/0'
    teacher_path = 'history/teacher/0'
    st_path = 'history/student_st/0'

    # loading history
    student_acc = load_hist(student_path, 5)
    teacher_acc = load_hist(teacher_path, 5)
    st_acc = load_hist(st_path, 1)
    
    # print test accuracy
    student_avg_acc = load_avg_test('history/student/test0', 5)
    student_best_acc = load_best_test('history/student/test0', 5)
    teacher_avg_acc = load_avg_test('history/teacher/test0', 5)
    teacher_best_acc = load_best_test('history/teacher/test0', 5)
    # st_avg_acc = load_avg_test('history/student_st/test0', 5)
    # st_best_acc = load_best_test('history/student_st/test0', 5)
    
    print('Student avg_acc: ' + str(student_avg_acc) + ' best_acc: ' + str(student_best_acc))
    print('Teacher avg_acc: ' + str(teacher_avg_acc) + ' best_acc: ' + str(teacher_best_acc))
    # print('Distillation avg_acc: ' + str(st_avg_acc) + ' best_acc: ' + str(st_best_acc))
    
    # plot result
    x = np.arange(50)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')

    plt.plot(x, student_acc, label='Student', linewidth=0.5, color='red')
    plt.plot(x, teacher_acc, label='Teacher', linewidth=0.5, color='blue')
    plt.plot(x, st_acc, label='Distillation', linewidth=0.5, color='orange')

    plt.xticks(np.arange(0, 55, 5))
    plt.yticks(np.arange(0.60, 0.90, 0.05))
    plt.xlim(0, 51)
    plt.ylim(0.60, 0.90)

    plt.legend()
    plt.savefig('./result/result00.png')


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
        with open(path + str(i) + '.pickle', mode='rb') as f:
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
        with open(path + str(i) + '.pickle', mode='rb') as f:
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