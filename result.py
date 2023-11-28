import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():           
    # setting path
    student_path = './history/resnet/student/'
    teacher_path = './history/resnet/teacher/'
    st_path = './history/resnet/st/'
    cam01_path = './history/resnet/cam/01_'
    cam02_path = './history/resnet/cam/02_'
    cam05_path = './history/resnet/cam/05_'
    
    # loading history
    student_acc = load_hist(student_path, 1)
    teacher_acc = load_hist(teacher_path, 5)
    st_acc = load_hist(st_path, 1)
    cam01_acc = load_hist(cam01_path, 1)
    cam02_acc = load_hist(cam02_path, 1)
    cam05_acc = load_hist(cam05_path, 1)
    
    # print test accuracy
    student_avg = load_avg_test(student_path, 1)
    student_best = load_best_test(student_path, 1)   
    teacher_avg = load_avg_test(teacher_path, 5)
    teacher_best = load_best_test(teacher_path, 5)    
    st_avg = load_avg_test(st_path, 1)
    st_best = load_best_test(st_path, 1)    
    cam01_avg = load_avg_test(cam01_path, 1)
    cam01_best = load_best_test(cam01_path, 1) 
    cam02_avg = load_avg_test(cam02_path, 1)
    cam02_best = load_best_test(cam02_path, 1)
    cam05_avg = load_avg_test(cam05_path, 1)
    cam05_best = load_best_test(cam05_path, 1)
      
    print('| Student             | avg: ' + str(student_avg) + ' | best: ' + str(student_best) + ' |')
    print('| Teacher             | avg: ' + str(teacher_avg) + ' | best: ' + str(teacher_best) + ' |')
    print('| Distillation        | avg: ' + str(st_avg)      + ' | best: ' + str(st_best) + ' |')
    print('| Proposed(rate=0.1)  | avg: ' + str(cam01_avg)   + ' | best: ' + str(cam01_best) + ' |')
    print('| Proposed(rate=0.2)  | avg: ' + str(cam02_avg)   + ' | best: ' + str(cam02_best) + ' |')
    print('| Proposed(rate=0.5)  | avg: ' + str(cam05_avg)   + ' | best: ' + str(cam05_best) + ' |')
    
    # plot result
    x = np.arange(200)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel('epoch')
    plt.ylabel('validation accuracy')
    
    plt.plot(x, student_acc, label='Student',            linewidth=0.5, color='red')
    plt.plot(x, teacher_acc, label='Teacher',            linewidth=0.5, color='blue')
    plt.plot(x, st_acc,      label='Distillation',       linewidth=0.5, color='orange')
    plt.plot(x, cam01_acc,   label='Proposed(rate=0.1)', linewidth=0.5, color='green')
    plt.plot(x, cam02_acc,   label='Proposed(rate=0.2)', linewidth=0.5, color='black')
    plt.plot(x, cam05_acc,   label='Proposed(rate=0.3)', linewidth=0.5, color='brown')
    
    plt.xticks(np.arange(0, 210, 20))
    plt.yticks(np.arange(0, 0.95, 0.05))
    plt.xlim(0, 201)
    plt.ylim(0.70, 0.90)
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