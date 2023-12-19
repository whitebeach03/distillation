import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():           
    epochs = 150
    
    # setting iteration
    if epochs == 200:
        teacher_iter = 5
        student_iter = 1
        st_iter      = 3 
        cam01_iter   = 2
        cam02_iter   = 3
        cam03_iter   = 0
        cam04_iter   = 3
        cam05_iter   = 0
        cam10_iter   = 1
    elif epochs == 100: 
        teacher_iter = 5 
        student_iter = 0 
        st_iter      = 5 
        cam01_iter   = 5 
        cam02_iter   = 0 
        cam03_iter   = 0
        cam04_iter   = 0
        cam05_iter   = 0
        cam10_iter   = 5
    elif epochs == 150: # 乱数系列 seed_everything
        teacher_iter = 10 
        student_iter = 0 
        st_iter      = 10 
        cam00_iter   = 5
        cam01_iter   = 10 
        cam02_iter   = 10
        cam03_iter   = 8 #NOW ice9(8,9)
        cam04_iter   = 2 #NOW synapse(2,3)
        cam05_iter   = 0 #TODO 0, 4
        cam10_iter   = 0
    
    # setting path
    teacher_path = './history/resnet/teacher/'
    student_path = './history/resnet/student/'
    st_path      = './history/resnet/st/'
    cam00_path   = './history/resnet/cam/00_'
    cam01_path   = './history/resnet/cam/01_'
    cam02_path   = './history/resnet/cam/02_'
    cam03_path   = './history/resnet/cam/03_'
    cam04_path   = './history/resnet/cam/04_'
    cam05_path   = './history/resnet/cam/05_'
    cam10_path   = './history/resnet/cam/10_'

    
    # print test accuracy
    teacher_avg  = load_avg_test(teacher_path, epochs, teacher_iter)
    teacher_best = load_best_test(teacher_path, epochs, teacher_iter)   
    student_avg  = load_avg_test(student_path, epochs, student_iter)
    student_best = load_best_test(student_path, epochs, student_iter)    
    st_avg       = load_avg_test(st_path, epochs, st_iter)
    st_best      = load_best_test(st_path, epochs, st_iter)    
    cam00_avg    = load_avg_test(cam00_path, epochs, cam00_iter)
    cam00_best   = load_best_test(cam00_path, epochs, cam00_iter) 
    cam01_avg    = load_avg_test(cam01_path, epochs, cam01_iter)
    cam01_best   = load_best_test(cam01_path, epochs, cam01_iter) 
    cam02_avg    = load_avg_test(cam02_path, epochs, cam02_iter)
    cam02_best   = load_best_test(cam02_path, epochs, cam02_iter)
    cam03_avg    = load_avg_test(cam03_path, epochs, cam03_iter)
    cam03_best   = load_best_test(cam03_path, epochs, cam03_iter) 
    cam04_avg    = load_avg_test(cam04_path, epochs, cam04_iter)
    cam04_best   = load_best_test(cam04_path, epochs, cam04_iter)
    cam05_avg    = load_avg_test(cam05_path, epochs, cam05_iter)
    cam05_best   = load_best_test(cam05_path, epochs, cam05_iter)
    cam10_avg    = load_avg_test(cam10_path, epochs, cam10_iter)
    cam10_best   = load_best_test(cam10_path, epochs, cam10_iter)
   
    print('| Teacher          | avg: ' + str(teacher_avg) + ' | best: ' + str(teacher_best) + ' |')
    print('| Student          | avg: ' + str(student_avg) + ' | best: ' + str(student_best) + ' |')
    print('| Distillation     | avg: ' + str(st_avg)      + ' | best: ' + str(st_best)      + ' |')
    print('| Proposed(1.0)    | avg: ' + str(cam00_avg)   + ' | best: ' + str(cam00_best)   + ' |')
    print('| Proposed(0.1)    | avg: ' + str(cam01_avg)   + ' | best: ' + str(cam01_best)   + ' |')
    print('| Proposed(0.2)    | avg: ' + str(cam02_avg)   + ' | best: ' + str(cam02_best)   + ' |')
    print('| Proposed(0.3)    | avg: ' + str(cam03_avg)   + ' | best: ' + str(cam03_best)   + ' |')
    print('| Proposed(0.4)    | avg: ' + str(cam04_avg)   + ' | best: ' + str(cam04_best)   + ' |')
    print('| Proposed(0.5)    | avg: ' + str(cam05_avg)   + ' | best: ' + str(cam05_best)   + ' |')
    print('| Proposed(0.1->0) | avg: ' + str(cam10_avg)   + ' | best: ' + str(cam10_best)   + ' |')
    
    # loading history and plot
    teacher_acc = load_hist(teacher_path, epochs, teacher_iter)
    student_acc = load_hist(student_path, epochs, student_iter)
    st_acc      = load_hist(st_path, epochs, st_iter)
    cam00_acc   = load_hist(cam00_path, epochs, cam00_iter)
    cam01_acc   = load_hist(cam01_path, epochs, cam01_iter)
    cam02_acc   = load_hist(cam02_path, epochs, cam02_iter)
    cam03_acc   = load_hist(cam03_path, epochs, cam03_iter)
    cam04_acc   = load_hist(cam04_path, epochs, cam04_iter)
    cam05_acc   = load_hist(cam05_path, epochs, cam05_iter)
    cam10_acc   = load_hist(cam10_path, epochs, cam10_iter)

    x = np.arange(epochs)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel('epoch')
    plt.ylabel('validation accuracy')
    # plt.plot(x, teacher_acc, label='Teacher',               linewidth=0.5, color='blue')
    # plt.plot(x, student_acc, label='Student',               linewidth=0.5, color='red')
    plt.plot(x, st_acc,      label='Distillation',      linewidth=0.5, color='orange')
    # plt.plot(x, cam00_acc,   label='Proposed(only)',    linewidth=0.5, color='red')
    plt.plot(x, cam01_acc,   label='Proposed(0.1)',     linewidth=0.5, color='green')
    plt.plot(x, cam02_acc,   label='Proposed(0.2)',     linewidth=0.5, color='magenta')
    plt.plot(x, cam03_acc,   label='Proposed(0.3)',    linewidth=0.5, color='black')
    plt.plot(x, cam04_acc,   label='Proposed(0.4)',    linewidth=0.5, color='cyan')
    # plt.plot(x, cam05_acc,   label='Proposed(0.5)',     linewidth=0.5, color='black')
    # plt.plot(x, cam10_acc,   label='Proposed(rate=0.1->0)', linewidth=0.5, color='green')
    plt.xticks(np.arange(0, epochs+10, epochs/10))
    plt.yticks(np.arange(0, 0.95, 0.05))
    plt.xlim(0, epochs+2)
    plt.ylim(0.60, 0.90)
    plt.legend()
    plt.savefig('./result/resnet_' + str(epochs) + '.png')
    
    # plot CAM_loss
    cam00_loss = load_camloss(cam00_path, epochs, cam00_iter)
    cam01_loss = load_camloss(cam01_path, epochs, cam01_iter)
    cam02_loss = load_camloss(cam02_path, epochs, cam02_iter)
    cam03_loss = load_camloss(cam03_path, epochs, cam03_iter)
    cam04_loss = load_camloss(cam04_path, epochs, cam04_iter)
    cam05_loss = load_camloss(cam05_path, epochs, cam05_iter)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel('epoch')
    plt.ylabel('CAM Loss')
    plt.plot(x, cam00_loss, label='CAM_ratio=0.0', linewidth=0.5, color='red')
    plt.plot(x, cam01_loss, label='CAM_ratio=0.1', linewidth=0.5, color='cyan')
    plt.plot(x, cam02_loss, label='CAM_ratio=0.2', linewidth=0.5, color='magenta')
    plt.plot(x, cam03_loss, label='CAM_ratio=0.3', linewidth=0.5, color='yellow')
    # plt.plot(x, cam04_loss, label='CAM_ratio=0.4', linewidth=0.5, color='blown')
    # plt.plot(x, cam05_loss, label='CAM_ratio=0.5', linewidth=0.5, color='black')
    plt.xticks(np.arange(0, epochs+10, epochs/10))
    plt.savefig('./result/CAMLoss_' + str(epochs) + '.png')
    

def load_hist(path, epochs, iteration):
    if iteration == 0:
        return 0
    dic = {}
    for i in range(iteration):
        with open(path + str(epochs) + '_' + str(i) + '.pickle', mode='rb') as f:
            dic[i] = pickle.load(f)
    val_acc = np.zeros(len(dic[i]['val_accuracy']))
    for i in range(iteration):
        val_acc += np.array(dic[i]['val_accuracy'])
    val_acc = val_acc / iteration
    return val_acc

def load_best_test(path, epochs, iteration):
    if iteration == 0:
        return 0
    dic = {}
    for i in range(iteration):
        with open(path + str(epochs) + '_' + 'test' + str(i) + '.pickle', mode='rb') as f:
            dic[i] = pickle.load(f)
    best_acc = 0
    for i in range(iteration):
        if dic[i]['acc'][0] >= best_acc:
            best_acc = dic[i]['acc'][0]
    best_acc *= 100
    best_acc = round(best_acc, 2)
    return best_acc

def load_avg_test(path, epochs, iteration):
    if iteration == 0:
        return 0
    dic = {}
    for i in range(iteration):
        with open(path + str(epochs) + '_' + 'test' + str(i) + '.pickle', mode='rb') as f:
            dic[i] = pickle.load(f)
    avg_acc = 0
    for i in range(iteration):
        avg_acc += dic[i]['acc'][0]
    avg_acc = avg_acc / iteration
    avg_acc *= 100
    avg_acc = round(avg_acc, 2)
    return avg_acc

def load_camloss(path, epochs, iteration):
    if iteration == 0:
        return 0
    dic = {}
    for i in range(iteration):
        with open(path + str(epochs) + '_' + str(i) + '.pickle', mode='rb') as f:
            dic[i] = pickle.load(f)
    cam_loss = np.zeros(len(dic[i]['cam_loss']))
    for i in range(iteration):
        arr = dic[i]['cam_loss']
        for j in range(len(dic[i]['cam_loss'])):
            dic[i]['cam_loss'][j] = dic[i]['cam_loss'][j].detach().cpu().numpy()
        cam_loss += np.array(dic[i]['cam_loss'])
    cam_loss = cam_loss / iteration
    return cam_loss

if __name__ == '__main__':
    main()
