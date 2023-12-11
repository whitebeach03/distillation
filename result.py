import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():           
    epochs = 200
    
    # iteration
    if epochs == 200:
        teacher_iter = 5
        student_iter = 1
        st_iter      = 3 
        cam01_iter   = 2
        cam02_iter   = 3
        cam04_iter   = 3
        cam10_iter   = 1
        
    elif epochs == 100: # seed_everythingで3回ずつやる(今はTeacher3回やってる) -> train_kd -> train_cam の順番
        teacher_iter = 3 # 5
        student_iter = 0 # 0
        st_iter      = 2 # 5
        cam01_iter   = 0 # 3
        cam02_iter   = 0 # 3
        cam10_iter   = 2 # 0
    
    # setting path
    student_path = './history/resnet/student/'
    teacher_path = './history/resnet/teacher/'
    st_path      = './history/resnet/st/'
    cam01_path   = './history/resnet/cam/01_'
    cam02_path   = './history/resnet/cam/02_'
    cam04_path   = './history/resnet/cam/04_'
    cam10_path   = './history/resnet/cam/10_'
    
    # loading history
    # student_acc = load_hist(student_path, epochs, student_iter)
    # teacher_acc = load_hist(teacher_path, epochs, teacher_iter)
    st_acc      = load_hist(st_path, epochs, st_iter)
    # cam01_acc   = load_hist(cam01_path, epochs, cam01_iter)
    # cam02_acc   = load_hist(cam02_path, epochs, cam02_iter)
    # cam04_acc   = load_hist(cam04_path, epochs, cam04_iter)
    cam10_acc   = load_hist(cam10_path, epochs, cam10_iter)
    
    # print test accuracy
    # student_avg  = load_avg_test(student_path, epochs, student_iter)
    # student_best = load_best_test(student_path, epochs, student_iter)   
    # teacher_avg  = load_avg_test(teacher_path, epochs, teacher_iter)
    # teacher_best = load_best_test(teacher_path, epochs, teacher_iter)    
    st_avg       = load_avg_test(st_path, epochs, st_iter)
    st_best      = load_best_test(st_path, epochs, st_iter)    
    # cam01_avg    = load_avg_test(cam01_path, epochs, cam01_iter)
    # cam01_best   = load_best_test(cam01_path, epochs, cam01_iter) 
    # cam02_avg    = load_avg_test(cam02_path, epochs, cam02_iter)
    # cam02_best   = load_best_test(cam02_path, epochs, cam02_iter)
    # cam04_avg    = load_avg_test(cam04_path, epochs, cam04_iter)
    # cam04_best   = load_best_test(cam04_path, epochs, cam04_iter)
    cam10_avg    = load_avg_test(cam10_path, epochs, cam10_iter)
    cam10_best   = load_best_test(cam10_path, epochs, cam10_iter)
    
    # print('| Student               | avg: ' + str(student_avg)  + ' | best: ' + str(student_best) + ' |')
    # print('| Teacher               | avg: ' + str(teacher_avg)  + ' | best: ' + str(teacher_best) + ' |')
    print('| Distillation           | avg: ' + str(st_avg)        + ' | best: ' + str(st_best)       + ' |')
    # print('| Proposed(rate=0.1)     | avg: ' + str(cam01_avg)     + ' | best: ' + str(cam01_best)    + ' |')
    # print('| Proposed(rate=0.2)    | avg: ' + str(cam02_avg)    + ' | best: ' + str(cam02_best)   + ' |')
    # print('| Proposed(rate=0.4)    | avg: ' + str(cam04_avg)    + ' | best: ' + str(cam04_best)   + ' |')
    print('| Proposed(rate=0.1->0)  | avg: ' + str(cam10_avg)     + ' | best: ' + str(cam10_best)    + ' |')
    
    # plot result
    x = np.arange(epochs)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel('epoch')
    plt.ylabel('validation accuracy')
    
    # plt.plot(x, student_acc, label='Student',               linewidth=0.5, color='red')
    # plt.plot(x, teacher_acc, label='Teacher',               linewidth=0.5, color='blue')
    plt.plot(x, st_acc,       label='Distillation',           linewidth=0.5, color='orange')
    # plt.plot(x, cam01_acc,    label='Proposed(rate=0.1)',     linewidth=0.5, color='green')
    # plt.plot(x, cam02_acc,   label='Proposed(rate=0.2)',    linewidth=0.5, color='black')
    # plt.plot(x, cam04_acc,   label='Proposed(rate=0.4)',    linewidth=0.5, color='brown')
    plt.plot(x, cam10_acc,    label='Proposed(rate=0.1->0)',  linewidth=0.5, color='black')
    
    plt.xticks(np.arange(0, epochs+10, epochs/10))
    plt.yticks(np.arange(0, 0.95, 0.05))
    plt.xlim(0, 100)
    plt.ylim(0.60, 0.90)
    plt.legend()
    plt.savefig('./result/result_' + str(epochs) + '.png')
    
    # cam_loss = load_camloss(cam01_path, epochs, cam01_iter)
    # fig = plt.figure()
    # fig.patch.set_facecolor('white')
    # plt.xlabel('epoch')
    # plt.ylabel('CAM Loss')
    # plt.plot(x, cam_loss, label='CAM-Distillation (0.1)', linewidth=0.5, color='red')
    # plt.xticks(np.arange(0, epochs+10, epochs/10))
    # plt.savefig('./result/CAMLoss.png')
    

def load_hist(path, epochs, iteration):
    dic = {}
    for i in range(iteration):
        with open(path + str(epochs) + '_' + str(i) + '.pickle', mode='rb') as f:
            dic[i] = pickle.load(f)
    val_acc = np.zeros(len(dic[i]['val_accuracy']))
    for i in range(iteration):
        val_acc += np.array(dic[i]['val_accuracy'])
    val_acc = val_acc / iteration
    return val_acc

def load_camloss(path, epochs, iteration):
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

def load_best_test(path, epochs, iteration):
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

if __name__ == '__main__':
    main()