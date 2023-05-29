import argparse
import tensorflow as tf
from network import *
import numpy as np
import random
import math
import os
from data_processing import DataLoader_atten_polar
from sklearn.metrics import roc_auc_score
import platform
from icecream import ic
from datetime import datetime

mode='Glim-net'
test_num = 348
lr_change= [3*1e-8, 1*1e-7, 6*1e-6, 6*1e-6, 1*1e-5, 5*1e-7]
strategy_num=5
strategy_list = [680, 172, 172, 88, 80, 48]
strategy_epoch_duration=[5,5,5,5,5,5]
Epoch = 400
epoch_test = 1
batch_size= 4


parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str, help='exp name')
parser.add_argument('--seed', type=int,default=1, help='seed')
parser.add_argument('--strategy_list', type=int,nargs='+', help='strategy list')
parser.add_argument('--strategy_num', type=int,default=5, help='strategy_num')
parser.add_argument('--epoch', type=int,default=400, help='epoch number')
parser.add_argument('--batch_size', type=int,default=4, help='batch_size')

args = parser.parse_args()

random_seed = args.seed
strategy_list=args.strategy_list
strategy_num=args.strategy_num
Epoch = args.epoch
batch_size = args.batch_size

exp_name = args.name
run_id = datetime.now().strftime(r'%m%d_%H%M%S')
save_path=f'{exp_name}/{run_id}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

acc_txt=f"{save_path}/acc.txt"
log_txt=f"{save_path}/log.txt"
test_details=f"{save_path}/test_detail.txt"
config_txt=f"{save_path}/config.txt"
throw_txt=f"{save_path}/throw.txt"


with open(config_txt, "w+")as f0:
    f0.write('mode: ' + mode)
    f0.write('\n')
    f0.write('strategy_num: %d ' % strategy_num)
    f0.write('\n')
    f0.write('epoch_test: %d' % epoch_test)
    f0.write('\n')
    f0.write('All epoch: %d ' % Epoch)
    f0.write('\n')
    for i in (strategy_list):
        f0.write('strategy_list: %d ' % i)
        f0.write('\n')
    for i in (lr_change):
        f0.write('lr_change: %.9f ' % i)
        f0.write('\n')
    f0.flush()


def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)

def main():
    g1 = tf.Graph()
    with g1.as_default():
        n_steps = 5
        best_model = -1

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf.set_random_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        batch_size_val = batch_size

        input = tf.placeholder(tf.float32, (batch_size, n_steps, 224, 224, 3), name='inputx')
        GT_label = tf.placeholder(tf.int64, (batch_size, n_steps), name='GT_labelx')   #size = [batch, n_steps

        label_polar_map = tf.placeholder(tf.float32, (batch_size, n_steps, 224, 224, 3), name='label_polar_mapx')
        delta_year=tf.placeholder(tf.float32, (batch_size, n_steps), name='delta_yearx')
        # time matrix
        time_matrix=tf.placeholder(tf.float32, (batch_size, n_steps, n_steps), name='time_matrix')

        is_training = tf.placeholder(tf.bool, name='is_training')

        label_predict_op = inference_model(input,GT_label,label_polar_map,delta_year,time_matrix, is_training)

        lr = tf.placeholder(tf.float32, shape=[])

        loss_per_batch=Loss_per_batch(label_predict_op, GT_label)

        loss_op = loss_liliu(label_predict_op, GT_label)   #[batch,n_steps]
        acc_op = top_k_accuracy(label_predict_op[0], GT_label)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss_op)

        saver = tf.train.Saver(max_to_keep=50)
        
        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())

            list_img_path_train_init = os.listdir('./data/train/image/all/')
            list_img_path_train_init.sort()
            list_img_path_train=list_img_path_train_init
            list_img_path_test = os.listdir('./data/test/image/all/')
            list_img_path_test.sort()
            """train"""
            dataloader_train = DataLoader_atten_polar(batch_size=batch_size, list_img_path= list_img_path_train,state='train')
            ic(dataloader_train.size)
            """test"""
            dataloader_test = DataLoader_atten_polar(batch_size=batch_size,list_img_path=list_img_path_test, state='test')
            """train_strategy"""
            dataloader_strategy = DataLoader_atten_polar(batch_size=batch_size, list_img_path=list_img_path_train, state='train')

            count = 0
            count_strategy=0
            flag_strategy = True
            print("Start Training!")
            with open(acc_txt, "w+") as f:
                with open(log_txt, "w+")as f2:
                    for epoch in range(0, Epoch): 
                        print('\nEpoch: %d' % (epoch + 1))
                        """-----------------------------------------throw------------------------------------------"""
                        if flag_strategy and (epoch % strategy_epoch_duration[count_strategy]) == 0 and epoch>0:
                            if count_strategy == (strategy_num-1):
                                flag_strategy = False

                            count_strategy += 1  # 1~10
                            train_leftnum_before = strategy_list[count_strategy-1]
                            train_leftnum_now = strategy_list[count_strategy]
                            loss_trainingset = [0 for i_loss_train in range(train_leftnum_before)]
                            list_img_path_train_before = list_img_path_train

                            for jj in range(int(train_leftnum_before / batch_size)):
                                image_train_test, year_train_test, _, Polar_train_test, GTlabel_train_test,my_time_matrix = dataloader_strategy.get_batch(shuffle=False)
                                loss_per_batch_1 = sess.run(loss_per_batch, feed_dict={input: image_train_test,
                                                                                       GT_label: GTlabel_train_test,
                                                                                       label_polar_map: Polar_train_test,
                                                                                       delta_year: year_train_test,
                                                                                       time_matrix: my_time_matrix,
                                                                                       lr:lr_change[count_strategy],
                                                                                       is_training:False})
 
                                for i_loss_chosen in range(batch_size):
                                    loss_trainingset[jj * 4 + i_loss_chosen] = loss_per_batch_1[i_loss_chosen]

                            matrix1 = np.zeros([train_leftnum_before, 2], dtype=np.float)
                            matrix2 = np.zeros([train_leftnum_now, 2], dtype=np.float)
                            for i_1 in range(train_leftnum_before):
                                matrix1[i_1] = [loss_trainingset[i_1], i_1]
                            matrix1 = sorted(matrix1, key=lambda cus: cus[0], reverse=True)  # big-->small

                            for i_2 in range(train_leftnum_now):
                                matrix2[i_2] = matrix1[i_2]
                            for i_3 in range(train_leftnum_now):
                                if i_3 == 0:
                                    list_img_path_train0 = [list_img_path_train_before[int(matrix2[i_3][1])]]
                                else:
                                    list_img_path_train0.append(list_img_path_train_before[int(matrix2[i_3][1])])
                            list_img_path_train = list_img_path_train0
                            list_img_path_train.sort()

                            with open(throw_txt, "a")as f3:
                                f3.write('strategy: %d ' % count_strategy)
                                f3.write('\n')
                                f3.flush()
                                for i_f3 in range(train_leftnum_before-train_leftnum_now):
                                    f3.write(list_img_path_train_before[int(matrix1[train_leftnum_now+i_f3][1])]+' : ')
                                    f3.flush()
                                    with open('data/train/label/all/'+ list_img_path_train_before[int(matrix1[train_leftnum_now+i_f3][1])] + '.txt', 'r') as f4:
                                        K = f4.readlines()
                                        for i_line in range(5):
                                            line = K[i_line + 1]
                                            line = line.strip('\n')
                                            line = int(line)
                                            f3.write(str(line))
                                            f3.flush()

                                    f3.write('\n')
                                    f3.flush()
                                if count_strategy == strategy_num:
                                    f3.write('Last left: ')
                                    f3.write('\n')
                                    f3.flush()
                                    for i_f3_lastleft in range(train_leftnum_now):
                                        f3.write(list_img_path_train[i_f3_lastleft] + ' : ')
                                        f3.flush()
                                        with open('data/train/label/all/' + list_img_path_train[i_f3_lastleft] + '.txt', 'r') as f5:
                                            K = f5.readlines()
                                            for i_line in range(5):
                                                line = K[i_line + 1]
                                                line = line.strip('\n')
                                                line = int(line)
                                                f3.write(str(line))
                                                f3.flush()
                                        f3.write('\n')
                                        f3.flush()


                            dataloader_train = DataLoader_atten_polar(batch_size=batch_size,list_img_path=list_img_path_train, state='train')
                            dataloader_strategy = DataLoader_atten_polar(batch_size=batch_size,list_img_path=list_img_path_train, state='train')
                            print('\nEpoch: %d, train_num: %d' % (epoch + 1, dataloader_train.size))
                        """-----------------------------------------train------------------------------------------"""
 
                        for i in range(int(strategy_list[count_strategy]/batch_size)):
                            # sota
                            image1,year1,GTmap1,Polar1, GTlabel1, time_matrix1= dataloader_train.get_batch()
                            
                            loss_train,_, acc,label_predict = sess.run([ loss_op,train_op, acc_op,label_predict_op],
                                                                       feed_dict={input: image1, 
                                                                                GT_label: GTlabel1,
                                                                                label_polar_map:Polar1,
                                                                                delta_year:year1,
                                                                                time_matrix:time_matrix1,
                                                                                lr:lr_change[count_strategy],
                                                                                is_training:True})

   
                            f2.write('strategy:%d, epoch:%03d  %05d |Loss: %.03f | Acc: %.3f%%' % (
                                count_strategy ,epoch , (i + 1), loss_train, 100. * acc))
                            print('strategy:%d, epoch:%03d  %05d |Loss: %.03f | Acc: %.3f%%' % (
                                count_strategy ,epoch , (i + 1), loss_train, 100. * acc))
                            f2.write('\n')
                            f2.flush()
                            count +=1


                        """-----------------------------------------test------------------------------------------"""
                        ic(list_img_path_test[:3])
                        with open(test_details, "a")as f6:
                            f6.write('epoch: %d' % epoch)
                            f6.write('\n')
                            f6.flush()
                            if epoch % epoch_test == 0:
                                print("testing")
                                tp = 0.0
                                fn = 0.0
                                tn = 0.0
                                fp = 0.0
                                y_true = [0.0 for _ in range(test_num * n_steps)]
                                y_scores = [0.0 for _ in range(test_num * n_steps)]

                                for j in range(int(test_num / batch_size_val)):


                                    imagev, yearv, GTmapv, Polarv, GTlabelv,my_time_matrix = dataloader_test.get_batch(shuffle=False)
                                    label_predict = sess.run(label_predict_op,
                                                             feed_dict={input: imagev, GT_label: GTlabelv,
                                                                        label_polar_map: Polarv,
                                                                        delta_year: yearv,
                                                                        time_matrix: my_time_matrix,
                                                                        is_training:False
                                                                        })

                                    GTlabelv_test = np.reshape(GTlabelv, [-1])  # batch_size* n_steps
                                    label_predict_test = np.reshape(label_predict[0], [-1, 2])  # batch_size*n_steps,2
                                    label_predict_0 = label_predict_test[:, 0]  # batch_size* n_steps
                                    label_predict_1 = label_predict_test[:, 1]  # batch_size* n_steps

                                    """----------------------------tptn---------------------------------"""
                                    for nb in range(batch_size_val * n_steps):
                                        if GTlabelv_test[nb] == 1 and (label_predict_1[nb] > label_predict_0[nb]):
                                            tp = tp + 1
                                        if GTlabelv_test[nb] == 0 and (label_predict_1[nb] < label_predict_0[nb]):
                                            tn = tn + 1
                                        if GTlabelv_test[nb] == 1 and (label_predict_1[nb] < label_predict_0[nb]):
                                            fn = fn + 1
                                        if GTlabelv_test[nb] == 0 and (label_predict_1[nb] > label_predict_0[nb]):
                                            fp = fp + 1
                                    """----------------------------AUC---------------------------------"""
                                    for nb in range(batch_size_val * n_steps):  # 20
                                        y_true[j * (batch_size_val * n_steps) + nb] = GTlabelv_test[nb]
                                        y_scores[j * (batch_size_val * n_steps) + nb] = (math.exp(   
                                            label_predict_1[nb])) / (math.exp(label_predict_1[nb]) + math.exp(                                         
                                            label_predict_0[nb]))
                                    """----------------------------print all result of 384---------------------------------"""
                                    for batch in range(batch_size_val):
                                        if j * 4 + batch == test_num:
                                            break
                                        f6.write('%s   :' % list_img_path_test[j * 4 + batch])
                                        f6.flush()
                                        for img_perimgpath in range(5):
                                            f6.write('%d' % GTlabelv[batch][img_perimgpath])
                                            f6.flush()
                                        f6.write(' ')
                                        f6.flush()
                                        for img_perimgpath in range(5):
                                            if label_predict[0][batch][img_perimgpath][1] > \
                                                    label_predict[0][batch][img_perimgpath][0]:
                                                f6.write('1')
                                                f6.flush()
                                            if label_predict[0][batch][img_perimgpath][1] < \
                                                    label_predict[0][batch][img_perimgpath][0]:
                                                f6.write('0')
                                                f6.flush()
                                        f6.write('   ')
                                        f6.flush()
                                        for img_perimgpath in range(5):
                                            p_glau = math.exp(label_predict[0][batch][img_perimgpath][1]) / (
                                                        math.exp(label_predict[0][batch][img_perimgpath][1]) + math.exp(
                                                    label_predict[0][batch][img_perimgpath][0]))
                                            f6.write('%.03f%% ' % (100. * p_glau))
                                            f6.flush()
                                        f6.write('\n')
                                        f6.flush()

                                acc = (tp + tn) / (tp + tn + fp + fn)
                                Sen = tp / (tp + fn)
                                Spe = tn / (tn + fp)

                                AUC = roc_auc_score(y_true, y_scores)
                                print("test accuracy: %.03f%% |test sen: %.03f%% |test spe: %.03f%% |test AUC: %.03f%%" % (
                                    100. * acc, 100. * Sen, 100. * Spe, 100. * AUC))
                                f.write(
                                    ' epoch %03d  | Acc: %.3f%% | sen: %.3f%% | spe: %.3f%% |AUC: %.3f%%|tp: %.3f | tn: %.3f| fp: %.3f | fn: %.3f' % (
                                        epoch, 100. * acc, 100. * Sen, 100. * Spe, 100. * AUC, tp, tn,
                                        fp, fn))
                                f.write('\n')
                                f.flush()


                        if epoch>0 and AUC > best_model and Sen > 0.857 and Spe > 0.856:
                            best_model = AUC
                            saver.save(sess, f'{save_path}/best_model.ckpt')
                        if Sen>0.86 and Spe >0.87:
                            saver.save(sess, f'{save_path}/{epoch}.ckpt')



if __name__ == '__main__':
    # remove_all_file(save_path)
    if platform.system() =='Linux':
        if (os.path.exists(throw_txt)):
            os.remove(throw_txt)
        os.mknod(throw_txt)
        if (os.path.exists(test_details)):
            os.remove(test_details)
        os.mknod(test_details)
    main()
