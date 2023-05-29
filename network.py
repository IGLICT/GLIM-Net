import tensorflow as tf

from transformer import Transformer
import numpy as np
import BasicNet
class Net(BasicNet.BasicNet):

    def __init__(self, is_training):
        super(Net, self).__init__()
        self.is_training = is_training
        self.predict_label = []
        self.predict_label_f =[]
        self.predict_salmap = []
        self.mask_input =[]

        self.test =[]
        self.train = []
        self.train_semi = []
        self.fc2 = []
        self.beta1 = 10
        self.beta2 = 1
        self.beta3 = 1



    def inference(self,x, reuse=False):
        # 224*224*3->112*112*64
        conv_1 = self.conv('1conv_1', x, ksize=7, filters_out=64, stride=2, reuse=reuse)

        # feature size at 56*56*64
        max_pool_1 = self.max_pool(conv_1, ksize=3, stride=2)
        block_1_1 = self.vv_building_block('1block_1_1', max_pool_1, output_channel=64, stride=1, reuse=reuse)

        # feature size at  28*28*128
        block_2_1 = self.vv_building_block('1block_2_1', block_1_1, output_channel=128, stride=2, reuse=reuse)

        # # feature size at 14*14*256
        block_3_1 = self.vv_building_block('1block_3_1', block_2_1, output_channel=256, stride=2, reuse=reuse)

        # feature size at 7*7*512
        block_4_1 = self.vv_building_block('1block_4_1', block_3_1, output_channel=512, stride=2, reuse=reuse)


        return block_4_1  # ave_pool_1


def inference_model(x,y,label_polar_map,delta_year,time_matrix,is_training=True):         # x= [batch, n_step, w, h, c]


    
    cnn_sequential_polar=multi_CNNoutput_polar('polar',x,label_polar_map,is_training=is_training)
    transformer_input = tf.reduce_mean(cnn_sequential_polar,axis=(2,3)) #[batch_size,n_steps,512]


    transformer = Transformer()

    mask = get_trans_mask()

    memory, src_masks = transformer.encode(transformer_input, mask, delta_year,time_matrix,training=is_training)

    
    logits, y_hat, ys = transformer.decode(y, memory, src_masks, delta_year,time_matrix,training=is_training)


    return y_hat, y_hat, y_hat

def get_trans_mask():
    mask = np.empty((4,5))
    for i in range(4):
        for j in range(5):
            mask[i][j]=False
    masks = tf.convert_to_tensor(mask)
    return masks

def multi_CNNoutput_polar(scope_name,x, label_attention_map, is_training):
    net = Net(is_training)
    with tf.variable_scope(scope_name):
        n_steps =x.get_shape().as_list()[1]
        for i in range(n_steps):
            x_0 = label_attention_map[:, i, :, :, :]

            if i == 0:
                y_0 = net.inference(x_0, reuse=False)
                y_0 = tf.expand_dims(y_0, 1)
                y = y_0
            else:
                y_0 = net.inference(x_0, reuse=True)
                y_0 = tf.expand_dims(y_0, 1)
                y = tf.concat([y,y_0], 1)

    return y

def loss_liliu(predict_label, labGT):  # label_predict_op=(batch, n_steps, n_outputs)  label=[batch, n_steps]
    weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)

    loss_weight = tf.add_n(weight_loss)


    batch_size = labGT.get_shape().as_list()[0]
    n_step = labGT.get_shape().as_list()[1]

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_label[0], labels=labGT)

    loss_label = tf.reduce_sum(ce) / (batch_size * n_step)

    # loss_p

    loss = loss_weight + loss_label  # + loss_direction

    return loss

def Loss_per_batch(predict_label, labGT):  # label_predict_op=(batch, n_steps, n_outputs)  label=[batch, n_steps]
    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict_label[0], labels=labGT) #[batch_size,n_steps]

    loss_label = tf.reduce_mean(ce,axis=1)

    return  loss_label

def top_k_accuracy(predict_label, labels, k=1):

    batch_size = labels.get_shape().as_list()[0]
    n_steps = labels.get_shape().as_list()[1]
    acc_all = 0
    for i in range(n_steps):
        right_1 = tf.to_float(tf.nn.in_top_k(predictions=predict_label[:, i, :], targets=labels[:, i], k=k))
        # ic(predict_label)
        # ic(labels)
        num_correct_per_batch = tf.reduce_sum(right_1)
        acc = num_correct_per_batch / batch_size
        acc_all = acc_all + acc


    return acc_all/n_steps

