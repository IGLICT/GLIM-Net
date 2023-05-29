import numpy as np
from PIL import Image
import glob
import os


class DataLoader_atten_polar(object):

    def __init__(self, batch_size, list_img_path,state='train'):
        # reading data list

        self.state=state
        
        self.path_to_image = 'data/'+self.state+'/image/all/'
        self.path_to_label = 'data/' + self.state + '/label/all/'
        self.path_to_atten = 'data/final_atten/'
        self.path_to_polar = 'data/'+self.state+'/image/all_polar/'
        self.batch_size = batch_size
        self.list_img_path = list_img_path
        self.size = len(self.list_img_path)
        self.num_batches = int(self.size / self.batch_size)
        self.cursor = 0
        self.batch_order=0

    def get_batch(self, shuffle = True):  # Returns

        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            self.batch_order = 0
            if shuffle:
                np.random.shuffle(self.list_img_path)
        img_batch = []
        year_batch = []
        pre_time = []
        Atten_map_batch = []
        Polar_map_batch = []
        label_batch=[]
        for idx in range(self.batch_size):
            img = []
            year = []
            next_time = []
            label=[]
            Atten_map = []
            Polar_map = []
            count = 0
            image_subpath=self.list_img_path[self.batch_order * self.batch_size + idx]
            with open(self.path_to_label+image_subpath+'.txt', 'r') as f:
                K = f.readlines()
                for i_line in range(5):
                    line= K[i_line+1]
                    line = line.strip('\n')
                    line = int(line)
                    label.append(line)
            image_sublist = glob.glob(self.path_to_image + image_subpath + '/' + '*.JPG')
            image_sublist.sort()
            if len(image_sublist) < 5:
                image_sublist = glob.glob(self.path_to_image + image_subpath + '/' + '*.jpg')
                image_sublist.sort()


            for idx_image in range(5):
                image=image_sublist[idx_image]
                img_name = os.path.split(image)[-1]

                polar_path = self.path_to_polar + image.split('/')[4] + '/' + img_name[:-4] + image[-4:]
                polar = Image.open(polar_path)
                polar = polar.resize((224, 224))
                polar = np.asarray(polar, np.uint8)
                polar = polar / 255.0

                image = Image.open(image)
                image = image.resize((224, 224))
                image = np.asarray(image, np.uint8)
                image = image / 255.0


                Polar_map.append(polar)
                img.append(image)

                if idx_image ==0:
                    year.append(int(0))
                else:
                    year0 = int(image_sublist[idx_image-1].split('/')[-1].split('_')[1])
                    month0 = int(image_sublist[idx_image-1].split('/')[-1].split('_')[2])
                    year1 = int(image_sublist[idx_image].split('/')[-1].split('_')[1])
                    month1 = int(image_sublist[idx_image].split('/')[-1].split('_')[2])
                    gap = 12 * (year1 - year0) + (month1 - month0)
                    year.append(int(gap+year[idx_image-1]))
                year0 = int(image_sublist[idx_image].split('/')[-1].split('_')[1])
                month0 = int(image_sublist[idx_image].split('/')[-1].split('_')[2])
                year1 = int(image_sublist[idx_image+1].split('/')[-1].split('_')[1])
                month1 = int(image_sublist[idx_image+1].split('/')[-1].split('_')[2])
                gap = 12 * (year1 - year0) + (month1 - month0)
                next_time.append(int(gap))

                count = count + 1
            assert count == 5
            # pre_year
            pre_time.append(np.array(next_time))
            img_batch.append(np.array(img))
            year_batch.append(np.array(year))
            

            # Atten_map_batch.append(np.array(Atten_map))
            Polar_map_batch.append(np.array(Polar_map))
            label_batch.append(np.array(label))
            self.cursor += 1
        
        ### construct time matrix
        ### (4, 5) shape of year_batch
        time = np.concatenate(((year_batch[0])[np.newaxis,:], (year_batch[1])[np.newaxis,:], (year_batch[2])[np.newaxis,:], (year_batch[3])[np.newaxis,:]),axis=0)
        time_matrix = np.zeros((4,5,5))

        for batch in range(time_matrix.shape[0]):
            for i in range(time_matrix.shape[1]):
                for j in range(time_matrix.shape[2]):
                    if i>=j:

                        time_matrix[batch][i][j] = time[batch][i] - time[batch][j]
                    else:

                        time_matrix[batch][i][j] = time[batch][j] - time[batch][i]

        self.batch_order += 1

        return np.array(img_batch),np.array(year_batch),np.array(Atten_map_batch),np.array(Polar_map_batch),np.array(label_batch),np.array([96 if data>96 else data for data in time_matrix.flat]).reshape(4, 5, 5)/96.0

