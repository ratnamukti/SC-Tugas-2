
import cv2
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import pickle
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel



kumpulan_file_gambar = [f for f in listdir("dataset") if isfile(join("dataset", f))]
# print(kumpulan_file_gambar)

dataset = {"nama":[], "image":[], "resize_image" :[]}

for nama_file_gambar in kumpulan_file_gambar:
    image_pixel = cv2.imread("dataset/" + nama_file_gambar)
    dataset['nama'].append(nama_file_gambar)
    dataset['image'].append(image_pixel)
    dataset_resize_150x150_brg = cv2.resize(image_pixel, (150, 150), interpolation=cv2.INTER_CUBIC) #Untuk mengubah ukuran gmbr

    dataset_resize = dataset_resize_150x150_brg.flatten() # Untuk yang resize di flatten jadi 1 dimensi
    dataset['resize_image'].append(dataset_resize)

   # print(len(dataset['image']))

dataset['resize_image'] = np.array(dataset['resize_image']) 
dataset['resize_image'].shape

# # mean red
# gambar_gajah = dataset['image'][0]
# gambar_gajah_merah = gambar_gajah[:,:,2]
# # print(type(gambar_gajah_merah))
# # print(gambar_gajah_merah.shape)
# mean_red = gambar_gajah_merah.mean()
# print(mean_red)

# # mean green
# gambar_gajah = dataset['image'][0]
# gambar_gajah_hijau = gambar_gajah[:,:,1]
# mean_green = gambar_gajah_hijau.mean()
# print(mean_green)

# # mean blue
# gambar_gajah = dataset['image'][0]
# gambar_gajah_biru = gambar_gajah[:,:,0]
# mean_blue = gambar_gajah_biru.mean()
# print(mean_blue)

# # mean grayscale
# gambar_gajah = dataset['image'][0]
# gambar_gray = cv2.cvtColor(gambar_gajah, cv2.COLOR_BGR2GRAY)
# gambar_gray.shape
# mean_gambar_gray = gambar_gray.mean()
# print(mean_gambar_gray)

# # Energy, Contrast, Homogeneity
# g = greycomatrix(gambar_gray, [1], [0], normed=True)
# homogenity_gajah = greycoprops(g, 'homogeneity')[0][0]
# energy_gajah = greycoprops(g, 'energy')[0][0]
# contrast_gajah = greycoprops(g, 'contrast')[0][0]
# print(homogenity_gajah)
# print(energy_gajah)
# print(contrast_gajah)

# # Fitur Extraction
# fitur = [mean_red, mean_green, mean_blue, mean_gambar_gray, homogenity_gajah, energy_gajah, contrast_gajah]
# type(fitur)
# fitur = np.array(fitur)
# print(fitur)

class FeatureExtraction(TransformerMixin):        
    def fit(self):
        return self
    
    def fit_transform(self, X, y=None):
        list_list_fitur = []
        for image in X:
            gambar_merah = image[:,:,2]
            gambar_hijau = image[:,:,1]
            gambar_biru = image[:,:,0]
            gambar_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            g = greycomatrix(gambar_gray, [1], [0], normed=True)
            
            mean_merah = gambar_merah.mean()
            std_merah = gambar_merah.std()

            mean_hijau = gambar_hijau.mean()
            std_hijau = gambar_hijau.std()

            mean_biru = gambar_biru.mean()
            std_biru = gambar_biru.std()

            mean_gray = gambar_gray.mean()
            std_gray = gambar_gray.std()
            mag_gray = (np.sum(gambar_gray))/mean_gray

            homogenity_gajah = greycoprops(g, 'homogeneity')[0][0]
            energy_gajah = greycoprops(g, 'energy')[0][0]
            contrast_gajah = greycoprops(g, 'contrast')[0][0]
            dissimilarity_gajah = greycoprops(g, 'dissimilarity')[0][0]
            correlation_gajah = greycoprops(g, 'correlation')[0][0]	

            rg_corr = np.corrcoef(gambar_merah, gambar_hijau)[0][1]
            rb_corr = np.corrcoef(gambar_merah, gambar_biru)[0][1]
            gb_corr = np.corrcoef(gambar_hijau, gambar_biru)[0][1]

            entropy = metrics.cluster.entropy(g)

            list_fitur = [mean_merah, std_merah ,mean_hijau, std_hijau ,mean_biru, std_biru ,mean_gray, std_gray,
                         mag_gray, homogenity_gajah, energy_gajah, contrast_gajah, rg_corr, rb_corr, gb_corr, entropy, dissimilarity_gajah, correlation_gajah]
            
            list_list_fitur.append(list_fitur)
        
        return np.array(list_list_fitur)
    
    def transform(self, img, y=None):
        return self.fit_transform(img)

fiturEks = FeatureExtraction()
hasil_fitur_ekstraksi = fiturEks.fit_transform(dataset['image'])
type(hasil_fitur_ekstraksi)
# print(hasil_fitur_ekstraksi.shape)

# MEMBUAT DATAFRAME
df = pd.DataFrame(hasil_fitur_ekstraksi, columns=['mean_merah', 'std_merah' ,'mean_hijau', 'std_hijau' ,'mean_biru', 'std_biru' ,'mean_gray', 'std_gray', 'mag_gray', 'homogenity_gajah', 'energy_gajah', 'contrast_gajah', 'rg_corr', 'rb_corr', 'gb_corr', 'entropy', 'dissimilarity_gajah', 'correlation_gajah'])
df['nama'] = dataset['nama']
#print(df)

def apakah_macan_panda_gajah(x):
    if "tiger" in x: #klo gajah ada di string x
        return 2
    if "gajah" in x:
        return 1
    if "panda" in x: 
        return 0

df['label'] = df['nama'].apply(lambda x: apakah_macan_panda_gajah(x))
# df.to_csv('frame1.csv',index=False)

# FEATURE SELECTION
X = df.iloc[:,0:7].get_values()
y = df['label'].get_values()
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
clf.feature_importances_  

model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new.shape               
# print(X_new.shape)

# MENGGUNAKAN ALGORITTMA ML DALAM PEMBUATAN MODEL
clf = DecisionTreeClassifier(random_state=666)
clf.fit(X_new, y)
clf.score(X_new, y)

tree.export_graphviz(clf,out_file="tree.dot")

# VALIDASI DGN MELAKUKAN 10-FOLD CROSS VALIDATION
predicted = cross_val_predict(clf, X_new, y, cv=10)
# print(metrics.accuracy_score(y, predicted))
clf.fit(X_new, y)

# MEMBUAT MODELNYA
pipe_dt = Pipeline([('bebas nama ini',FeatureExtraction()), ('ftr_slc', SelectFromModel(ExtraTreesClassifier())), ('dt_id3', DecisionTreeClassifier(criterion='entropy',random_state=666))])
pipe_dt.fit(dataset['image'],y)
predicted = cross_val_predict(pipe_dt, dataset['image'], y, cv=10)
akurasicv = metrics.accuracy_score(y,predicted)

# FITUR PIXEL GAMBAR
clf2 = DecisionTreeClassifier(criterion='entropy', random_state=666)
X_pixel = dataset['resize_image']
fitness = clf2.fit(X_pixel, y)
skor =clf2.score(X_pixel, y)

# print(akurasicv)
# print(fitness)
# print(skor)

predicted = cross_val_predict(clf2, X_pixel, y, cv=10)
# print(metrics.accuracy_score(y, predicted))


# SIMPAN MODEL DGN PICKLE
pickle.dump(pipe_dt, open("DecisionTreeClasifier22.p","wb"))
a = 5
pickle.dump(a, open("a.p","wb"))
initadisayadump = pickle.load( open("a.p","rb"))
print(initadisayadump)

deciTree = pickle.load( open('DecisionTreeClasifier22.p', 'rb' ))


