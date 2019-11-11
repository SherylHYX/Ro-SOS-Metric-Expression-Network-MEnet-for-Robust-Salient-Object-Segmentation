import scipy.io
tsplit = scipy.io.loadmat('datasets/HKU_IS/testImgSet')
data = {'HKU_IS': ['datasets/HKU_IS/imgs/'+line[0][0] for line in tsplit['testImgSet']]}
