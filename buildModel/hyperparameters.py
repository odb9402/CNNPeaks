#Model hyperparameters
class_threshold = 0.5

init_depth = 1
pnRate_threshold = 50
batch_size = 1 
evaluation_size = 1
generations = 4000
eval_every = 20
learning_rate = 0.00003
target_size = 12000
topK_set_a = 6000
topK_set_b = 3000
topK_set_c = 1500
topK_set_d = 750

filter_size_a = 101
filter_size_b = 301

conv1_ref_features = 4

max_pool_size_ref1 = 2
max_pool_size_ref2 = 2
max_pool_size_ref3 = 2

conv1_features = 16
conv2_features = 32 

conv1a_features = 16
conv1b_features = 16
conv1c_features = 16
convMax1_features = 16
convAvg1_features = 16

conv2a_features = 32
conv2b_features = 32
conv2c_features = 32
convMax2_features = 32
convAvg2_features = 32

conv3a_features = 48
conv3b_features = 48
conv3c_features = 48
convMax3_features = 64
convAvg3_features = 64

conv4a_features = 128
conv4b_features = 128
conv4c_features = 128
convMax4_features = 32
convAvg4_features = 32

conv5a_features = 192
conv5b_features = 192
conv5c_features = 192
convMax5_features = 128
convAvg5_features = 128

conv6a_features = 512
conv6b_features = 512
conv6c_features = 512
conv6d_features = 512

conv7a_features = 720
conv7b_features = 720
conv7c_features = 720
conv7d_features = 720

max_pool_size_stem = 2
max_pool_size1 = 2
max_pool_size2 = 2
max_pool_size3 = 2
max_pool_size4 = 3
max_pool_size5 = 2
max_pool_size6 = 2

fully_connected_size1 = 500 
fully_connected_size2 = 250
threshold_division = 10
#10 for beta 
#50 for alpha