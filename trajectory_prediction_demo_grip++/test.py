import torch
import numpy as np 

from utils.data_process import preprocess_data,data_loader,my_load_model
from model.model import Model


max_x = 1. 
max_y = 1. 
history_frames = 6 # 3 second * 2 frame/second

future_frames = 6 # 3 second * 2 frame/second
batch_size_test = 1

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(pra_model, pra_data_loader,test_result_file):
	# pra_model.to(dev)
	pra_model.eval()
	rescale_xy = torch.ones((1,2,1,1)).to(dev)
	rescale_xy[:,0] = max_x
	rescale_xy[:,1] = max_y
	all_overall_sum_list = []
	all_overall_num_list = []
	with open(test_result_file, 'w') as writer:
		# train model using training data
		for iteration, (ori_data, A, mean_xy) in enumerate(pra_data_loader):
			# data: (N, C, T, V)
			# C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
			data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)
			input_data = data[:,:,:history_frames,:] # (N, C, T, V)=(N, 4, 6, 120)
			output_mask = data[:,-1,-1,:] # (N, V)=(N, 120)
			# print(data.shape, A.shape, mean_xy.shape, input_data.shape)

			ori_output_last_loc = no_norm_loc_data[:,:2,history_frames-1:history_frames,:]
		
			A = A.float().to(dev)
			predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=future_frames, pra_teacher_forcing_ratio=0, pra_teacher_location=None) # (N, C, T, V)=(N, 2, 6, 120)
			predicted = predicted *rescale_xy 

			for ind in range(1, predicted.shape[-2]):
				predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2)
			predicted += ori_output_last_loc

			now_pred = predicted.detach().cpu().numpy() # (N, C, T, V)=(N, 2, 6, 120)
			now_mean_xy = mean_xy.detach().cpu().numpy() # (N, 2)
			now_ori_data = ori_data.detach().cpu().numpy() # (N, C, T, V)=(N, 11, 6, 120)
			now_mask = now_ori_data[:, -1, -1, :] # (N, V)
			
			now_pred = np.transpose(now_pred, (0, 2, 3, 1)) # (N, T, V, 2)
			now_ori_data = np.transpose(now_ori_data, (0, 2, 3, 1)) # (N, T, V, 11)
			
			# print(now_pred.shape, now_mean_xy.shape, now_ori_data.shape, now_mask.shape)

			for n_pred, n_mean_xy, n_data, n_mask in zip(now_pred, now_mean_xy, now_ori_data, now_mask):
				# (6, 120, 2), (2,), (6, 120, 11), (120, )
				num_object = np.sum(n_mask).astype(int)
				# only use the last time of original data for ids (frame_id, object_id, object_type)
				# (6, 120, 11) -> (num_object, 3)
				n_dat = n_data[-1, :num_object, :3].astype(int)
				for time_ind, n_pre in enumerate(n_pred[:, :num_object], start=1):
					# (120, 2) -> (n, 2)
					# print(n_dat.shape, n_pre.shape)
					for info, pred in zip(n_dat, n_pre+n_mean_xy):
						information = info.copy()
						information[0] = information[0] + time_ind
						result = ' '.join(information.astype(str)) + ' ' + ' '.join(pred.astype(str)) + '\n'
						# print(result)
						writer.write(result)

def run_test(pretrained_model_path, pra_data_path,test_result_file):

	graph_args={'max_hop':2, 'num_node':120}
	model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True)
	model.to(dev)

	pra_model = my_load_model(model, pretrained_model_path)
	loader_test = data_loader(pra_data_path, pra_batch_size=batch_size_test, pra_shuffle=False, pra_drop_last=False, train_val_test='test')

	test_model(pra_model, loader_test,test_result_file)
	

if __name__ == '__main__':

	pretrained_model_path = 'result/trained_models/model_epoch_0049.pt'
	pra_data_path = 'result/processed_files/test_data.pkl'
	test_result_file = 'result/test_result/prediction_result.txt'

	run_test(pretrained_model_path, pra_data_path,test_result_file)