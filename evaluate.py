import torch
import numpy as np 

from utils.data_process import preprocess_data, compute_RMSE, data_loader,my_load_model
from model.model import Model


max_x = 1. 
max_y = 1. 
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size_val = 32


def display_result2(pra_results, pra_pref='vheicle'):
	all_overall_sum_list, all_overall_num_list = pra_results
	overall_sum_time = np.sum(all_overall_sum_list**0.5, axis=0)
	overall_num_time = np.sum(all_overall_num_list, axis=0)
	overall_loss_time = (overall_sum_time / overall_num_time) 

	ADE = np.mean(overall_loss_time)
	FDE = overall_loss_time[-1] 
	print('-'*10,pra_pref,'-'*10)
	print(f'ADE={ADE:.3f} m')
	print(f'FDE={FDE:.3f} m')
	print()

	return [ADE, FDE]


def evaluate_calulate(pra_model, pra_data_loader):
	# pra_model.to(dev)
	pra_model.eval()
	rescale_xy = torch.ones((1,2,1,1)).to(dev)
	rescale_xy[:,0] = max_x
	rescale_xy[:,1] = max_y
	all_overall_sum_list = []
	all_overall_num_list = []

	all_car_sum_list = []
	all_car_num_list = []
	all_human_sum_list = []
	all_human_num_list = []
	all_bike_sum_list = []
	all_bike_num_list = []
	# train model using training data
	for iteration, (ori_data, A, _) in enumerate(pra_data_loader):
		# data: (N, C, T, V)
		# C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
		data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)

		for now_history_frames in range(6, 7):
			input_data = data[:,:,:now_history_frames,:] # (N, C, T, V)=(N, 4, 6, 120)
			output_loc_GT = data[:,:2,now_history_frames:,:] # (N, C, T, V)=(N, 2, 6, 120)
			output_mask = data[:,-1:,now_history_frames:,:] # (N, C, T, V)=(N, 1, 6, 120)

			ori_output_loc_GT = no_norm_loc_data[:,:2,now_history_frames:,:]
			ori_output_last_loc = no_norm_loc_data[:,:2,now_history_frames-1:now_history_frames,:]

			# for category
			cat_mask = ori_data[:,2:3, now_history_frames:, :] # (N, C, T, V)=(N, 1, 6, 120)
			
			A = A.float().to(dev)
			predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2], pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT) # (N, C, T, V)=(N, 2, 6, 120)

			predicted = predicted*rescale_xy
			# output_loc_GT = output_loc_GT*rescale_xy

			for ind in range(1, predicted.shape[-2]):
				predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2)
			predicted += ori_output_last_loc

			### overall dist
			# overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, output_loc_GT, output_mask)		
			overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, ori_output_loc_GT, output_mask)		
			# all_overall_sum_list.extend(overall_sum_time.detach().cpu().numpy())
			all_overall_num_list.extend(overall_num.detach().cpu().numpy())
			# x2y2 (N, 6, 39)
			now_x2y2 = x2y2.detach().cpu().numpy()
			now_x2y2 = now_x2y2.sum(axis=-1)
			all_overall_sum_list.extend(now_x2y2)

			### car dist
			car_mask = (((cat_mask==1)+(cat_mask==2))>0).float().to(dev)
			car_mask = output_mask * car_mask
			car_sum_time, car_num, car_x2y2 = compute_RMSE(predicted, ori_output_loc_GT, car_mask)		
			all_car_num_list.extend(car_num.detach().cpu().numpy())
			# x2y2 (N, 6, 39)
			car_x2y2 = car_x2y2.detach().cpu().numpy()
			car_x2y2 = car_x2y2.sum(axis=-1)
			all_car_sum_list.extend(car_x2y2)

			### human dist
			human_mask = (cat_mask==3).float().to(dev)
			human_mask = output_mask * human_mask
			human_sum_time, human_num, human_x2y2 = compute_RMSE(predicted, ori_output_loc_GT, human_mask)		
			all_human_num_list.extend(human_num.detach().cpu().numpy())
			# x2y2 (N, 6, 39)
			human_x2y2 = human_x2y2.detach().cpu().numpy()
			human_x2y2 = human_x2y2.sum(axis=-1)
			all_human_sum_list.extend(human_x2y2)

			### bike dist
			bike_mask = (cat_mask==4).float().to(dev)
			bike_mask = output_mask * bike_mask
			bike_sum_time, bike_num, bike_x2y2 = compute_RMSE(predicted, ori_output_loc_GT, bike_mask)		
			all_bike_num_list.extend(bike_num.detach().cpu().numpy())
			# x2y2 (N, 6, 39)
			bike_x2y2 = bike_x2y2.detach().cpu().numpy()
			bike_x2y2 = bike_x2y2.sum(axis=-1)
			all_bike_sum_list.extend(bike_x2y2)

	all_objects_sum_list = []
	for single_list in [all_car_sum_list, all_human_sum_list, all_bike_sum_list]:
		all_objects_sum_list.extend(single_list)
	all_objects_sum_list = np.array(all_objects_sum_list)


	all_objects_num_list = []
	for single_list in [all_car_num_list, all_human_num_list, all_bike_num_list]:
		all_objects_num_list.extend(single_list)
	all_objects_num_list = np.array(all_objects_num_list)

	
	all_objects_sum_time = np.sum(all_objects_sum_list**0.5, axis=0)
	all_objects_num_time = np.sum(all_objects_num_list, axis=0)
	overall_err_time = (all_objects_sum_time / all_objects_num_time) 

	# 常用指标
	print('\n','='*20,'metrics_A','='*20)
	print(f'ADE={np.mean(overall_err_time):.3f} m')
	print(f'FDE={overall_err_time[-1]:.3f} m')

	#-----------------------------
	
	# ApolloScape数据集指标
	print('\n','='*20,'metrics_B','='*20)

	result_car = display_result2([np.array(all_car_sum_list), np.array(all_car_num_list)], pra_pref='car')
	result_human = display_result2([np.array(all_human_sum_list), np.array(all_human_num_list)], pra_pref='human')
	result_bike = display_result2([np.array(all_bike_sum_list), np.array(all_bike_num_list)], pra_pref='bike')



	result=[0.0,0.0]
	result[0] = 0.20*result_car[0] + 0.58*result_human[0] + 0.22*result_bike[0]
	result[1] = 0.20*result_car[1] + 0.58*result_human[1] + 0.22*result_bike[1]

	print('-'*10,'WS','-'*10)
	print(f'WSADE={result[0]:.3f} m')
	print(f'WSFDE={result[1]:.3f} m')
	print()

 



def evaluate_model(pretrained_model_path,pra_traindata_path):
	graph_args={'max_hop':2, 'num_node':120}
	model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True)
	model.to(dev)

	pra_model = my_load_model(model, pretrained_model_path)
	loader_val = data_loader(pra_traindata_path, pra_batch_size=batch_size_val, pra_shuffle=False, pra_drop_last=False, train_val_test='val') 

	evaluate_calulate(pra_model, loader_val)


if __name__ == '__main__': 
        	
    pretrained_model_path = 'result/trained_models/model_epoch_0049.pt'
    pra_traindata_path = 'result/processed_files/train_val_data.pkl'
    evaluate_model(pretrained_model_path,pra_traindata_path)
