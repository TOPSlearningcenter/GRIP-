
import matplotlib.pyplot as plt
import os 
import numpy as np 
import torch


from model.model import Model
from utils.data_process import preprocess_data, data_loader,my_load_model

max_x = 1. 
max_y = 1. 
history_frames = 6 # 3 second * 2 frame/second
future_frames = 6 # 3 second * 2 frame/second

batch_size_train = 64 
batch_size_val = 32
batch_size_test = 1
total_epoch = 50
base_lr = 0.01
lr_decay_epoch = 5
# dev = 'cuda:0' 
dev = torch.device("cpu")
work_dir = './trained_models'
log_file = os.path.join(work_dir,'log_test.txt')
test_result_file = 'prediction_result.txt'



def vis_pred(model, data_loader, idx,visualizepred_path):
	# 获取设备
	dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将模型设置为评估模式
	model.eval()
	rescale_xy = torch.ones((1, 2, 1, 1)).to(dev)
	rescale_xy[:, 0] = max_x
	rescale_xy[:, 1] = max_y

    
    # 初始化figure
	# fig, ax = plt.subplots(figsize=(10, 5))
	plt.figure(figsize=(10, 5))
    
    # 遍历数据加载器直到找到指定索引的数据
	for batch_idx, (ori_data, A, mean_xy) in enumerate(data_loader):
        # 使用相同的预处理步骤
		data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)

        # 预处理
		input_data = data[:,:,:history_frames,:]  # (N, C, T, V)=(N, 4, 6, 120)
		ori_output_last_loc = no_norm_loc_data[:,:2,history_frames-1:history_frames,:]

        # 将A移动到设备
		A = A.float().to(dev)

        # 如果找到了指定索引的数据
		if batch_idx == idx:
            # 使用模型进行预测
			with torch.no_grad():
				predictions = model(
                    pra_x=input_data,
                    pra_A=A,
                    pra_pred_length=future_frames,
                    pra_teacher_forcing_ratio=0,
                    pra_teacher_location=None
                )
            
			predictions = predictions *rescale_xy 

			for ind in range(1, predictions.shape[-2]):
				predictions[:,:,ind] = torch.sum(predictions[:,:,ind-1:ind+1], dim=-2)
			predictions += ori_output_last_loc


			now_pred = predictions.detach().cpu().numpy() # (N, C, T, V)=(N, 2, 12, 120)
			now_ori_data = ori_data.detach().cpu().numpy() # (N, C, T, V)=(N, 11, 12, 120)
			now_mask = now_ori_data[:, -1, -1, :] # (N, V)
			
			now_pred = np.transpose(now_pred, (0, 3, 1, 2)) # (N, V, 2, T)
			now_ori_data = np.transpose(now_ori_data, (0, 3, 1, 2)) # (N, V, 11, T)
			
			for n_pred, n_data, n_mask in zip(now_pred, now_ori_data, now_mask):
				# (120, 2,12),  (120, 10,12), (120, )
				num_object = np.sum(n_mask).astype(int)

				# (120, 10,12) -> (num_object, 3,12)
				post_traj = n_data[:num_object,3:5,:].astype(float)

				ob_ind = 1

				for n_pre,pos in zip(n_pred[:num_object,:], post_traj):

					pred_x = np.insert(n_pre[0], 0, pos[0][history_frames-1])
					pred_y = np.insert(n_pre[1], 0, pos[1][history_frames-1])

					plt.plot(pred_x,pred_y, label='pred_traj'+str(ob_ind),marker='*',linestyle='--')
					plt.plot(pos[0][history_frames-1:],pos[1][history_frames-1:],label='gt_traj'+str(ob_ind),marker='+',linestyle='--')
					plt.plot(pos[0][:history_frames],pos[1][:history_frames],label='post_traj'+str(ob_ind),marker='o')
					# break

					ob_ind += 1

					if ob_ind > 5:
						break
				
				break
			
			plt.xlabel('x')
			plt.ylabel('y')
			plt.legend()
			plt.title('Trajectory Visualization')
			
			fileName = os.path.join(visualizepred_path,str(idx)+'.jpg')
			plt.savefig(fileName, dpi=300)  # 保存为图片
			
			plt.show()
			

	return



def visulization(pretrained_model_path,pra_traindata_path,indx,visualize_path):

	graph_args={'max_hop':2, 'num_node':120}
	model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True)
	model.to(dev)

	loader_val = data_loader(pra_traindata_path, pra_batch_size=batch_size_val, pra_shuffle=False, pra_drop_last=False, train_val_test='val') 
	pra_model = my_load_model(model, pretrained_model_path)

	vis_pred(pra_model, loader_val,indx,visualize_path)



if __name__ == '__main__': 
	
    pretrained_model_path = 'result/trained_models/model_epoch_0049.pt'
    pra_traindata_path = 'result/processed_files/train_val_data.pkl'

    visulization(pretrained_model_path,pra_traindata_path,indx=30,visualize_path='result/visulize_val')