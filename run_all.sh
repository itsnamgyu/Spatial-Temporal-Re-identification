echo "PREPARE DUKE"
python3 prepare.py --Duke --Path "/home/itsnamgyu/data/dukemtmc-reid/DukeMTMC-reID"
echo "PREPARE MARKET"
python3 prepare.py --Market --Path "/home/itsnamgyu/data/market1501"

echo "TRAIN MARKET"
python3 train_market.py --PCB --gpu_ids 0 --name ft_ResNet50_pcb_market_e --erasing_p 0.5 --train_all --data_dir "./dataset/market_rename/"
echo "TRAIN DUKE"
python3 train_duke.py --PCB --gpu_ids 0 --name ft_ResNet50_pcb_duke_e --erasing_p 0.5 --train_all --data_dir "./dataset/DukeMTMC_prepare/"

echo "TEST MARKET"
python3 test_st_market.py --PCB --gpu_ids 2 --name ft_ResNet50_pcb_market_e --test_dir "./dataset/market_rename/"
echo "TEST DUKE"
python3 test_st_duke.py --PCB --gpu_ids 2 --name ft_ResNet50_pcb_duke_e --test_dir "./dataset/DukeMTMC_prepare/"

echo "GEN ST AND EVAL MARKET"
python3 gen_st_model_market.py --name ft_ResNet50_pcb_market_e --data_dir "./dataset/market_rename/"
python3 evaluate_st.py --name ft_ResNet50_pcb_market_e

echo "GEN ST AND EVAL DUKE"
python3 gen_st_model_duke.py --name ft_ResNet50_pcb_duke_e  --data_dir  "./dataset/DukeMTMC_prepare/"
python3 evaluate_st.py --name ft_ResNet50_pcb_duke_e

echo "RERANK AND EVAL MARKET"
python3 gen_rerank_all_scores_mat.py --name ft_ResNet50_pcb_market_e
python3 evaluate_rerank_market.py --name ft_ResNet50_pcb_market_e

echo "RERANK AND EVAL DUKE"
python3 gen_rerank_all_scores_mat.py --name ft_ResNet50_pcb_duke_e
python3 evaluate_rerank_duke.py --name ft_ResNet50_pcb_duke_e
