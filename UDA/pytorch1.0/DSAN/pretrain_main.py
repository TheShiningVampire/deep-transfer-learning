import torch
from torchvision import datasets, transforms
from core import eval_src, eval_tgt, train_src, train_tgt
from utils import get_data_loader, init_model, init_random_seed, load_kimiaNet

from data_loader_pretrain import load_pretrain_data

if __name__ == '__main__':
    # init random seed
    init_random_seed(2021)

    # load dataset
    src_data_loader = load_pretrain_data("./dataset/Bach_Breakhis_pretrain/Bach/", "train", 4, train = True)
    src_data_loader_eval = load_pretrain_data("./dataset/Bach_Breakhis_pretrain/Bach/", "val", 4, train = False)

    tgt_data_loader = load_pretrain_data("./dataset/Bach_Breakhis_pretrain/Breakhis/", "train", 4, train = True)

    tgt_data_loader_eval = load_pretrain_data("./dataset/Bach_Breakhis_pretrain/Breakhis/", "val", 4, train = False)

    # load models
    # loading KimiaNet
    print("=== Loading KimiaNet Encoder and Classifier ===")
    src_encoder, src_classifier = load_kimiaNet(
        pt_model_path='./KimiaNet_Weights/weights/KimiaNetPyTorchWeights.pth',
        input_size=[512, 384],
        num_classes=2,
        adapt=False)
    
    """ 
    path_src_encoder = 'snapshots/KimiaNet-ADDA-source-encoder-5-30.pt'
    path_src_classifier = 'snapshots/KimiaNet-ADDA-source-classifier-5-30.pt'
    src_encoder.load_state_dict(torch.load(path_src_encoder))
    src_classifier.load_state_dict(torch.load(path_src_classifier))
    """

    
    
    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    # print(src_encoder)
    print(">>> Source Classifier <<<")
    # print(src_classifier)

    src_encoder, src_classifier, avg_train_losses, avg_valid_losses, train_accuracies, valid_accuracies = train_src(src_encoder, src_classifier, src_data_loader, src_data_loader_eval)

    # saving tensors for plotting
    # Pretrain: 512 dim, with ES
    # Pretrain-2: 512 dim, w/o ES
    torch.save(avg_train_losses, 'saved_tensors/Pretrain-WO-KIMIA-tr-loss.pt')
    torch.save(avg_valid_losses, 'saved_tensors/Pretrain-WO-KIMIA-val-loss.pt')
    torch.save(train_accuracies, 'saved_tensors/Pretrain-WO-KIMIA-tr-acc.pt')
    torch.save(valid_accuracies, 'saved_tensors/Pretrain-WO-KIMIA-val-acc.pt')

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    _, Acc = eval_src(src_encoder, src_classifier, src_data_loader_eval)
    print("Final source encoder Acc: {:2%}".format(Acc))
    

    # # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    # print(">>> source only <<<")
    _, Acc2 = eval_src(src_encoder, src_classifier, tgt_data_loader_eval)
    print("Final Target dataset Acc: {:2%}".format(Acc2))
