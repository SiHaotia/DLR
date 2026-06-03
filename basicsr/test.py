import logging
import torch
from os import path as osp
import os
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)    
    torch.backends.cudnn.benchmark = True
    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    # logger.info(get_env_info())
    # logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
        print(model.net_g.encoder_level1.featemp.shape)
        # model.net_g.mask
        features = {'channel':model.net_g.encoder_level1.featemp}
        torch.save(features, "datasets/visual/mask/channel.pth")
        # ir_level_1 = torch.cat(model.net_g.ir_level_1,dim=0)
        # vi_level_1 = torch.cat(model.net_g.vi_level_1, dim=0)
        # ir_level_2 = torch.cat(model.net_g.ir_level_2, dim=0)
        # vi_level_2 = torch.cat(model.net_g.vi_level_2, dim=0)
        # ir_level_3 = torch.cat(model.net_g.ir_level_3, dim=0)
        # vi_level_3 = torch.cat(model.net_g.vi_level_3, dim=0)
        # print(ir_level_1.shape, ir_level_2.shape, ir_level_3.shape)
        # print(vi_level_1.shape, vi_level_2.shape, vi_level_3.shape)
        # features = {
        #     "ir_level_1": ir_level_1,
        #     "vi_level_1": vi_level_1,
        #     "ir_level_2": ir_level_2,
        #     "vi_level_2": vi_level_2,
        #     "ir_level_3": ir_level_3,
        #     "vi_level_3": vi_level_3,
        # }

        # 保存到文件，比如 features.pth
        # torch.save(features, "experiments/OB_feature/features.pth")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
