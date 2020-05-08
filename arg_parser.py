import argparse
'''
example of a command argument assuming that current directory is my_utility/python_script
    python __init__.py --merge
'''

'''
    with default
        parser.add_argument('--dataset', type=str, default='gene_disease', help='specify type of dataset to be used')

    with action
        parser.add_argument('--subgraph', action="store_true", help='NOT CURRENTLY COMPATIBLE WITH THE PROGRAM;Use only node in the largest connected component instead of all nodes disconnected graphs')

    with nargs, this is used to extract provided arguments as a list 
        eg --something 1 2 3 4 5 
            args.something == [1,2, 3,4,5] is true
        parser.add_argument('--weighted_class', default=[1,1,1,1,1], nargs='+', help='list of weighted_class of diseases only in order <0,1,2,3,4,5>')
'''
parser = argparse.ArgumentParser()
parser.add_argument('--log', action='store_true', help="")

#--------general
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
#--------setup
parser.add_argument('--dataset', type=str, default=None, help='')
parser.add_argument('--k_fold_split', '-kf',type=str, default=3, help='')
parser.add_argument('--downsample','-ds', action='store_true', help="")

#--------model name
parser.add_argument('--run_gcn_gan', action='store_true', help="")
parser.add_argument('--run_gcn_only', action='store_true', help="")

#--------hyper_parameters
parser.add_argument('--main_epoch', '-me',type=int, default=None, help='')
parser.add_argument('--num_gan_epoch', '-ng',type=int, default=None, help='')
parser.add_argument('--preserved_edges_percent','-pep', type=float, default=None, help='')


#--------plotting
parser.add_argument('--plot_roc', '-pr', action='store_true', help="")
parser.add_argument('--plot_cv_roc', '-pcr', action='store_true', help="")
parser.add_argument('--plot_emb', '-pe',action='store_true', help="")

#--------save
parser.add_argument('--save_emb', '-se', action='store_true', help="")
parser.add_argument('--save_file', '-sf', action='store_true', help="")
parser.add_argument('--save_plot', '-sp', action='store_true', help="")
parser.add_argument('--save_cv_file', '-scf', action='store_true', help="")
parser.add_argument('--save_cv_plot', '-scp', action='store_true', help="")
parser.add_argument('--plot_and_save_all_cv', '-pasac', action='store_true', help="")

#--------manual or not
parser.add_argument('--manual', action='store_true', help="")
#-- utilities


args = parser.parse_args()
#--------check for args conflict

if args.plot_and_save_all_cv:
    # args.plot_cv_roc = True
    args.save_cv_file = True
    args.save_cv_plot = True
    args.save_emb = True

if args.manual:
    assert not args.run_gcn_only, ''
    assert not args.run_gcn_gan, ''
    assert args.num_gan_epoch is None, ''
    assert args.main_epoch is None, ''
    assert args.preserved_edges_percent is None, ''
    assert args.dataset is None, ''
else:
    assert args.dataset in ['cora','citeseer'], ''
    assert sum([args.run_gcn_gan,
                args.run_gcn_only]) == 1, "only of of the following can be true at the same time \n" \
                                          "1. args.run_gcn_gan, 2. args.run_gcn_only"
    assert args.preserved_edges_percent <= 1, ''
    if args.run_gcn_only:
        assert args.num_gan_epoch is None, ""
        if args.main_epoch  is None:
            args.main_epoch = 100
        if args.preserved_edges_percent is None:
            args.preserved_edges_percent = 1


    if args.run_gcn_gan:
        assert args.num_gan_epoch is not None, ""
        if args.main_epoch  is None:
            args.main_epoch = 100
        if args.num_gan_epoch is None:
            args.num_gan_epoch = 1
        if args.preserved_edges_percent is None:
            args.preserved_edges_percent = 1
