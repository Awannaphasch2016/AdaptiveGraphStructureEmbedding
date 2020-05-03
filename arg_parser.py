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
parser.add_argument('--run_gcn_only', action='store_true', help="")
parser.add_argument('--save_file', '-sf', action='store_true', help="")
parser.add_argument('--save_plot', '-sp', action='store_true', help="")
parser.add_argument('--save_cv_file', '-scf', action='store_true', help="")
parser.add_argument('--save_cv_plot', '-scp', action='store_true', help="")
parser.add_argument('--plot_roc', '-pr', action='store_true', help="")
parser.add_argument('--plot_cv_roc', '-pcr', action='store_true', help="")
parser.add_argument('--k_fold_split', type=str, default=3, help='')
parser.add_argument('--num_gan_epoch', type=str, default=3, help='')
#-- utilities


args = parser.parse_args()
