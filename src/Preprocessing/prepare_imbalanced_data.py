import numpy as np
import torch

class ModelInputData():
    def __init__(self, dir, dataset, is_downsampled=True):

        self.downsample = is_downsampled
        self.dir = dir
        if dataset == 'cora':
            self.data = self.preparing_cora_for_new_purposed_model()
            self.cora_prepare_ind_for_trainning_and_test_set()
        elif dataset == 'citeseer':
            self.data = self.preparing_citeseer_for_new_purposed_model()
            self.citeseer_prepare_ind_for_trainning_and_test_set()

    def relabel_minority_and_majority_classes(self,data):

        uniq_labels = np.unique(data.y, return_counts=True)
        minority_class = np.unique(data.y, return_counts=True)[1].argmin()
        new_y = np.array([0 if i == minority_class else 1 for i in data.y])
        return new_y

    def preparing_citeseer_for_new_purposed_model(self):
        from dgl.data.citation_graph import load_citeseer

        data = load_citeseer()
        data.y = torch.tensor(data.labels)
        data.x = torch.tensor(data.features).type(torch.long)
        # TODO how to get edge_index from citeseer
        data.edge_index = torch.tensor(list(data.graph.edges)).reshape(2,-1)

        data.y_before_relabel = data.y
        new_y = self.relabel_minority_and_majority_classes(data)
        data.y = torch.tensor(new_y)
        data.num_classes = torch.tensor(np.unique(data.y).shape[0])
        data.num_features = torch.tensor(data.x.shape[1])

        return data

    def preparing_cora_for_new_purposed_model(self):

        data, _ = torch.load(
            f'{self.dir}\\..\\Notebook\\Examples\\data\\Cora\\Cora\\processed\\data.pt')
        # data, _ = torch.load(
        #     r'C:\Users\Anak\PycharmProjects\AdaptiveGraphStructureEmbedding\Notebook\Examples\data\Cora\Cora\processed\data.pt')
        data.y_before_relabel = np.array(data.y)
        new_y = self.relabel_minority_and_majority_classes(data)
        data.y = torch.tensor(new_y).type(torch.long)
        data.num_classes = np.unique(data.y).shape[0]

        return data

    def citeseer_prepare_ind_for_trainning_and_test_set(self):
        labeling_percent = 0.036
        num_labeling = int(self.data.x.shape[0] * labeling_percent)

        # =====================
        # ==control for probability for each class
        # =====================
        self.label_count = np.unique(self.data.y_before_relabel,
                                     return_counts=True)
        self.label_count_dict = {i: 1 / self.data.y.shape[0] for (i, j) in
                                 zip(self.label_count[0], self.label_count[1])}
        self.p = np.array([self.label_count_dict[i] for i in
                           self.data.y_before_relabel.numpy()])
        self.p /= self.p.sum()
        selected_data_ind = np.random.choice(np.arange(self.data.x.shape[0]),
                                             size=num_labeling, replace=False,
                                             p=self.p)

        selected_data_ind_bool = np.zeros(self.data.x.shape[0])
        selected_data_ind_bool[selected_data_ind] = 1

        self.trainning_selected_ind = torch.tensor(selected_data_ind).type(torch.long)
        self.test_selected_ind = torch.tensor(
            np.where(selected_data_ind_bool == 0)[0]).type(torch.long)

        self.data.train_mask = None
        self.data.test_mask = None

        self.trainning_selected_min_ind = np.intersect1d(np.where(self.data.y == 0),
                                                         np.where(
                                                             selected_data_ind_bool== 1))
        self.trainning_selected_maj_ind = np.intersect1d(np.where(self.data.y == 1),
                                                         np.where(
                                                             selected_data_ind_bool == 1))
        self.test_selected_min_ind = np.intersect1d(np.where(self.data.y == 0),
                                                    np.where(
                                                        np.logical_not(selected_data_ind_bool)== 1))
        self.test_selected_maj_ind = np.intersect1d(np.where(self.data.y == 1),
                                                    np.where(
                                                        np.logical_not(selected_data_ind_bool)== 1))

        self.data.y = self.data.y.type(torch.long)


    def cora_prepare_ind_for_trainning_and_test_set(self):
        labeling_percent = 0.052
        num_labeling = int(self.data.x.shape[0] * labeling_percent)

        # =====================
        # ==control for probability for each class
        # =====================

        self.label_count = np.unique(self.data.y_before_relabel,
                                     return_counts=True)
        self.label_count_dict = {i: 1 / self.data.y.shape[0] for (i, j) in
                                 zip(self.label_count[0], self.label_count[1])}
        self.p = np.array([self.label_count_dict[i] for i in
                           self.data.y_before_relabel])
        self.p /= self.p.sum()
        selected_data_ind = np.random.choice(np.arange(self.data.x.shape[0]),
                                             size=num_labeling, replace=False,
                                             p=self.p)

        selected_data_ind_bool = np.zeros(self.data.x.shape[0])
        selected_data_ind_bool[selected_data_ind] = 1

        # TODO is trainning_selected_ind what I am lookgin for ?  imbalance train_test split
        self.trainning_selected_ind = torch.tensor(selected_data_ind).type(torch.long)
        self.test_selected_ind = torch.tensor(
            np.where(selected_data_ind_bool == 0)[0]).type(torch.long)


        # # TODO
        # self.data.train_mask = selected_data_ind_bool
        # self.data.test_mask = np.logical_not(selected_data_ind_bool)
        self.data.train_mask = None
        self.data.test_mask = None

        self.trainning_selected_min_ind = np.intersect1d(np.where(self.data.y == 0),
                                                    np.where(
                                                        selected_data_ind_bool== 1))
        self.trainning_selected_maj_ind = np.intersect1d(np.where(self.data.y == 1),
                                                    np.where(
                                                        selected_data_ind_bool == 1))
        self.test_selected_min_ind = np.intersect1d(np.where(self.data.y == 0),
                                               np.where(
                                                   np.logical_not(selected_data_ind_bool)== 1))
        self.test_selected_maj_ind = np.intersect1d(np.where(self.data.y == 1),
                                               np.where(
                                                   np.logical_not(selected_data_ind_bool)== 1))


    def set_data(self,fake_data=None):
    # def set_data(self, minreal_minfake_majreal_x, minreal_minfake_majreal_y,
    #              emb_after_conv1, num_select, fake_data=None):

        self.fake_data = fake_data
        # self.emb_after_conv1 = emb_after_conv1
        # self.minreal_minfake_majreal_x = minreal_minfake_majreal_x
        # self.minreal_minfake_majreal_y = minreal_minfake_majreal_y

    def set_train_test_data_index(self):

        self.prepare_gcn_dataset()
        # self.prepare_gcn_dataset(
        #     self.num_select, self.minreal_minfake_majreal_x,
        #     self.minreal_minfake_majreal_y, self.fake_data,
        #     self.trainning_selected_min_ind,
        #     self.trainning_selected_maj_ind,
        #     self.test_selected_min_ind,
        #     self.test_selected_maj_ind, fake_data = self.fake_data)
        # )

    def prepare_gcn_dataset(self):

        trainning_selected_min_ind = self.trainning_selected_min_ind
        trainning_selected_maj_ind = self.trainning_selected_maj_ind
        test_selected_min_ind = self.test_selected_min_ind
        test_selected_maj_ind = self.test_selected_maj_ind
        fake_data = self.fake_data

        if fake_data is not None:

            num_select = fake_data.shape[0]

            trainning_select_min_real_ind = torch.tensor(
                np.random.choice(trainning_selected_min_ind, size=num_select,
                                 replace=False)).type(torch.long)

            if self.downsample:
                # TODO change maj real to
                trainning_select_maj_real_ind = torch.tensor(
                    np.random.choice(trainning_selected_maj_ind,
                                     size=num_select * 2,
                                     replace=False)).type(torch.long)
            else:
                trainning_select_maj_real_ind = torch.tensor(trainning_selected_maj_ind).type(torch.long)


            test_select_maj_real_ind = torch.tensor(test_selected_maj_ind).type(
                torch.long)
            test_select_min_real_ind = torch.tensor(test_selected_min_ind).type(
                torch.long)
            select_min_fake_ind = torch.tensor(
                np.arange(fake_data.shape[0]) + self.data.y.shape[0]).type(
                torch.long)

            trainning_select_minfake_minreal_majreal_ind = torch.cat((
                trainning_select_min_real_ind,
                trainning_select_maj_real_ind,
                select_min_fake_ind),
                0)

            test_select_minfake_minreal_majreal_ind = torch.cat((
                test_select_min_real_ind,
               test_select_maj_real_ind,
                # test_select_min_real_ind,
                # test_select_maj_real_ind,
                select_min_fake_ind),
                0)

            trainning_select_minreal_majreal_ind = torch.cat((
                trainning_select_min_real_ind,
                trainning_select_maj_real_ind),
                0)


            test_select_minreal_majreal_ind = torch.cat((
                test_select_min_real_ind,
                test_select_maj_real_ind),
                0)
            #=====================
            #==permutate ind
            #=====================
            self.trainning_select_minfake_minreal_majreal_ind = trainning_select_minfake_minreal_majreal_ind[torch.randperm(trainning_select_minfake_minreal_majreal_ind.shape[0])]
            self.test_select_minfake_minreal_majreal_ind = test_select_minfake_minreal_majreal_ind[torch.randperm(test_select_minfake_minreal_majreal_ind.shape[0])]
            self.test_select_minreal_majreal_ind = test_select_minreal_majreal_ind[torch.randperm(test_select_minreal_majreal_ind.shape[0])]
            self.trainning_select_minreal_majreal_ind = trainning_select_minreal_majreal_ind[torch.randperm(trainning_select_minreal_majreal_ind.shape[0])]

            self.select_min_fake_ind = select_min_fake_ind
            self.test_select_maj_real_ind= test_select_maj_real_ind
            self.test_select_min_real_ind= test_select_min_real_ind
            self.trainning_select_maj_real_ind= trainning_select_maj_real_ind
            self.trainning_select_min_real_ind= trainning_select_min_real_ind

        else:

            trainning_select_min_real_ind = torch.tensor(trainning_selected_min_ind).type(torch.long)
            # trainning_select_maj_real_ind = torch.tensor(trainning_selected_maj_ind).type(torch.long)

            trainning_select_maj_real_ind = torch.tensor(
                np.random.choice(trainning_selected_maj_ind,
                                 size=trainning_select_min_real_ind.shape[0] * 2,
                                 replace=False)).type(torch.long)

            test_select_min_real_ind = torch.tensor(test_selected_min_ind).type(torch.long)
            test_select_maj_real_ind = torch.tensor(test_selected_maj_ind).type(torch.long)

            test_select_minreal_majreal_ind = torch.cat(( test_select_min_real_ind, test_select_maj_real_ind), 0)
            trainning_select_minreal_majreal_ind = torch.cat(( trainning_select_min_real_ind, trainning_select_maj_real_ind), 0)
            #=====================
            #==permutate ind
            #=====================
            self.test_select_minreal_majreal_ind = test_select_minreal_majreal_ind[torch.randperm(test_select_minreal_majreal_ind.shape[0])]
            self.trainning_select_minreal_majreal_ind = trainning_select_minreal_majreal_ind[torch.randperm(trainning_select_minreal_majreal_ind.shape[0])]

            self.test_select_maj_real_ind= test_select_maj_real_ind
            self.test_select_min_real_ind= test_select_min_real_ind
            self.trainning_select_maj_real_ind= trainning_select_maj_real_ind
            self.trainning_select_min_real_ind= trainning_select_min_real_ind


        # # =====================
        # # ==min_fake, min_real, maj boolean ind
        # # =====================
        # trainning_select_min_real_boolean = torch.zeros(
        #     minreal_minfake_majreal_x.shape[0])
        # trainning_select_min_real_boolean[trainning_select_min_real_ind] = 1
        #
        # trainning_select_maj_real_boolean = torch.zeros(
        #     minreal_minfake_majreal_x.shape[0])
        # trainning_select_maj_real_boolean[trainning_select_maj_real_ind] = 1
        #
        # test_select_min_real_boolean = torch.zeros(
        #     minreal_minfake_majreal_x.shape[0])
        # test_select_min_real_boolean[test_select_min_real_ind] = 1
        #
        # test_select_maj_real_boolean = torch.zeros(
        #     minreal_minfake_majreal_x.shape[0])
        # test_select_maj_real_boolean[test_select_maj_real_ind] = 1
        #
        # select_min_fake_boolean = torch.zeros(
        #     minreal_minfake_majreal_x.shape[0])
        # select_min_fake_boolean[select_min_fake_ind] = 1
        #
        #
        # # =====================
        # # ==trainning_select_minreal_minfake_majreal_ind_boolean, test_select_minreal_minfake_majreal_ind_boolean
        # # =====================
        # select_minreal_minfake_majreal_ind_boolean = torch.zeros(
        #     minreal_minfake_majreal_x.shape[0])
        # select_minreal_minfake_majreal_ind_boolean[
        #     trainning_select_minfake_minreal_majreal_ind] = 1
        # trainning_select_minreal_minfake_majreal_ind_boolean = select_minreal_minfake_majreal_ind_boolean.type(
        #     torch.BoolTensor)
        #
        # select_minreal_minfake_majreal_ind_boolean = torch.zeros(
        #     minreal_minfake_majreal_x.shape[0])
        # select_minreal_minfake_majreal_ind_boolean[
        #     test_select_minfake_minreal_majreal_ind] = 1
        # test_select_minreal_minfake_majreal_ind_boolean = select_minreal_minfake_majreal_ind_boolean.type(
        #     torch.BoolTensor)
        #
        # select_minreal_nominfake_majreal_ind_boolean = torch.zeros(
        #     minreal_minfake_majreal_x.shape[0])
        # select_minreal_nominfake_majreal_ind_boolean[
        #     test_select_minfake_nominreal_majreal_ind] = 1
        # test_select_minreal_nominfake_majreal_ind_boolean = select_minreal_nominfake_majreal_ind_boolean.type(
        #     torch.BoolTensor)
        #
        # # select_minreal_majreal_ind_boolean = torch.zeros(
        # #     self.data.y.shape[0])
        # # select_minreal_majreal_ind_boolean[
        # #     test_selected_min_ind] = 1
        # # test_select_minreal_majreal_ind_boolean = select_minreal_majreal_ind_boolean.type(
        # #     torch.BoolTensor)
        #
        # self.trainning_select_min_real_boolean = trainning_select_min_real_boolean.type(
        #     torch.BoolTensor)
        # self.trainning_select_maj_real_boolean = trainning_select_maj_real_boolean.type(
        #     torch.BoolTensor)
        # self.test_select_min_real_boolean = test_select_min_real_boolean.type(
        #     torch.BoolTensor)
        # self.test_select_maj_real_boolean = test_select_maj_real_boolean.type(
        #     torch.BoolTensor)
        # self.select_min_fake_boolean = select_min_fake_boolean.type(
        #     torch.BoolTensor)
        #
        # # self.data.train_mask = trainning_select_minreal_minfake_majreal_ind_boolean
        # # self.data.test_mask = test_select_minreal_nominfake_majreal_ind_boolean
        # # self.data.test_mask = test_select_minreal_majreal_ind_boolean
        #
        #
        # self.trainning_select_minreal_minfake_majreal_ind_boolean = trainning_select_minreal_minfake_majreal_ind_boolean
        # self.test_select_minreal_nominfake_majreal_ind_boolean = test_select_minreal_nominfake_majreal_ind_boolean
        # # self.test_select_minreal_minfake_majreal_ind_boolean = test_select_minreal_minfake_majreal_ind_boolean
        # # self.test_select_minreal_majreal_ind_boolean = test_select_minreal_majreal_ind_boolean

