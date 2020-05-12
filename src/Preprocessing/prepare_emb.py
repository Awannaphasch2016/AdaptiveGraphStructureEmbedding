import numpy as np

def apply_tsne_on_emb( emb, run_gcn=None):
    assert run_gcn is not None, "run_gcn must be specified to avoid ambiguity"
    from sklearn.manifold import TSNE

    emb_2d = TSNE(n_components=2).fit_transform(emb)
    return emb_2d

    # output (4, 2)
    # if not run_gcn:
    #     emb_dict.setdefault('test', emb_2d[self.data.test_selected_ind])
    #     emb_dict.setdefault('train', emb_2d[self.data.trainning_selected_ind])
    #     emb_dict.setdefault('min_real', np.concatenate((emb_2d[
    #                                                         self.data.trainning_select_min_real_ind.cpu().detach().numpy()],
    #                                                     emb_2d[
    #                                                         self.data.test_select_min_real_ind.cpu().detach().numpy()]),
    #                                                    axis=0))
    #     emb_dict.setdefault('min_fake', emb_2d[self.data.select_min_fake_ind])
    #     emb_dict.setdefault('maj', np.concatenate((emb_2d[
    #                                                    self.data.trainning_select_maj_real_ind.cpu().detach().numpy()],
    #                                                emb_2d[
    #                                                    self.data.test_select_maj_real_ind.cpu().detach().numpy()]),
    #                                               0))
    # else:
    #     emb_dict.setdefault('test', emb_2d[self.data.test_selected_ind])
    #     emb_dict.setdefault('train', emb_2d[self.data.trainning_selected_ind])
    #     emb_dict.setdefault('min_real', np.concatenate((emb_2d[
    #                                                         self.data.trainning_select_min_real_ind.cpu().detach().numpy()],
    #                                                     emb_2d[
    #                                                         self.data.test_select_min_real_ind.cpu().detach().numpy()]),
    #                                                    axis=0))
    #     emb_dict.setdefault('maj', np.concatenate((emb_2d[
    #                                                    self.data.trainning_select_maj_real_ind.cpu().detach().numpy()],
    #                                                emb_2d[
    #                                                    self.data.test_select_maj_real_ind.cpu().detach().numpy()]),
    #                                               0))

    return emb_dict
