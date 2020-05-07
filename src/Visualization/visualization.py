
import numpy as np
import pandas as pd

def draw_graph_before_embedding(G, file_to_be_plotted=None, strategy=None, label=None, dataset = None):
    assert isinstance(file_to_be_plotted, str),  "is_embedding must be specified to avoid ambiguity"
    assert isinstance( label, str), ''
    assert isinstance(dataset, str), ''
    if strategy is not None:
        assert isinstance(strategy, str), ''


    # TODO shoud I create karate_club classes ?(currently networkx object represent karate_club_graph)
    # G= preprocess_karate_club(add_edges=True)
    file_type = file_to_be_plotted.split('.')[-1]
    if file_type == 'embeddings':
        is_embedding = True
    elif file_type == 'adjlist':
        is_embedding = False
    else:
        raise ValueError('file_type is not recognized ')

    if not is_embedding:

        draw_nx_graph(G, label=label)
    else:

        # TODO here>> raise error if path_to_saved_emb_filed does not exist and args.enforce_end2end is False
        df = pd.read_csv(file_to_be_plotted, sep=' ', skiprows=0, header=1)
        emb_df = pd.DataFrame(
            np.insert(df.values, 0, values=list(df.columns), axis=0))
        print('content of saved_emb_file is shown below ')
        print(emb_df.head())
        emb_df.set_index(0, inplace=True)
        # del emb_df.index.name
        emb_df.columns = list(range(emb_df.shape[1]))

        emb_np = emb_df.to_numpy()
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        # X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
        X_embedded = TSNE(n_components=2).fit_transform(emb_np)
        # X_embedded = PCA(n_components=2).fit_transform(emb_np)
        x_embedding_df = pd.DataFrame(X_embedded, index=emb_df.index)
        # print(x_embedding_df)

        pos = {}
        for node in list(x_embedding_df.index):
            if strategy is not None:
                # if int(node) - 1 in pos:
                if int(node) in pos:
                    raise ValueError('error')
                # pos[int(node) - 1] = x_embedding_df.loc[node].to_numpy().tolist()
                pos[int(node) ] = x_embedding_df.loc[node].to_numpy().tolist()
            else:
                if int(node)  in pos:
                    raise ValueError('error')
                pos[int(node) ] = x_embedding_df.loc[node].to_numpy().tolist()

        # assert len(set(list(pos.keys()))) == len(list(G.nodes())), 'dk'
        # assert max(list(pos.keys())) == max(list(G.nodes())), 'd'
        # assert min(list(pos.keys())) == min(list(G.nodes())), 'd'
        # TODO check why do I need G here? what did I do to G before I need to pass it in.
        draw_nx_graph(G, pos=pos, label=label)


def draw_nx_graph(G, pos=None, label=None):
    """
    G should have attribute name
    :param G: type = networkx
    :return:
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    assert isinstance( label, str), ""

    # G = nx.cubical_graph()
    if pos is None:
        pos = nx.spring_layout(G)  # positions for all nodes
    # print(pos)
    # nodes
    options = {"node_size": 50, "alpha": 0.8}
    colors = {}
    label_ind = {}
    count = 0

    for i, j in G.nodes(data=True):
        if j[label] not in label_ind:
            label_ind[j[label]] = count
            count += 1
        colors[i] = label_ind[j[label]]

    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()),
                           node_color=list(colors.values()), ax=ax, **options)

    # TODO here>> plot citeseer graph with edges
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()
    print()
    exit()

    # edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # TODO show 2 types of edges. edges of nodes with the same class and nodes with different classes
    colors_edges = {}
    colors_ind = {}
    count = 0
    for edges in G.edges():
        if G.nodes[edges[0]][label] == G.nodes[edges[1]][label]:
            if G.nodes[edges[0]][label] not in colors_ind:
                count += 1
                colors_ind[G.nodes[edges[0]][label]] = count
            colors_edges[edges] = colors_ind[G.nodes[edges[0]][label]]
        else:
            # TODO OPTIMIZABLE keys of colors_ind below could be further optimize if it takes too long to computer
            if (G.nodes[edges[0]][label],
                G.nodes[edges[1]][label]) not in colors_ind or \
                    (G.nodes[edges[1]][label],
                     G.nodes[edges[0]][label]) not in colors_ind:
                count += 1
                colors_ind[(G.nodes[edges[1]][label],
                            G.nodes[edges[0]][label])] = count

                colors_ind[(G.nodes[edges[0]][label],
                            G.nodes[edges[1]][label])] = count

            colors_edges[edges] = colors_ind[(G.nodes[edges[1]][label],
                                              G.nodes[edges[0]][label])]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=list(G.edges()),
        width=4,
        alpha=0.5,
        edge_color=list(colors_edges.values()),
    )

    # some math labels
    labels = {}

    # # TODO how to get types of attributes value
    for i, j in G.nodes(data=True):
        labels[i] = j[label]
    # nx.draw_networkx_labels(G, pos, labels, font_size=16)

    # plt.axis("off")
    # ax.tick_params( left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()
    print()
