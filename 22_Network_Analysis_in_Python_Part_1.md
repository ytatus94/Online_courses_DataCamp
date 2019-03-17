22. Network Analysis in Python (Part 1)

- import networkx as nx
- G = nx.Graph() 是 Undirected graph 沒有方向性，只是將兩個節點連接起來，例如 A -- B
- type(G) 是 networkx.classes.graph.Graph 型態
D = nx.DiGraph() 是 Directed graph 有方向性，例如 A --> B
- type(D) 是 networkx.classes.digraph.DiGraph 型態
M = nx.MultiGraph() 是 networkx.classes.multigraph.MultiGraph 型態
- MD = nx.MultiDiGraph() 是 networkx.classes.multidigraph.MultiDiGraph 型態
- G.add_nodes_from([list]) 新增節點
- G.nodes() 傳回 nodes 的列表，可以有參數 data=True 可以取得 metadata (每個節點的屬性)
- len(G.nodes()) 有多少個節點
- G.node[1]['key'] = 'value' 把 key value 加到第一個節點，key 就是該節點的屬性，value 是該屬性的值，G 變成 [(1, {'key':'value'}), (2, ..), ...]
- 例如：G.node[n]['degree'] = nx.degree(G, n) 其中 nx.degree(G, n) 算節點 n 的 degree
- G.add_edge(A, B)
G.edges() 傳回 tuple 的列表，tuple 內的兩個元素是該 edge 兩端的 nodes，可以有參數 data=True 可以取得 metadata (每個 edge 的屬性)
- len(G.edges()) 節點之間有多少條線連接
- G_lmc.add_edges_from(zip([node]*len(G.neighbors(node)), G.neighbors(node))) 在目前的節點和他的鄰居之間建立 edges
G.has_edge(n1, n2) 判斷 n1 和 n2 之間有沒有 edge 連起來
- nx.draw(G) 可以加上參數 with_labels=True 標記節點
- plt.show() 要有這行才會畫出來
- list comprehension: [ output expression for iterator variable in iterable if conditional expression ]

- 把 edge 加上屬性：G.edge[node1][node2]['attribute'] = value node1 和 node2 是 edge 的兩端點
- 例如：T.edge[1][10]['weight'] = 2 就是把節點 1 和 10 的權重設為 2
- self loop 起點和終點是同一個節點
T.number_of_selfloops() 計算 T 裡面有多少個 self loop
- assert 判斷後面的敘述是否是 True，若不是就傳回 AssertionError
- 例如：assert T.number_of_selfloops() == len(find_selfloop_nodes(T))
- import nxviz as nv
圖形的種類有 MatrixPlot(), ArcPlot(), CircosPlot()
- In a MatrixPlot, the matrix is the representation of the edges.
有方向性的 MatrixPlot 是不對稱的，沒方向性的 MatrixPlot 是對稱的
例如：MatrixPlot(graph=largest_ccs, node_grouping='grouping')
- a2 = ArcPlot(T, node_order='category', node_color='category') 可以指定用哪個 key 屬性來對節點排序 (node_order) 和上色 (node_color)，這裡的 category 是某個 key 屬性
- a = ArcPlot(graph=G, node_order='degree')
- c = CircosPlot(G, node_order='degree', node_grouping='grouping', node_color='grouping')
- A = nx.to_numpy_matrix(T) 把 T 轉成 Numpy matrix
- T_conv = nx.from_numpy_matrix(A, create_using=nx.DiGraph()) 把 Numpy matrix 轉成 Graph 物件，預設是 Graph 型態，可以用 creat_using=nx.DiGraph() 轉成其他形態
- G.neighbors(n) 傳回節點 n 的鄰居，就是一個和 n 連在一起的節點的列表
- Breadth-first search (BFS)
給定起點和終點的節點，由起點開始往外一層一層找，看鄰居有沒有包含終點，若是沒有包含，就再往下一層找，若是包含了就停止搜尋
- degree centrality: 我目前有的鄰居數目 / 我可以有的鄰居數目
nx.degree_centrality(G) 傳回字典，key 是節點，value 是該節點的 degree centrality 的分數，self-loop 不列入考慮
- list( nx.degree_centrality(G).values() ) 因為 degree centrality 傳回一個字典，加上 .values() 可顯示所有的分數，在用 list() 將分數們轉換成列表
- Betweenness centrality =  經過該節點的最短路徑數目 / 整個 Graph 內所有可能最短路徑的數目
nx.betweenness_centrality(G) 傳回字典，key 是節點，value 是該節點的 between centrality 的分數
- 若節點是 fully connect with the other 則 betweenness centrality = 0，通常 barbell 圖的兩端的節點會是這種情況
- The .extend() method appends all the items in a given list.
- 例如：queue.extend([n for n in neighbors if n not in visited_nodes])
- G = nx.barbell_graph(m1=5, m2=1) m1 是 barbell 兩端的節點數目，m2 是 barbell 橋上的節點數目
- Cliques 就是一個緊緊關聯在一起的群體，ㄧ節點和群體內其他的節點都有連在一起
最簡單的是一個 edge 最簡單的一個 complex clique 是三角形
cliques are "groups of nodes that are fully connected to one another"
- a maximal clique is a clique that cannot be extended by adding another node in the graph.
nx.find_cliques(G) 找 G 的最大 clique，傳回一個 generator
- 可以用 list(nx.find_cliques(G)) 來看哪些節點形成一個 clique.
- barbell 圖中連接橋的一個 edge 也是一個 clique
- combinations(iterable, n) returns combinations of size n from iterable，iterable 是一個序列(例如一個列表)，就是說從 iterable 中任選 n 個結合再一起的意思
- from itertools import combinations
for n1, n2 in combinations(G.nodes(), 2):
- nx.triangles(G) 傳回字典，key 是節點，value 是三角形數目
- G = nx.erdos_renyi_graph(n=20, p=0.2) 產生 20 個節點，每個節點之間有 edge 的機率是 0.2
- G.subgraph([節點列表]) 依照節點列表列出的節點，取出 G 中的子圖
- nodeset = nodeset.union(nbrs)
- nx.connected_component_subgraphs(G) 計算最大的 connected component subgraph
- 就是找出所有的沒有和其他子圖連結再一起的子圖
- largest_ccs = sorted(nx.connected_component_subgraphs(G), key=lambda x: len(x))[-1]
- recommended = defaultdict(int)
