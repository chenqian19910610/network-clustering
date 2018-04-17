"""https://github.com/chenqian19910610/network-clustering/blob/master/network-clustering-simple.ipynb"""
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN

ox.config(log_console=True, use_cache=True)

stats = ox.basic_stats(ox.graph_from_place('Emeryville, California, USA', network_type='bike'))
# print(stats)

place = 'Emeryville, California, USA'
gdf = ox.gdf_from_place(place)
area = ox.project_gdf(gdf).unary_union.area
G = ox.graph_from_place(place, network_type='bike',buffer_dist=500)

eps = 300
minpts = 3
n_firms = 30
n_clusters = 3

np.random.seed(7)
firm_centers = np.random.choice(G.nodes(), size=n_clusters, replace=False)

xs = []
ys = []
for osmid in firm_centers:
    x = G.node[osmid]['x']
    y = G.node[osmid]['y']
    np.random.seed(1)
    xs.extend(np.random.normal(loc=x, scale=0.001, size=int(n_firms / len(firm_centers))))
    ys.extend(np.random.normal(loc=y, scale=0.001, size=int(n_firms / len(firm_centers))))

firms = pd.DataFrame({'x': xs, 'y': ys})
len(firms)

fig, ax = ox.plot_graph(G, node_color='#aaaaaa', node_size=0, show=False, close=True)
ax.scatter(x=firms['x'], y=firms['y'], c='k', marker='.', s=50, zorder=3)
fig.canvas.draw()
fig.savefig('scatter.png')

eps_rad = eps / 3671000. #meters to radians
db = DBSCAN(eps=eps_rad, min_samples=minpts, metric='haversine', algorithm='ball_tree')
firms['spatial_cluster'] = db.fit_predict(np.deg2rad(firms[['y', 'x']]))

len(firms['spatial_cluster'].unique())

color_map = {-1:'gray', 0:'g', 1:'r', 2:'m', 3:'b'}
point_colors = [color_map[c] for c in firms['spatial_cluster']]
fig, ax = ox.plot_graph(G, node_size=0, show=False, close=True)
ax.scatter(x=firms['x'], y=firms['y'], c=point_colors, marker='.', s=50, zorder=3)
fig.canvas.draw()
fig.savefig('cluster.png')





# stats = ox.basic_stats(G, area=area)
# extended_stats = ox.extended_stats(G, ecc=True, bc=True, cc=True)
# for key, value in extended_stats.items():
#     stats[key] = value
# # print(pd.Series(stats))
#
# stats = ox.basic_stats(G, area=area)
# for k, count in stats['streets_per_node_counts'].items():
#     stats['int_{}_count'.format(k)] = count
# for k, proportion in stats['streets_per_node_proportion'].items():
#     stats['int_{}_prop'.format(k)] = proportion
#
# # unpack dicts into individiual keys:values
# stats = ox.basic_stats(G, area=area)
# for k, count in stats['streets_per_node_counts'].items():
#     stats['int_{}_count'.format(k)] = count
# for k, proportion in stats['streets_per_node_proportion'].items():
#     stats['int_{}_prop'.format(k)] = proportion
#
# # delete the no longer needed dict elements
# del stats['streets_per_node_counts']
# del stats['streets_per_node_proportion']
#
# print(pd.DataFrame(pd.Series(stats)).T)
# # print(stats['int_1_prop'])
# # print(stats['int_3_prop'])
# # print(stats['int_4_prop'])
#
# # print(G.graph['streets_per_node'])
# #
# # print(list(G.nodes(data=True))[0:2])
# nc = ['r' if G.graph['streets_per_node'][node]==1 else 'none' for node in G.nodes()]
# fig, ax = ox.plot_graph(G, node_color=nc, node_zorder=2, fig_height=4)
#
# # define a lat-long point, create network around point, define origin/destination nodes
# # location_point = (37.791427, -122.410018)
# # G = ox.graph_from_point(location_point, distance=500, distance_type='network', network_type='walk')
# # origin_node = ox.get_nearest_node(G, location_point)
# # destination_node = G.nodes()
#
# # find the route between these nodes then plot it
# # route = nx.shortest_path(G, origin_node, destination_node, weight='length')
# # print(route)
#
# # print(G.node[route[0]])
# # print(G.edge[route[0]][route[1]])
# # print(G.edge[route[0]][route[1]][0]['length'])
# # totallen= nx.shortest_path_length(G, origin_node, destination_node, weight='length')
# # fig1, ax1 = ox.plot_graph_route(G, route)
#
# # project the network to UTM (zone calculated automatically) then plot the network/route again
# # G_proj = ox.project_graph(G)
# # fig2, ax2 = ox.plot_graph_route(G_proj, route)
#
#
# # define origin/desination points then get the nodes nearest to each
# origin_point = (37.838008, -122.283776)
# destination_point = (37.843158, -122.293904)
# origin_node = ox.get_nearest_node(G, origin_point)
# destination_node = ox.get_nearest_node(G, destination_point)
#
# route = nx.shortest_path(G, origin_node, destination_node, weight='length')
# fig3, ax3 = ox.plot_graph_route(G, route, origin_point=origin_point, destination_point=destination_point)
#
# G = ox.graph_from_address('N Corsica Pl, Chandler, Arizona', distance=800, network_type='walk')
# nc = ['r' if G.graph['streets_per_node'][node]==1 else 'none' for node in G.nodes()]
# fig, ax = ox.plot_graph(G, node_color=nc, node_zorder=2, fig_height=4)
# origin = (33.307808, -111.907407)
# destination = (33.309446, -111.901064)
# origin_node = ox.get_nearest_node(G, origin)
# destination_node = ox.get_nearest_node(G, destination)
# route = nx.shortest_path(G, origin_node, destination_node)
# fig4, ax4 = ox.plot_graph_route(G, route, save=True, filename='route')
#
# location_point = (37.806195, -122.295276)
# G = ox.graph_from_point(location_point, distance=900, clean_periphery=False)
# origin = (37.810915, -122.300548)
# destination = (37.800915, -122.280548)
# origin_node = ox.get_nearest_node(G, origin)
# destination_node = ox.get_nearest_node(G, destination)
# route = nx.shortest_path(G, origin_node, destination_node)
# fig, ax = ox.plot_graph_route(G, route)