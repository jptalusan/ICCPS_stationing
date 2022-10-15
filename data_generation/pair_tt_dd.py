import os
import osmnx as ox
import networkx as nx
import pandas as pd
import itertools
import pickle
fp = '/home/jptalusan/mta_simulator/code/data/stops_node_matching.pkl'
stop_nodes = pd.read_pickle(fp)

fp  = '/home/jptalusan/mta_simulator/code/data/davidson_graph.graphml'
G = ox.load_graphml(fp)

stop_nodes[stop_nodes['stop_id_original'].isin(['12ADEMNN', '12ADIVNN'])]
all_stops_list = stop_nodes.stop_id_original.tolist()
all_nodes_list = stop_nodes.nearest_node.tolist()
all_stops_combos = list(itertools.combinations(all_stops_list, 2))
all_nodes_combos = list(itertools.combinations(all_nodes_list, 2))
all_nodes_combos[0], all_stops_combos[0]
from multiprocessing import Process, Queue, cpu_count, Manager
from time import sleep

queue = Queue()

def mp_worker(L, queue):

    while queue.qsize() >0 :
        record = queue.get()
        current_node = record[0]
        next_node = record[1]
        try:
            r = ox.shortest_path(G, current_node, next_node, weight='length')
            cols = ['osmid', 'length', 'travel_time']
            attrs = ox.utils_graph.get_route_edge_attributes(G, r)
            tt = pd.DataFrame(attrs)[cols]['travel_time'].sum()
            dd = pd.DataFrame(attrs)[cols]['length'].sum()
        except:
            tt = -1
            dd = -1
        L.append((current_node, next_node, tt, dd))


def mp_handler():
    with Manager() as manager:
        L = manager.list()
        
        # Spawn two processes, assigning the method to be executed 
        # and the input arguments (the queue)
        processes = [Process(target=mp_worker, args=(L,queue,)) for _ in range(cpu_count() - 4)]

        for process in processes:
            process.start()

        for process in processes:
            process.join()
            
        fp = os.path.join('/home/jptalusan/mta_simulator/code/data/pair_dd_tt.pkl')
        with open(fp, 'wb') as f:
            pickle.dump(list(L), f)

        # print(L)
if __name__ == '__main__':

    for pair in all_nodes_combos:
        queue.put(pair)

    mp_handler()

    fp = os.path.join('/home/jptalusan/mta_simulator/code/data/pair_dd_tt.pkl')
    with open(fp, 'rb') as f:
        res = pickle.load(f)
    print(res)
