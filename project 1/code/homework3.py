# -*- coding: utf-8 -*-
"""hw1.py

"""

from queue import PriorityQueue,Queue
from collections import deque 
import math
import numpy as np
import heapq as heap
from copy import copy, deepcopy

def read_input_file():
  with open('input.txt','r') as f:
    lines = f.readlines()
  for i in range (0,len(lines)):
    lines[i] = lines[i].replace("\n","")
  search_type, dimensions_size, entrance,exit, number_of_grid_with_actions,full_graph = lines[0],lines[1],lines[2],lines[3],lines[4],lines[5:] ## graph with actions
  graph_dict = {}
  visited_dict = {}
  for node in full_graph:
    node_info = node.split(" ",len(node))
    #graph.append(node_info[0]+ ' ' +node_info[1]+ ' ' +node_info[2])
    node_coor = node_info[0]+ ' ' +node_info[1]+ ' ' +node_info[2]
    node_coor = " ".join ([node_info[0], node_info[1], node_info[2]])
    graph_dict[node_coor] = node
    visited_dict[node_coor] = -1
 # for key,val in zip(graph,full_graph):
  #  graph_dict[key] = val
  #for i in range (0,len(graph)):
   # graph_dict[graph[i]] = full_graph[i]
  input_processing(search_type,entrance,exit,dimensions_size,graph_dict,visited_dict)
def input_processing(algor_type,entrance,exit,dimensions_size,graph_dict,visited_dict):
    if algor_type =="BFS":
        bfs(entrance,exit,dimensions_size,graph_dict,visited_dict)
    elif algor_type =="UCS":
        ucs(entrance,exit,dimensions_size,graph_dict,visited_dict)
    elif algor_type =="A*":
        a_star(entrance,exit,dimensions_size,graph_dict,visited_dict)
    else:
        final_output_file([],False)

"""# Related Functions"""

def extract_3d_coordinates(node):
  temp = node.split(' ')
  concat_temp = " ".join ([temp[0], temp[1], temp[2]])
  return concat_temp

def out_of_bound(dimensions_size,node):
  dimension_size = dimensions_size.split(" ")
  node = node.split(" ")
  # check if any coordinnates(x,y,z) is more than dimension_size (x,y,z)
  if (int(node[0]) > int(dimension_size[0]) - 1 ) or (int(node[1])> int(dimension_size[1]) - 1)  or (int(node[2]) > int(dimension_size[2])- 1):
    return True #out_of_bound
  else:
    return False
  
def heuristic_3d(node,goal):
  temp_node = node.split(' ')
  temp_goal = goal.split(' ')
  return math.sqrt(((int(temp_node[0]) - int(temp_goal[0]))**2) +  ((int(temp_node[1]) - int(temp_goal[1]))**2) +  ((int(temp_node[2]) - int(temp_goal[2]))**2))

def Cloning(list):
    #list_copy = []
    #list_copy.extend(list)
    list_copy = list[:]
    return list_copy

"""# Action Mappings"""

def action_mapping (node,graph_dict):
  mapped_node = graph_dict[node]
  output_nodes = []
  node_info = mapped_node.split(" ",len(mapped_node))
  x,y,z = node_info[0],node_info[1],node_info[2]
  #extract actions
  for action in node_info[3:]:
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
    if action == '1':
      x_temp = x_temp + 1 
      #type casting
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp]) 
      output_nodes.append([output,10]) 
    elif action == '2':
      x_temp = x_temp -1 
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,10]) 
    elif action == '3':
      y_temp = y_temp +1 
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,10]) 
    elif action == '4':
      y_temp = y_temp -1 
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,10]) 
    elif action == '5':
      z_temp = z_temp +1 
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,10]) 
    elif action == '6':
      z_temp = z_temp -1 
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,10]) 
    elif action == '7':
      x_temp = x_temp +1 
      y_temp = y_temp +1 
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,14]) 
    elif action == '8':
      x_temp = x_temp +1 
      y_temp = y_temp -1 
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,14]) 
    elif action == '9':
      x_temp = x_temp -1 
      y_temp = y_temp +1  
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,14]) 
    elif action == '10':
      x_temp = x_temp -1 
      y_temp = y_temp -1 
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,14]) 
    elif action == '11':
      x_temp = x_temp +1 
      z_temp = z_temp +1  
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,14]) 
    elif action == '12':
      x_temp = x_temp +1 
      z_temp = z_temp -1 
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,14]) 
    elif action == '13':
      x_temp = x_temp -1 
      z_temp = z_temp +1 
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,14]) 
    elif action == '14':
      x_temp = x_temp -1 
      z_temp = z_temp -1  
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,14]) 
    elif action == '15':
      y_temp = y_temp +1 
      z_temp = z_temp +1  
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,14]) 
    elif action == '16':
      y_temp = y_temp +1 
      z_temp = z_temp -1 
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,14]) 
    elif action == '17':
      y_temp = y_temp -1 
      z_temp = z_temp +1 
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,14]) 
    elif action == '18':
      y_temp = y_temp -1 
      z_temp = z_temp -1 
      x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
      output += " ".join([x_temp,y_temp,z_temp])  
      output_nodes.append([output,14]) 
  return output_nodes

def action_mapping_2 (node,graph_dict):
  mapped_node = graph_dict[node]
  output_nodes = []
  node_info = mapped_node.split(" ",len(mapped_node))
  action_list = node_info[3:]
  action = action_list.pop(0)
  x,y,z = int(node_info[0]),int(node_info[1]),int(node_info[2])
  #extract actions
  # for action in node_info[3:]:
  x_temp,y_temp,z_temp = int(x),int(y),int(z)
  output = ''
  if action == '1':
    x_temp = x_temp + 1 
    #type casting
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp]) 
    output_nodes.append([output,10,1])
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '2':
    x_temp = x_temp -1 
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,10,2]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '3':
    y_temp = y_temp +1 
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,10,3]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '4':
    y_temp = y_temp -1 
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,10,4]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '5':
    z_temp = z_temp +1 
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,10,5]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '6':
    z_temp = z_temp -1 
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,10,6]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '7':
    x_temp = x_temp +1 
    y_temp = y_temp +1 
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,14,7]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '8':
    x_temp = x_temp +1 
    y_temp = y_temp -1 
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,14,8]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '9':
    x_temp = int(x_temp) -1 
    y_temp = int(y_temp) +1  
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,14,9]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '10':
    x_temp = x_temp -1 
    y_temp = y_temp -1 
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,14,10]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '11':
    x_temp = x_temp +1 
    z_temp = z_temp +1  
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,14,11]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '12':
    x_temp = x_temp +1 
    z_temp = z_temp -1 
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,14,12]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '13':
    x_temp = x_temp -1 
    z_temp = z_temp +1 
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,14,13]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '14':
    x_temp = x_temp -1 
    z_temp = z_temp -1  
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,14,14]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '15':
    y_temp = y_temp +1 
    z_temp = z_temp +1  
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,14,15]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '16':
    y_temp = y_temp +1 
    z_temp = z_temp -1 
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,14,16]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '17':
    y_temp = y_temp -1 
    z_temp = z_temp +1 
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,14,17]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  if action == '18':
    y_temp = y_temp -1 
    z_temp = z_temp -1 
    x_temp,y_temp,z_temp = str(x_temp),str(y_temp),str(z_temp)
    output += " ".join([x_temp,y_temp,z_temp])  
    output_nodes.append([output,14,17]) 
    if action_list:
       action = action_list.pop(0) 
    x_temp,y_temp,z_temp = int(x),int(y),int(z)
    output = ''
  return output_nodes

def construct_path_from_dict (start,parent_dict,end):
  node = end
  list_node = [end]
  cost =0
  while node in parent_dict.keys():
    node = parent_dict[node]  
    list_node.insert(0,node)
  list_node[0] = list_node[0] + ' 0'
  for i in range (0,len(list_node) -1):
    parent = list_node[i].split(" ",len(list_node[i])) 
    child = list_node[i+1].split(" ",len(list_node[i+1])) 
    x_p,y_p,z_p = int(parent[0]),int(parent[1]),int(parent[2])
    x_c,y_c,z_c = int(child[0]),int(child[1]),int(child[2])
    diff = abs(x_p - x_c) + abs(y_p - y_c) + abs(z_p - z_c)
    if list_node[i] == start:
      list_node[i] = list_node[i] + ' 0'
    elif diff == 1:
      list_node[i+1] = list_node[i+1] + ' 10'
    else:
      list_node[i+1] = list_node[i+1] + ' 14' 
  return list_node

def construct_path_from_dict_bfs (start,parent_dict,end):
  node = end
  list_node = [end]
  cost =0
  while node in parent_dict.keys():
    node = parent_dict[node]  
    list_node.insert(0,node)
  list_node[0] = list_node[0] + ' 0'
  for i in range (0,len(list_node) -1):
    parent = list_node[i].split(" ",len(list_node[i])) 
    child = list_node[i+1].split(" ",len(list_node[i+1])) 
    x_p,y_p,z_p = int(parent[0]),int(parent[1]),int(parent[2])
    x_c,y_c,z_c = int(child[0]),int(child[1]),int(child[2])
    diff = abs(x_p - x_c) + abs(y_p - y_c) + abs(z_p - z_c)
    if list_node[i] == start:
      list_node[i] = list_node[i] + ' 0'
    elif diff == 1:
      list_node[i+1] = list_node[i+1] + ' 1'
    else:
      list_node[i+1] = list_node[i+1] + ' 1' 
  return list_node

"""# BFS"""

def bfs(start,end,dimensions_size,graph_dict,visited_dict):
  visited, visited_check, next_path,path = [] ,{},[],[]
  queue = deque([])
  queue.append(start) #append start node
  visited_dict[start] = 0
  parent_dict ={}
  while queue:
    #First path in Q
    node= queue.popleft()
    # last node from path
    if node == end:
      p = construct_path_from_dict_bfs(start,parent_dict,end)
      final_output_file(p,True)
      return
    next_path = action_mapping(node,graph_dict) #neighbor_dict[node] #
    #for i in range(0,len(next_path)):
    for next in next_path:
      if next[0] in visited_dict.keys():
        if visited_dict[next[0]] == -1 and not out_of_bound(dimensions_size,next[0]):
        #if next[0] not in visited_check and not out_of_bound(dimensions_size,next[0]):
          parent_dict[next[0]] = node
          queue.append(next[0])
          visited_dict[next[0]] = 1
  final_output_file(path,False)

"""# UCS

"""

def ucs(start,end,dimensions_size,graph_dict,visited_dict):
  q = PriorityQueue()
  path= []
  next_path = []
  visited_dict[start] = 0
  parent_dict ={}
  action = []
  q.put((0,start)) #append start node
  while not q.empty():
    cost ,node = q.get()
    node = extract_3d_coordinates(node)
    if node == end: 
      p = construct_path_from_dict(start,parent_dict,end)
      final_output_file(p,True)
      return
    else:
      next_path = action_mapping(node,graph_dict)
      for next in next_path:
        if next[0] in visited_dict.keys():
          total_cost = cost + next[1]
          if (visited_dict[next[0]] == -1 or total_cost < visited_dict[next[0]]) and not out_of_bound(dimensions_size,next[0]) :
            # action_list = action + [next[2]]
            parent_dict[next[0]] = node
            q.put((total_cost, next[0]))
            visited_dict[next[0]] = total_cost
  final_output_file(path,False)

"""# A*"""

def a_star(start,end,dimensions_size,graph_dict,visited_dict):
  q = PriorityQueue()
  visited_dict[start] = 0
  path = []
  parent_dict ={}
  action = []
  q.put((0,start)) #append start node
  while not q.empty():
    cost, node = q.get()
    node = extract_3d_coordinates(node)
    if node == end:
      p = construct_path_from_dict(start,parent_dict,end)
      final_output_file(p,True)
      return
    else:
      next_path = action_mapping(node,graph_dict)
      for next in next_path:
        if next[0] in visited_dict.keys():
          new_cost = cost + int(next[1])
          a_star_score = new_cost + heuristic_3d(next[0], end)
          if (visited_dict[next[0]] == -1 or a_star_score <= visited_dict[next[0]]) and not out_of_bound(dimensions_size,next[0]):
            parent_dict[next[0]] = node
            q.put((a_star_score,next[0]))
            visited_dict[next[0]] = a_star_score #dict of node and cost (visited)
  final_output_file(path,False)

"""# Output

"""

def print_result_format(final_path):
  cost = 0
  for i in range(0,len(final_path)):
    temp = final_path[i].split(' ')
    cost = cost +  int(temp[-1])
  #print(cost)
  #print(len(final_path))
  for i in range(0,len(final_path)):
    print(final_path[i])
  final_output_file(final_path)

def final_output_file(path,success):
  out = open('output.txt','w+')
  if success == True: 
    outstring=""
    cost = 0
    for i in range(0,len(path)):
      temp = path[i].split(' ')
      cost = cost +  int(temp[-1])
    out.write(str(cost))
    out.write("\n")
    out.write(str(len(path)))
    out.write("\n")
    for i in range(0,len(path)):
      #print(type(step))
      outstring += path[i]
      outstring+="\n"       
    out.write(outstring.rstrip())
    out.close()
    return
  else:
    out.write('FAIL')
    out.close()
    return
read_input_file()