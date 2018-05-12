# Search methods

import search

ab = search.GPSProblem('A', 'B', search.romania)

tf = search.GPSProblem('T', 'F', search.romania)

nm = search.GPSProblem('N', 'M', search.romania)


#print search.breadth_first_graph_search(ab).path()
#print search.depth_first_graph_search(ab).path()
#print search.iterative_deepening_search(ab).path()
#print search.depth_limited_search(ab).path()


print "-----Camino de A a B-----"
print "Sin subestimacion: ", search.Ramificacion_search(ab).path()
print "Con subestimacion: ", search.RamificacionConSubestimacion_search(ab).path()

print "-----Camino de T a F-----"
print "Sin subestimacion: ", search.Ramificacion_search(tf).path()
print "Con subestimacion: ",search.RamificacionConSubestimacion_search(tf).path()

print "-----Camino de N a M-----"
print "Sin subestimacion: ", search.Ramificacion_search(nm).path()
print "Con subestimacion: ",search.RamificacionConSubestimacion_search(nm).path()

#print search.astar_search(ab).path()

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450
