from twisted.internet import protocol, reactor
from twisted.internet import ssl as twisted_ssl
import dns.resolver
import netifaces as ni


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import ast
from sklearn.metrics import mean_squared_error
import math
from tabula.io import read_pdf



class KnnLocalizer:

    """ KNN Regression Class for indoor localization using a 3-dimensional label space
        e.g.  Longitude, Latitude, and Floor

    Attributes:
        features - training data features - Average RSSI
        labels -  training data labels - Coordinates
        k (int) number of nearest neighbors

    """

    def __init__(self, features, labels, k=3):

        self.features = features
        self.labels = labels
        self.k = k

    @staticmethod
    def euclidean_distance(array1, array2): #Calculate the difference between each RSSI

        """ Euclidean distance function without squareroot taken for efficiency
		Args:
			Two arrays of equal length

		Returns:
			Distance squared between RSSI1 and RSSI2
		"""
        #print(array1) #wifi fingerprint
        #print(array2) #dataset
        distance = np.sum((array1-array2)**2)
        distance = math.sqrt(distance)  
        return distance
    
    @staticmethod
    def timsort(array, element):

        """ Sort method based on Python's sort() function that sorts tuples in descending order

        Args:
            array (array or list of tuples) array of tuples containing distance and label
            element (Int) the index of the element within tuple that is used to sort tuples

        Returns:
            Array of tuples sorted in descending order at index given by element
        """

        array.sort(key=lambda tup: tup[element])
        return array


    def knn_regression(self, instance):

        """ KNN regression function modified to return regression predictions
         of 3-dimensional labelspace: (RSSI, logitude, latitude).

        Args:
            Test_features (array) test RSSI for which to predict location

        Returns:
            An array containing the top 3 predicted coordinates
        """
        size_input = len(instance)
        temp_results = []
        for i in range(len(self.features)):
            
            RSSI = self.euclidean_distance(instance, self.features[i][:size_input])
            temp_results.append((self.labels[i], RSSI)) #each localisation with the difference of RSSI   
        sorted_distances = self.timsort(temp_results, 1)
        return sorted_distances[:3] #return the 3 localisation with minimum RSSI 

    def fit_predict(self, test_data):

        """ KNN Localizer prediction function

        Args:
        	test_data (array) test RSSI

        Returns:
        	List of arrays containing coordinate predictions
        """

        predictions = []
        for sample in test_data:
            predictions.append(self.knn_regression(sample))
        return predictions
    

MAX_CHARS = 128
def get_mean_point(points, debug=False):
    # Initialize mean point
    mean_point = [0,0]

    try:
        if len(points) > 0:
        # Calculate the mean position of the points
            for point in points:
                mean_point[0] += point[0]
                mean_point[1] += point[1]
            mean_point = [mean_point[0]/len(points), mean_point[1]/len(points)]

    except:
        print('[get_mean_point] ERROR: points [' + str(len(points)) + '] = ' + str(points)[:MAX_CHARS])
    finally:
        if debug:
            print('[get_mean_point] debug: mean_point = ' + str(mean_point))

    return mean_point    


def get_position_f(new_predictions, mean):
    distances = []
    for x in new_predictions[0]:
        distances.append(localizer.euclidean_distance(mean[0], x[0]))

    min_pos = distances.index(min(distances))
    pos = new_predictions[0][min_pos][0][0]
    for i in range(len(data['Coordinates'].values)):
        if data['Coordinates'].values[i][0] == pos:
            print("Coordinates from KNN-mean prediction:", data['Coordinates'].values[i])
            print("Place: ", data['Tag'].values[i])
        if data['Coordinates'].values[i][0] == new_predictions[0][0][0][0]:
            print("Coordinates from KNN prediction:", data['Coordinates'].values[i])
            print("Place: ", data['Tag'].values[i])


def get_position_s(new_predictions, mean, map): #Get the tagged position according to the prediction
    
    distances = []
    for x in map:
        distances.append(localizer.euclidean_distance(np.array(mean), np.array(x)))

    min_pos = distances.index(min(distances))
    pos = new_predictions[0][min_pos]
    print("Result from the mean predicition: \n", pos[0][0])
    print("Coordinate from mean prediction: \n", pos[0][1])



#Load second database
# df = read_pdf("RSSI.pdf", pages="all")
# places = ['A0', 'A1', 'B0', 'B1', 'C0', 'C1', 'D0', 'D1', 'E0', 'E1']
# for x in range(len(df)):
#   #df[x].loc[len(df[x])] = df[x].columns
#   df[x]['place'] = places[x]
#   df[x] = df[x].set_axis(["status", "net", "rssi", "mac", "max_rssi", 'coordinates', 'place'], axis=1)
  

# data = pd.concat([df[0], df[1], df[2], df[3], df[4], df[5], df[6], df[7], df[8], df[9]])
# data['max_rssi'] = data['max_rssi'].astype(int)
# data['rssi'] = data['rssi'].astype(int)
# data['delta'] = data.apply(lambda y: y['max_rssi'] - y['rssi'], axis=1)
# print(data)
# rssi_values = []
# for x in places:
#   prev = data.loc[data['place'] == x]
#   prev = prev.sort_values(by='delta', ascending=True)
#   rssi_values.append(prev['rssi'].astype(int).values.tolist())


# x = rssi_values
# cord = data["coordinates"].drop_duplicates().apply(ast.literal_eval).unique()
# y = [item for sublist in [zip(places, cord)] for item in sublist]


#Load first database
data = pd.read_excel("Database_1.xlsx")
data = data.drop(data.columns[[1, 2]], axis=1)
data.columns = ["Coordinates", "1", "2", "3", "Average", "MAC"]
data['Average'] = data['Average'].apply(pd.to_numeric)
data[['Tag', 'Coordinates']] = data.pop('Coordinates').str.split('-', n=1, expand=True)
data["Coordinates"] = data["Coordinates"].str.strip().apply(ast.literal_eval).apply(pd.to_numeric)
#data = data[data['Tag'].str.contains('F2')==False]
#Build predict variables
x = data[['1', '2', '3']].values
y = data['Coordinates'].values
k = 1 #neighbors
localizer = KnnLocalizer(x, y, k)
    
# Adapted from http://stackoverflow.com/a/15645169/221061

class TCPProxyProtocol(protocol.Protocol):
    """
    TCPProxyProtocol listens for TCP connections from a
    client (eg. a phone) and forwards them on to a
    specified destination (eg. an app's API server) over
    a second TCP connection, using a ProxyToServerProtocol.

    It assumes that neither leg of this trip is encrypted.
    """
    def __init__(self):
        self.buffer = None
        self.proxy_to_server_protocol = None
 
    def connectionMade(self):
        """
        Called by twisted when a client connects to the
        proxy. Makes an connection from the proxy to the
        server to complete the chain.
        """
        print("Connection made from CLIENT => PROXY")
        proxy_to_server_factory = protocol.ClientFactory()
        #proxy_to_server_factory.protocol = ProxyToServerProtocol
        proxy_to_server_factory.server = self
 
        #reactor.connectTCP(DST_IP, DST_PORT,
                           #proxy_to_server_factory)
 
    def dataReceived(self, data):
        """
        Called by twisted when the proxy receives data from
        the client. Sends the data on to the server.

        CLIENT ===> PROXY ===> DST
        """
        print("")
        print("CLIENT => SERVER")
        print(data)
        print("")
        if self.proxy_to_server_protocol:
            self.proxy_to_server_protocol.write(data)
        else:
            self.buffer = data
            
            #Get values of RSSI from data
            pos = []
            loop = 0
            new_data = data
            while(1):
                loop = new_data.find(b'rssi')
                if(loop != -1):
                    pos.append(int(new_data[loop+6:loop+9])) 
                    r = new_data[loop+1:]
                    new_data = r
                else : break    
                
            
            print(pos)
            if(len(pos) > 0):
                pred = localizer.fit_predict(np.array([pos[:3]])) #Get the 3 most possible routers
                #pred = localizer.fit_predict(np.array([pos])) #second database
                #print("Result from pred:\n", pred[0][0][0][0])
                print("Result from pred:\n", pred[0])
                #top = [i[0][0] for i in pred[0]]
                #print("Other top results: \n", top)
                #map = [i[0][1] for i in pred[0]]
                mean = get_mean_point(pred[:][0])
                print("Mean position give: ", mean[0])
                get_position_f(pred, mean)
                #mean = get_mean_point(map) #Return the mean position among the routers
                #get_position_s(pred, mean, map) #Get the most probably fingerprint got compared to the mean
                




 
    def write(self, data):
        self.transport.write(data)
 
 
# class ProxyToServerProtocol(protocol.Protocol):
#     """
#     ProxyToServerProtocol connects to a server over TCP.
#     It sends the server data given to it by an
#     TCPProxyProtocol, and uses the TCPProxyProtocol to
#     send data that it receives back from the server on
#     to a client.
#     """

#     def connectionMade(self):
#         """
#         Called by twisted when the proxy connects to the
#         server. Flushes any buffered data on the proxy to
#         server.
#         """
#         print("Connection made from PROXY => SERVER")
#         self.factory.server.proxy_to_server_protocol = self
#         self.write(self.factory.server.buffer)
#         self.factory.server.buffer = ''
 
#     def dataReceived(self, data):
#         """
#         Called by twisted when the proxy receives data
#         from the server. Sends the data on to to the client.

#         DST ===> PROXY ===> CLIENT
#         """
#         print("")
#         print("SERVER => CLIENT")
#         print(FORMAT_FN(data))
#         print("")
#         self.factory.server.write(data)
 
#     def write(self, data):
#         if data:
#             self.transport.write(data)


def _noop(data):
    return data

def get_local_ip(iface):
    ni.ifaddresses(iface)
    return ni.ifaddresses(iface)[ni.AF_INET][0]['addr']

FORMAT_FN = _noop



LISTEN_PORT = 80
DST_PORT = 80
DST_HOST = "nonhttps.com"
local_ip = get_local_ip('wlp1s0')

# Look up the IP address of the target
print("Querying DNS records for %s..." % DST_HOST)
a_records = dns.resolver.resolve(DST_HOST, 'A')
print("Found %d A records:" % len(a_records))
for r in a_records:
    print("* %s" % r.address)
print("")
assert(len(a_records) > 0)

# THe target may have multiple IP addresses - we
# simply choose the first one.
DST_IP = a_records[0].address
print("Choosing to proxy to %s" % DST_IP)

print("""
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
-#-#-#-#-#-RUNNING  TCP PROXY-#-#-#-#-#-
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

Dst IP:\t%s
Dst port:\t%d
Dst hostname:\t%s

Listen port:\t%d
Local IP:\t%s
""" % (DST_IP, DST_PORT, DST_HOST, LISTEN_PORT, local_ip))
 
print("""
Next steps:

1. Make sure you are spoofing DNS requests from the
device you are trying to proxy request from so that they
return your local IP (%s).
2. Make sure you have set the destination and listen ports
correctly (they should generally be the same).
3. Use the device you are proxying requests from to make
requests to %s and check that they are logged in this
terminal.
4. Look at the requests, write more code to replay them,
fiddle with them, etc.

Listening for requests on %s:%d...
""" % (local_ip, DST_HOST, local_ip, LISTEN_PORT))

factory = protocol.ServerFactory()
factory.protocol = TCPProxyProtocol
reactor.listenTCP(LISTEN_PORT, factory)
reactor.run()
