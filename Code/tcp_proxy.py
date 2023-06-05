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
from math import sqrt

class KnnLocalizer:

    """ KNN Regression Class for indoor localization using a 3-dimensional label space
        e.g.  Longitude, Latitude, and Floor

    Attributes:
        features - training data features - Average RSSI
        labels -  training data labels - Coordinates
        k (int) number of nearest neighbors

    """

    def __init__(self, features, labels, k=3):

        self.features = np.array(features)
        self.labels = np.array(labels)
        self.k = k

    @staticmethod
    def euclidean_distance(array1, array2): #Calculate the diference between each RSSI

        """ Euclidean distance function without squareroot taken for efficiency

		Args:
			Two arrays of equal length

		Returns:
			Distance squared between RSSI1 and RSSI2
		"""

        distance = np.sum((array1-array2)**2)
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
            An array containing coordinate prediction
        """
        temp_results = []
        for i in range(len(self.features)):
            
            RSSI = self.euclidean_distance(instance, self.features[i])
            temp_results.append((self.labels[i], RSSI)) #each localisation with the difference of RSSI   
        sorted_distances = self.timsort(temp_results, 1)
        return np.stack(sorted_distances[0][0], axis=0) #return localisation with minimum difference of RSSI 

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
    
#Load database
data = pd.read_excel("Database.xlsx")
data = data.drop(data.columns[[1, 2]], axis=1)
data.columns = ["Coordinates", "1", "2", "3", "Average", "MAC"]
data['Average'] = data['Average'].apply(pd.to_numeric)
data[['Tag', 'Coordinates']] = data.pop('Coordinates').str.split('-', n=1, expand=True)
data["Coordinates"] = data["Coordinates"].str.strip().apply(ast.literal_eval).apply(pd.to_numeric)
#Build predict variables
x = data['Average'].values.reshape(-1,1)
y = data['Coordinates'].values
k = 3 #neighbors
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
            pos_i = data.find(b'rssi') #rssi":
            if(pos_i >= 0):
                value = int(data[pos_i+6:pos_i+9]) #rssi value
                lista = localizer.fit_predict(np.array(value).reshape(-1,1))
                print(lista) 





 
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
