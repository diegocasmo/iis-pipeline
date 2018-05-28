import threading
import time
import unittest
import socket
import sys 

from multiprocessing import Process, Queue

END_MARKER = '\n'


class Furnet(threading.Thread):

	def __init__(self, host, port):
		threading.Thread.__init__(self)
		self.daemon = True
		
		self.host = host
		self.port = port
		
		self.socket = None
		self.receive_queue = Queue()
		self.running = False
		self.connected = False

	def connect(self):
		try:
			#create an INET, STREAMing socket
			self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.socket.connect((self.host, self.port))
			self.running = True
			self.connected = True
			self.start()
		except Exception as e:
			print "Could not connect to server", e
		
	def close(self):
		""" Close the connection """
		self.running = False
		self.connected = False
		self.socket.close()
		
	def send_message(self, *args):
		""" Create the message string and send the message """
		message_string = ':'.join([str(e) for e in args]) + '\n'
		success = self.socket.sendall(message_string)
		
		# Print message we sent if we sent successfully
		#if success is None:
			#print "Sent message", message_string[0:-1]
			#f = open('myfile', 'a')
                        #f.write(message_string[0:-1]+'\n')  # python will convert \n to os.linesep
                        #f.close()
		return success is None

	def say(self, utterance):
		""" 
		Sends an utterance request to the furhat system
		@param utterance - String that should be uttered
		"""
		return self.send_message('say', utterance)
		
	def gaze(self, x, y, z, head_movement=True):
		""" 
		Passes on a gaze request to a point in space
		@param x - The x coordinate to focus
		@param y - The y coordinate to focus
		@param z - The z coordinate to focus
		@param head_movement - whether head movement 
			should be part of the gaze 
		"""
		assert isinstance(x, float), "Parameter x needs to be float"
		assert isinstance(y, float), "Parameter y needs to be float"
		assert isinstance(z, float), "Parameter z needs to be float"
		
		head_move = 1 if head_movement else 0
		
		return self.send_message('gaze', str(x), str(y), str(z), head_move)
		
	def gesture(self, identifier):
		""" 
		Passes on a request to play a specific gesture
		@param identifier - the name identifying the gesture
		"""
		return self.send_message('gesture', identifier)
	
	
	def extract_message(self, message_buffer):
		""" Extracts one message from the buffer """
		message = message_buffer[:message_buffer.find(END_MARKER)]
		remaining = message_buffer[message_buffer.find(END_MARKER)+len(END_MARKER):]
		return remaining, message
		
	def run(self):
		message_buffer = ''
		
		while self.running:
			message_buffer += self.socket.recv(8192)
			
			while END_MARKER in message_buffer:
				message_buffer, message = self.extract_message(message_buffer)
				self.receive_queue.put(message)
			
			time.sleep(0.02)
		
		print "Ending input reader"


class TestFurnet(unittest.TestCase):
		
	def test_message_splitter_split_off_message(self):
		fn = Furnet('localhost', 1337)
		
		_buffer = "Some message" + END_MARKER + "Other message start"
		
		remaining, message = fn.extract_message(_buffer)
		
		self.assertEquals(remaining, "Other message start")
		self.assertEquals(message, "Some message")
		
	def test_message_splitter_split_off_message_multi_marker(self):
		fn = Furnet('localhost', 1337)
		
		_buffer = "Some message" + END_MARKER + "Other message" + END_MARKER
		
		remaining, message = fn.extract_message(_buffer)
		
		self.assertEquals(message, "Some message")
		self.assertEquals(remaining, "Other message" + END_MARKER)
		
		
	

class Demo():
	""" A demo class doing some simple operations """
	
	def __init__(self, host, port):
		self.furnet = Furnet(host, port)
		
	def execute(self):
		self.furnet.connect()
		
		if self.furnet.connected:
			self.furnet.say('Hello World!')
			self.furnet.gaze(1.0, 1.0, 1.0, True)
			self.furnet.gesture('smile')

			print "Waiting for executed ack responses",
			sys.stdout.flush()
			
			# We should get 3 elements back when all of those 
			# commands are executed, while waiting we sleep
			while True:
				if self.furnet.receive_queue.qsize() > 0:
					print "Received", self.furnet.receive_queue.get()
			self.furnet.close()
		
		
# Block to run tests
if __name__ == '__main__':	
	#unittest.main()
	
	demo = Demo('localhost', 1337)
	demo.execute()

