
from datetime import datetime
class Log:

	@classmethod
	def info(self,text):
 		print("["+'\033[1;33m'+datetime.now().strftime("%H:%M:%S")+'\033[0m'+"] ["+'\033[1;32m'+"INFO"+'\033[1;33m'+"] "+text)
 	

	@classmethod
	def warning(self,text):
		print("["+'\033[1;33m'+datetime.now().strftime("%H:%M:%S")+'\033[0m'+"] ["+'\033[1;33m'+"WARNING"+ '\033[0m'+"] "+text)
		

	@classmethod
	def high(self,text):
		print("["+'\033[1;33m'+datetime.now().strftime("%H:%M:%S")+'\033[0m'+"] ["+'\033[1;31m'+"CRITICAL"+'\033[0m'+"] "+text)
 		
 		