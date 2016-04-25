import shutil, glob, os, time

try:
	import numpy
except:
	raise Exception("\n numpy was not found!\n Try to install it first: ' pip install numpy ' ")

def try_load(filename):
	file_to = os.path.basename(filename)
	shutil.copy2(filename, file_to)
	time.sleep(0.1)
	try:
		import interface as bbox
		bbox.get_time()
		succeed = True
	except:
		succeed = False
		os.remove(file_to)

	return succeed

if __name__ == "__main__":
	res_file = ""
	for filename in glob.glob("interface.*"):
		print ("Removing %s. Don't worry!" % filename)
		os.remove(filename)
		
	for filename in glob.glob("lib/*/interface.*"):
		if try_load(filename):
			res_file = filename
			break
		
	if res_file != "":
		shutil.copy2(filename, "simple_bot/.")
		shutil.copy2(filename, "regression_bot/.")
		shutil.copy2(filename, "fast_bot/.")
		shutil.copy2(filename, "checkpoints_greedy_bot/.")
		
		print ("\nYour Python version is supported! Path to the appropriate bbox lib is '%s'." % res_file)
		print ("We copied this lib to all examples folders.")
		print ("==================================== \n\n")
		print ("Now, let's try to run simple_bot: 'cd simple_bot && python bot.py'")
	else:
		print ("\nSorry, your Python version is not supported!")
		print ("Please make sure you have numpy >= 1.10! Your version is %s" % numpy.__version__)
		print ("If upgrading numpy does not help, we recommend you to install Anaconda from https://www.continuum.io/downloads")
		print ("(You can easily uninstall it any time.)")
		