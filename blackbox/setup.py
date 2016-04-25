import shutil, glob, os, time

try:
	import numpy
except:
	raise Exception("\n Numpy was not found!\n Try to install it first: ' pip install numpy ' ")


def try_load(file):
	file_to = os.path.basename(file)
	shutil.copy2(file, file_to)
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

	for file in glob.glob("interface.*"):
		print ("Removing %s. Don't worry!" % file)
		os.remove(file)

	for file in glob.glob("lib/*/interface.*"):
		if try_load(file):
			res_file = file
			break

	if res_file != "":
		shutil.copy2(file, "simple_bot/.")
		shutil.copy2(file, "regression_bot/.")
		shutil.copy2(file, "fast_bot/.")
		shutil.copy2(file, "checkpoints_greedy_bot/.")

		print ("\nYour python version is supported! Path to appropriate bbox lib is '%s'." % res_file)
		print ("We copied this lib to all examples folders.")
		print ("==================================== \n\n")
		print ("Now let's try to run simple_bot: 'cd simple_bot && python bot.py'")

	else:

		print ("\nSorry, but your python version is not supported!")
		print ("Please make sure you have numpy >= 1.10! Your version is %s" % numpy.__version__)
		print ("If upgrading numpy do not help, we recommend you to install Anaconda from here https://www.continuum.io/downloads .")
		print ("(you can easily uninstall it anytime)")
		